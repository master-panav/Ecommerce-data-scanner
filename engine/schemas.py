"""Bronze ingestion: read, pre-clean and normalise any e-commerce dataset for PySpark.

Works with the five public datasets below and with arbitrary retail/clickstream files:
headers are cleaned, placeholder values become NULL, empty rows/columns are dropped,
columns are matched by flexible name aliases, and every conversion is cast-safe
(bad values become NULL instead of crashing under ANSI mode). The output is one
canonical transaction schema used by all Silver/Gold analytics and Spark MLlib models:
    invoice_no, stock_code, description, quantity, invoice_date,
    unit_price, customer_id, country, event_type, category
"""

from __future__ import annotations

import gzip
import re
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from pyspark.sql import Column, DataFrame, SparkSession, functions as F
from pyspark.sql.types import DateType, DoubleType, NumericType, StringType, TimestampType

from engine.io_utils import read_csv, read_parquet

DATASET_CATALOG: Dict[str, Dict[str, str]] = {
    "uci_online_retail": {
        "name": "UCI Online Retail (2010–2011)",
        "rows": "541,909",
        "size": "22.6 MB (.xlsx)",
        "url": "https://archive.ics.uci.edu/dataset/352/online+retail",
        "focus": "Core transactional BI, RFM K-Means, FP-Growth",
    },
    "uci_online_retail_ii": {
        "name": "UCI Online Retail II (2009–2011)",
        "rows": "1,067,371",
        "size": "43.5 MB (.xlsx)",
        "url": "https://archive.ics.uci.edu/ml/datasets/online+retail+II",
        "focus": "2-year longitudinal & cross-dataset comparison",
    },
    "olist_brazil": {
        "name": "Olist Brazilian E-Commerce (9 tables)",
        "rows": "~100,000+ orders",
        "size": "~45 MB (9 CSVs)",
        "url": "https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce",
        "focus": "Distributed multi-table Spark Joins & regional BI",
    },
    "rees46_clickstream": {
        "name": "REES46 Multi-Category Clickstream",
        "rows": "285M events (7 months)",
        "size": "1.6–2.7 GB/month (.csv.gz)",
        "url": "https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store",
        "focus": "Multi-GB clickstream funnel (View -> Cart -> Purchase)",
    },
    "instacart_basket": {
        "name": "Instacart Market Basket Analysis",
        "rows": "3M+ orders / 30M+ items",
        "size": "~700 MB (CSVs)",
        "url": "https://www.kaggle.com/datasets/yasserh/instacart-online-grocery-basket-analysis-dataset",
        "focus": "High-scale Spark MLlib FP-Growth basket mining",
    },
}


class SchemaMappingError(ValueError):
    """Raised when a file lacks the columns needed for retail analytics."""


# --------------------------------------------------------------------------------------
# Flexible column mapping. Names are compared after lower-casing and removing everything
# that is not a letter or digit, so "Invoice No", "invoice-no" and "InvoiceNo" all match.
# --------------------------------------------------------------------------------------
FIELD_ALIASES: Dict[str, Tuple[str, ...]] = {
    "event_type": ("eventtype", "event", "action", "eventname", "interactiontype"),
    "invoice_date": ("invoicedate", "orderdate", "date", "datetime", "timestamp", "transactiondate", "purchasedate",
                     "eventtime", "orderpurchasetimestamp", "time", "createdat", "orderdatetime"),
    "invoice_no": ("invoiceno", "invoice", "invoiceid", "invoicenumber", "orderid", "orderno", "ordernumber", "order",
                   "transactionid", "transactionno", "transaction", "receiptno", "billno"),
    "session": ("usersession", "sessionid", "session"),
    "customer_id": ("customerid", "customeruniqueid", "customer", "customerno", "customernumber", "clientid", "userid",
                    "buyerid", "memberid"),
    "stock_code": ("stockcode", "productid", "productcode", "productno", "sku", "itemid", "itemcode", "asin", "articleno"),
    "quantity": ("quantity", "qty", "units", "unitssold", "quantityordered", "orderquantity"),
    "unit_price": ("unitprice", "price", "priceeach", "itemprice", "sellingprice", "saleprice", "rate", "mrp"),
    "line_total": ("totalprice", "linetotal", "totalamount", "total", "amount", "sales", "revenue", "saleamount", "ordervalue"),
    "description": ("description", "productname", "product", "itemname", "item", "title", "productdescription", "brand"),
    "country": ("country", "countryname", "region", "state", "city", "market", "location"),
    "category": ("category", "categorycode", "productcategory", "categoryname", "department", "productline", "segment"),
}
NULL_TOKENS = ("", "nan", "null", "none", "na", "n/a", "nat", "-", "?")
EVENT_SYNONYMS = {
    "purchase": ("purchase", "buy", "order", "checkout", "transaction", "sale"),
    "cart": ("cart", "addtocart", "add_to_cart", "add-to-cart"),
    "remove_from_cart": ("remove_from_cart", "removefromcart", "remove"),
    "view": ("view", "pageview", "page_view", "detail", "click"),
}
TIMESTAMP_FORMATS = (
    "yyyy-MM-dd HH:mm:ss", "yyyy-MM-dd HH:mm", "yyyy-MM-dd", "yyyy/MM/dd HH:mm:ss", "yyyy/MM/dd",
    "M/d/yyyy H:mm:ss", "M/d/yyyy H:mm", "M/d/yyyy", "M/d/yy H:mm", "M/d/yy",
    "d/M/yyyy H:mm:ss", "d/M/yyyy H:mm", "d/M/yyyy", "d-M-yyyy H:mm", "d-M-yyyy", "dd.MM.yyyy HH:mm", "dd.MM.yyyy",
)
_NUMBER_RE = r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$"


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def detect_schema_type(columns: list[str]) -> str:
    """Label the source schema (for display only; the mapping itself is generic)."""
    raw = {c.strip().lower() for c in columns}
    keys = {_norm(c) for c in columns}
    if {"invoice_no", "unit_price", "customer_id"} <= raw:
        return "canonical"
    if {"eventtype", "usersession", "productid"} <= keys:
        return "rees46_clickstream"
    if keys & set(FIELD_ALIASES["event_type"]):
        return "clickstream"
    if {"orderid", "productid", "addtocartorder"} <= keys:
        return "instacart_basket"
    if {"orderid", "orderitemid", "price"} <= keys:
        return "olist_brazil"
    if {"invoiceno", "unitprice", "customerid"} <= keys:
        return "uci_online_retail"
    if {"invoice", "price", "customerid"} <= keys:
        return "uci_online_retail_ii"
    return "generic_retail"


def map_columns(df: DataFrame) -> Dict[str, str]:
    """Match each canonical field to at most one source column: exact alias first, then containment."""
    keys: Dict[str, str] = {}
    for c in df.columns:
        keys.setdefault(_norm(c), c)
    used: set[str] = set()
    mapping: Dict[str, str] = {}
    for field, aliases in FIELD_ALIASES.items():
        hit = next((keys[a] for a in aliases if a in keys and keys[a] not in used), None)
        if hit is None:  # e.g. "Order Date (UTC)" -> "orderdateutc" contains "orderdate"
            hit = next((c for a in aliases if len(a) >= 5 for k, c in keys.items() if a in k and c not in used), None)
        if hit is not None:
            mapping[field] = hit
            used.add(hit)
    if "invoice_date" not in mapping:  # fall back to the first real date/time column
        for f in df.schema.fields:
            if isinstance(f.dataType, (TimestampType, DateType)) and f.name not in used:
                mapping["invoice_date"] = f.name
                break
    return mapping


# --------------------------------------------------------------------------------------
# Cast-safe conversions: malformed values become NULL instead of failing the job.
# Values are validated (regex / range guard) before casting, so this also holds with ANSI mode on.
# --------------------------------------------------------------------------------------
def _str(col: Column) -> Column:
    return F.trim(col.cast(StringType()))


def _id(col: Column) -> Column:
    """IDs read as floats ("536365.0") are turned back into "536365" so they group correctly."""
    return F.regexp_replace(_str(col), r"^(-?\d+)\.0+$", "$1")


def _num(col: Column, dtype) -> Column:
    if isinstance(dtype, NumericType):
        return col.cast(DoubleType())
    s = _str(col)
    s = F.regexp_replace(s, r"^(-?\d+),(\d{1,2})$", "$1.$2")  # decimal comma: "12,50"
    s = F.regexp_replace(s, r"[^0-9eE+\-.]", "")  # currency symbols, thousands separators, spaces
    return F.when(s.rlike(_NUMBER_RE), s).cast(DoubleType())


def _ts(col: Column, dtype) -> Column:
    if isinstance(dtype, (TimestampType, DateType)):
        return col.cast(TimestampType())
    s = F.regexp_replace(_str(col), r"\s*(UTC|GMT|Z)$", "")
    s = F.regexp_replace(s, r"(\d)T(\d)", "$1 $2")
    return F.coalesce(F.try_to_timestamp(s), *[F.try_to_timestamp(s, F.lit(fmt)) for fmt in TIMESTAMP_FORMATS])


# --------------------------------------------------------------------------------------
# Bronze pre-cleaning (runs before any mapping or analytics)
# --------------------------------------------------------------------------------------
def _clean_headers(columns: list[str]) -> list[str]:
    seen: Dict[str, int] = {}
    out = []
    for i, c in enumerate(columns):
        name = re.sub(r"\s+", " ", str(c).replace("﻿", "")).strip().replace(".", "_").replace("`", "")
        name = name or f"column_{i + 1}"
        seen[name] = seen.get(name, 0) + 1
        out.append(name if seen[name] == 1 else f"{name}_{seen[name]}")
    return out


def clean_bronze(df: DataFrame, profile: bool = True) -> Tuple[DataFrame, Dict[str, object]]:
    """Fix headers, trim text, turn placeholders ("", "NaN", "null", "N/A"...) into NULL and
    drop fully empty columns and rows. Returns the cleaned frame and what was removed."""
    df = df.toDF(*_clean_headers(df.columns))
    string_cols = {f.name for f in df.schema.fields if isinstance(f.dataType, StringType)}
    df = df.select([
        F.when(F.lower(F.trim(F.col(f"`{c}`"))).isin(*NULL_TOKENS), None).otherwise(F.trim(F.col(f"`{c}`"))).alias(c)
        if c in string_cols else F.col(f"`{c}`")
        for c in df.columns
    ])
    stats: Dict[str, object] = {"empty_rows": 0, "empty_columns": []}
    if profile:
        all_null = F.lit(True)
        for c in df.columns:
            all_null = all_null & F.col(f"`{c}`").isNull()
        row = df.agg(
            F.sum(F.when(all_null, 1).otherwise(0)).alias("__empty_rows"),
            *[F.count(F.col(f"`{c}`")).alias(f"__nn_{i}") for i, c in enumerate(df.columns)],
        ).collect()[0]
        stats["empty_rows"] = int(row["__empty_rows"] or 0)
        stats["empty_columns"] = [c for i, c in enumerate(df.columns) if row[f"__nn_{i}"] == 0]
        if stats["empty_columns"]:
            df = df.drop(*stats["empty_columns"])
    return df.dropna(how="all"), stats


# --------------------------------------------------------------------------------------
# Readers
# --------------------------------------------------------------------------------------
def _read_excel_to_spark(spark: SparkSession, file_path: Path) -> DataFrame:
    """Load every non-empty sheet; mixed text/number columns become strings with real NULLs."""
    sheets = pd.read_excel(file_path, sheet_name=None, engine="openpyxl")
    frames = [df for df in sheets.values() if not df.empty]
    if not frames:
        raise SchemaMappingError(f"`{file_path.name}` has no non-empty sheets.")
    pdf = pd.concat(frames, ignore_index=True)
    pdf.columns = [str(c) for c in pdf.columns]
    pdf = pdf.loc[:, ~pdf.columns.str.startswith("Unnamed")]
    for col in pdf.columns:
        series = pdf[col]
        if not (pd.api.types.is_numeric_dtype(series) or pd.api.types.is_datetime64_any_dtype(series)):
            pdf[col] = series.map(lambda v: None if pd.isna(v) else str(v)).astype(object)
    return spark.createDataFrame(pdf)


def _detect_text_encoding(path: Path, sample_bytes: int = 4 * 1024 * 1024) -> str:
    """Return UTF-8 when the file decodes cleanly, else ISO-8859-1 (UCI CSV exports use Latin-1)."""
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    try:
        with opener(path, "rb") as fh:
            sample = fh.read(sample_bytes)
    except OSError:
        return "UTF-8"
    try:
        sample.decode("utf-8")
    except UnicodeDecodeError as exc:
        if exc.start < len(sample) - 4:  # a multi-byte char cut off at the end is still valid UTF-8
            return "ISO-8859-1"
    return "UTF-8"


def _csv(spark: SparkSession, path: Path) -> DataFrame:
    return clean_bronze(read_csv(spark, path, encoding=_detect_text_encoding(path)), profile=False)[0]


def load_raw_dataset(spark: SparkSession, path: str | Path) -> Tuple[DataFrame, str, Dict[str, object]]:
    """Read any supported source, pre-clean it and map it to the canonical schema.

    Returns (canonical_df, schema_label, prep_report); prep_report holds the column mapping
    and the empty rows/columns removed before processing.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Bronze source not found: {p}")

    if p.is_dir() and (p / "olist_orders_dataset.csv").exists():
        return _load_olist_folder(spark, p), "olist_brazil", {"column_mapping": {"source": "Olist 4-table join"}}
    if p.is_dir() and (p / "orders.csv").exists() and (p / "products.csv").exists():
        return _load_instacart_folder(spark, p), "instacart_basket", {"column_mapping": {"source": "Instacart 3-table join"}}

    if p.is_dir():
        raw_df = read_parquet(spark, p) if any(p.rglob("*.parquet")) else read_csv(spark, p)
    else:
        suffix = p.suffix.lower()
        if suffix == ".xlsx":
            raw_df = _read_excel_to_spark(spark, p)
        elif suffix == ".parquet":
            raw_df = read_parquet(spark, p)
        elif suffix in {".csv", ".gz", ".txt"}:
            raw_df = read_csv(spark, p, encoding=_detect_text_encoding(p))
        else:
            raise SchemaMappingError(f"Unsupported file type `{suffix}`. Use .csv, .csv.gz, .xlsx or .parquet.")

    clean_df, stats = clean_bronze(raw_df)
    if not clean_df.columns:
        raise SchemaMappingError(f"`{p.name}` has no non-empty columns.")
    mapping = map_columns(clean_df)
    _validate_mapping(mapping, clean_df.columns, p.name)
    return normalize_dataframe(clean_df, mapping), detect_schema_type(clean_df.columns), {"column_mapping": mapping, **stats}


def _validate_mapping(mapping: Dict[str, str], columns: list[str], source_name: str) -> None:
    missing = []
    if "invoice_date" not in mapping:
        missing.append("a date/time column (e.g. InvoiceDate, order_date, timestamp)")
    if "unit_price" not in mapping and "line_total" not in mapping:
        missing.append("a price or amount column (e.g. UnitPrice, price, total, sales)")
    if missing:
        raise SchemaMappingError(
            f"Could not recognise the columns in `{source_name}`. Missing {' and '.join(missing)}. "
            f"Found columns: {', '.join(columns[:25])}."
        )


# --------------------------------------------------------------------------------------
# Canonical normalisation
# --------------------------------------------------------------------------------------
def normalize_dataframe(df: DataFrame, mapping: Dict[str, str]) -> DataFrame:
    """Map any single-table retail or clickstream source into the canonical transaction schema."""
    dtypes = {f.name: f.dataType for f in df.schema.fields}

    def col(field: str) -> Column:
        return F.col(f"`{mapping[field]}`")

    def dtype(field: str):
        return dtypes.get(mapping.get(field))

    ts = _ts(col("invoice_date"), dtype("invoice_date"))
    customer = _id(col("customer_id")) if "customer_id" in mapping else F.lit(None).cast(StringType())
    stock = _id(col("stock_code")) if "stock_code" in mapping else F.lit(None).cast(StringType())

    # Basket key: invoice column, else the clickstream session, else customer + timestamp
    invoice_cols = [_id(col(f)) for f in ("invoice_no", "session") if f in mapping]
    invoice = F.coalesce(*invoice_cols) if invoice_cols else F.concat_ws("_", customer, ts.cast(StringType()))

    qty = _num(col("quantity"), dtype("quantity")) if "quantity" in mapping else F.lit(1.0)
    qty_int = F.when(F.abs(qty) < 2_000_000_000, F.round(qty)).cast("int")
    price = _num(col("unit_price"), dtype("unit_price")) if "unit_price" in mapping else F.lit(None).cast(DoubleType())
    if "line_total" in mapping:  # derive the unit price from a line total when no price is given
        total = _num(col("line_total"), dtype("line_total"))
        price = F.coalesce(price, F.when(qty_int != 0, total / F.abs(qty_int)))

    if "event_type" in mapping:
        raw_event = F.lower(_str(col("event_type")))
        event = raw_event
        for canonical, words in EVENT_SYNONYMS.items():
            event = F.when(raw_event.isin(*words), F.lit(canonical)).otherwise(event)
    else:
        event = F.lit("purchase")

    text = {f: _str(col(f)) for f in ("description", "country", "category") if f in mapping}
    desc_parts = [text[f] for f in ("description", "category") if f in text]
    return df.select(
        invoice.alias("invoice_no"),
        stock.alias("stock_code"),
        F.coalesce(*desc_parts, stock, F.lit("Unknown")).alias("description"),
        qty_int.alias("quantity"),
        ts.alias("invoice_date"),
        price.alias("unit_price"),
        customer.alias("customer_id"),
        F.coalesce(text.get("country", F.lit(None)), F.lit("Unknown")).alias("country"),
        event.alias("event_type"),
        F.coalesce(text.get("category", F.lit(None)), F.lit("uncategorized")).alias("category"),
    )


# --------------------------------------------------------------------------------------
# Multi-table sources (distributed joins)
# --------------------------------------------------------------------------------------
def _load_olist_folder(spark: SparkSession, folder: Path) -> DataFrame:
    """Join Olist's relational CSV tables (broadcasting the small product table)."""
    orders = _csv(spark, folder / "olist_orders_dataset.csv")
    items = _csv(spark, folder / "olist_order_items_dataset.csv")
    customers = _csv(spark, folder / "olist_customers_dataset.csv")
    products = _csv(spark, folder / "olist_products_dataset.csv")

    joined = (
        items.join(orders, on="order_id", how="inner")
        .join(customers, on="customer_id", how="left")
        .join(F.broadcast(products), on="product_id", how="left")
    )
    dtypes = {f.name: f.dataType for f in joined.schema.fields}
    is_canceled = F.lower(_str(F.col("order_status"))).isin("canceled", "unavailable")
    order_id = _str(F.col("order_id"))
    category = _str(F.col("product_category_name"))
    return joined.select(
        F.when(is_canceled, F.concat(F.lit("C"), order_id)).otherwise(order_id).alias("invoice_no"),
        _str(F.col("product_id")).alias("stock_code"),
        F.coalesce(category, _str(F.col("product_id"))).alias("description"),
        F.when(is_canceled, F.lit(-1)).otherwise(F.lit(1)).alias("quantity"),
        _ts(F.col("order_purchase_timestamp"), dtypes["order_purchase_timestamp"]).alias("invoice_date"),
        _num(F.col("price"), dtypes["price"]).alias("unit_price"),
        _str(F.col("customer_unique_id")).alias("customer_id"),
        F.concat(F.lit("Brazil - "), F.coalesce(_str(F.col("customer_state")), F.lit("UNK"))).alias("country"),
        F.lit("purchase").alias("event_type"),
        F.coalesce(category, F.lit("general")).alias("category"),
    )


def _load_instacart_folder(spark: SparkSession, folder: Path) -> DataFrame:
    """Join Instacart basket tables (broadcasting products and departments)."""
    orders = _csv(spark, folder / "orders.csv")
    prior_path = folder / "order_products__prior.csv"
    items = _csv(spark, prior_path if prior_path.exists() else folder / "order_products__train.csv")
    products = _csv(spark, folder / "products.csv")

    depts_path = folder / "departments.csv"
    if depts_path.exists():
        products = products.join(F.broadcast(_csv(spark, depts_path)), on="department_id", how="left")
        category = F.coalesce(_str(F.col("department")), F.lit("grocery"))
    else:
        category = F.lit("grocery")

    joined = items.join(orders, on="order_id", how="inner").join(F.broadcast(products), on="product_id", how="left")

    # Instacart only gives a relative day/hour; place orders on a reference timeline
    synthetic_ts = F.expr(
        "timestampadd(HOUR, try_cast(order_hour_of_day as int), "
        "timestampadd(DAY, try_cast(order_number as int) * 7 + try_cast(order_dow as int), timestamp'2024-01-01 00:00:00'))"
    )
    return joined.select(
        _str(F.col("order_id")).alias("invoice_no"),
        _str(F.col("product_id")).alias("stock_code"),
        F.coalesce(_str(F.col("product_name")), _str(F.col("product_id"))).alias("description"),
        F.lit(1).alias("quantity"),
        synthetic_ts.alias("invoice_date"),
        F.lit(4.50).alias("unit_price"),
        _str(F.col("user_id")).alias("customer_id"),
        F.lit("United States").alias("country"),
        F.lit("purchase").alias("event_type"),
        category.alias("category"),
    )
