"""Bronze ingestion: read, pre-clean and normalise any e-commerce dataset for PySpark.

Works with the public datasets in DATASET_CATALOG and with arbitrary retail, supply-chain or
clickstream files (including wide 20-50+ column exports):
  * headers are cleaned, placeholder values become NULL, empty rows/columns are dropped;
  * every column is profiled (type, nulls, distinct values);
  * canonical fields are matched by flexible name aliases, with a value-based fallback for dates;
  * every conversion is cast-safe (bad values become NULL instead of crashing under ANSI mode);
  * all columns that are not mapped are kept as `x_<name>` attributes for slice-and-dice BI.

Canonical transaction schema used by the Silver/Gold layers and Spark MLlib models:
    invoice_no, stock_code, description, quantity, invoice_date, unit_price, customer_id,
    country, event_type, category, order_status, profit, discount, source_total, x_*
"""

from __future__ import annotations

import gzip
import re
from pathlib import Path
from typing import Dict, List, Tuple

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
        "focus": "Distributed multi-table Spark Joins, delivery & review BI",
    },
    "dataco_supply_chain": {
        "name": "DataCo Smart Supply Chain",
        "rows": "~180,000 order items, 53 columns",
        "size": "~95 MB (.csv)",
        "url": "https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis",
        "focus": "Wide-table BI: profit, shipping delays, markets & segments",
    },
    "pakistan_ecommerce": {
        "name": "Pakistan's Largest E-Commerce Dataset",
        "rows": "~580,000 orders",
        "size": "~100 MB (.csv)",
        "url": "https://www.kaggle.com/datasets/zusmani/pakistans-largest-ecommerce-dataset",
        "focus": "Order-status cancellations/refunds, payment methods",
    },
    "superstore_sales": {
        "name": "SuperStore Sales",
        "rows": "~50,000 order lines",
        "size": "~8 MB (.csv)",
        "url": "https://www.kaggle.com/datasets/laibaanwer/superstore-sales-dataset",
        "focus": "Profit & discount analysis by segment, market, ship mode",
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
# that is not a letter or digit, so "Order Item Quantity", "order_item_quantity" and
# "OrderItemQuantity" all match. Aliases are listed in priority order.
# --------------------------------------------------------------------------------------
FIELD_ALIASES: Dict[str, Tuple[str, ...]] = {
    "event_type": ("eventtype", "event", "action", "eventname", "interactiontype"),
    "invoice_date": ("invoicedate", "orderdate", "orderdatedateorders", "orderpurchasetimestamp", "purchasedate",
                     "transactiondate", "createdat", "orderdatetime", "eventtime", "date", "datetime", "timestamp", "time"),
    "invoice_no": ("invoiceno", "invoice", "invoiceid", "invoicenumber", "orderid", "orderno", "ordernumber",
                   "incrementid", "transactionid", "transactionno", "receiptno", "billno"),
    "session": ("usersession", "sessionid", "session"),
    "customer_id": ("customerid", "customeruniqueid", "ordercustomerid", "customerno", "customernumber", "clientid",
                    "userid", "buyerid", "memberid", "customername", "customer"),
    "stock_code": ("stockcode", "productid", "productcardid", "productcode", "productno", "sku", "itemid", "itemcode",
                   "asin", "articleno"),
    "quantity": ("quantity", "qty", "qtyordered", "orderitemquantity", "quantityordered", "orderquantity", "units",
                 "unitssold"),
    "unit_price": ("unitprice", "price", "orderitemproductprice", "productprice", "priceeach", "itemprice",
                   "sellingprice", "saleprice", "mrp"),
    "line_total": ("orderitemtotal", "linetotal", "totalprice", "grandtotal", "totalamount", "total", "amount", "sales",
                   "revenue", "saleamount", "ordervalue"),
    "profit": ("profit", "orderprofitperorder", "benefitperorder", "netprofit", "grossprofit"),
    "discount": ("discount", "discountamount", "orderitemdiscount", "discountvalue"),
    "status": ("orderstatus", "status", "deliverystatus"),
    "description": ("description", "productname", "itemname", "itemdescription", "product", "item", "title", "brand"),
    "country": ("country", "ordercountry", "countryname", "customercountry", "region", "orderregion", "state", "city",
                "market", "location"),
    "category": ("category", "categoryname", "productcategory", "productcategoryname", "categorycode", "department",
                 "departmentname", "productline", "segment"),
}
CANCEL_STATUS_RE = r"cancel|refund|return|reject|void"
NULL_TOKENS = ("", "nan", "null", "none", "na", "n/a", "nat", "-", "?", "#n/a")
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
    "d-MMM-yyyy", "d-MMM-yy", "d MMM yyyy", "MMM d, yyyy", "MMMM d, yyyy",
)
_NUMBER_RE = r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$"
MAX_EXTRA_COLUMNS = 80


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _snake(name: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "_", str(name)).strip("_").lower() or "col"


def _c(name: str) -> Column:
    return F.col(f"`{name}`")


def detect_schema_type(columns: List[str]) -> str:
    """Label the source schema (for display only; the mapping itself is generic)."""
    raw = {c.strip().lower() for c in columns}
    keys = {_norm(c) for c in columns}
    if {"invoice_no", "unit_price", "customer_id"} <= raw:
        return "canonical"
    if {"eventtype", "usersession", "productid"} <= keys:
        return "rees46_clickstream"
    if keys & set(FIELD_ALIASES["event_type"]):
        return "clickstream"
    if {"orderitemquantity", "deliverystatus"} <= keys:
        return "dataco_supply_chain"
    if {"incrementid", "qtyordered"} <= keys:
        return "pakistan_ecommerce"
    if {"subcategory", "profit"} <= keys and ("shipmode" in keys or "segment" in keys):
        return "superstore_sales"
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
    """Match each canonical field to at most one source column.

    Pass 1: exact alias match. Pass 2: a column name that starts or ends with an alias
    (e.g. "order date (DateOrders)" starts with "orderdate"). Pass 3 (dates only): the
    column whose values parse as dates most often.
    """
    keys: Dict[str, str] = {}
    for c in df.columns:
        keys.setdefault(_norm(c), c)
    for c in df.columns:  # "Price2" (Spark's rename of a duplicate header), "category_name_1"
        keys.setdefault(_norm(c).rstrip("0123456789"), c)
    used: set[str] = set()
    mapping: Dict[str, str] = {}
    for field, aliases in FIELD_ALIASES.items():
        hit = next((keys[a] for a in aliases if a in keys and keys[a] not in used), None)
        if hit is None:
            hit = next(
                (c for a in aliases if len(a) >= 6 for k, c in keys.items()
                 if (k.startswith(a) or k.endswith(a)) and c not in used),
                None,
            )
        if hit is not None:
            mapping[field] = hit
            used.add(hit)
    if "invoice_date" not in mapping:
        guess = _guess_date_column(df, used)
        if guess:
            mapping["invoice_date"] = guess
    return mapping


def _guess_date_column(df: DataFrame, used: set[str]) -> str | None:
    """Pick the column whose sampled values parse as timestamps most often (>= 80%)."""
    fields = [f for f in df.schema.fields if f.name not in used]
    for f in fields:
        if isinstance(f.dataType, (TimestampType, DateType)):
            return f.name
    candidates = [f.name for f in fields if isinstance(f.dataType, StringType)]
    if not candidates:
        return None
    sample = df.select([_c(c) for c in candidates]).limit(500)
    row = sample.agg(*[
        (F.count(_ts(_c(c), StringType())) / F.greatest(F.count(_c(c)), F.lit(1))).alias(f"r{i}")
        for i, c in enumerate(candidates)
    ]).collect()[0]
    best = max(range(len(candidates)), key=lambda i: row[f"r{i}"] or 0)
    return candidates[best] if (row[f"r{best}"] or 0) >= 0.8 else None


def extra_columns(df: DataFrame, mapping: Dict[str, str]) -> Dict[str, str]:
    """Unmapped source columns kept as attributes: {"x_<snake_name>": "Original Name"}."""
    used = set(mapping.values())
    out: Dict[str, str] = {}
    for c in df.columns:
        if c in used or len(out) >= MAX_EXTRA_COLUMNS:
            continue
        base = f"x_{_snake(c)}"
        name, i = base, 2
        while name in out:
            name, i = f"{base}_{i}", i + 1
        out[name] = c
    return out


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
    s = F.regexp_replace(s, r"[^0-9eE+\-.]", "")  # currency symbols, thousands separators, spaces, %
    return F.when(s.rlike(_NUMBER_RE), s).cast(DoubleType())


def _ts(col: Column, dtype) -> Column:
    if isinstance(dtype, (TimestampType, DateType)):
        return col.cast(TimestampType())
    s = F.regexp_replace(_str(col), r"\s*(UTC|GMT|Z)$", "")
    s = F.regexp_replace(s, r"(\d)T(\d)", "$1 $2")
    return F.coalesce(F.try_to_timestamp(s), *[F.try_to_timestamp(s, F.lit(fmt)) for fmt in TIMESTAMP_FORMATS])


# --------------------------------------------------------------------------------------
# Bronze pre-cleaning and column profiling (runs before any mapping or analytics)
# --------------------------------------------------------------------------------------
def _clean_headers(columns: List[str]) -> List[str]:
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
    drop fully empty columns and rows. With profile=True also returns a per-column profile."""
    df = df.toDF(*_clean_headers(df.columns))
    string_cols = {f.name for f in df.schema.fields if isinstance(f.dataType, StringType)}
    df = df.select([
        F.when(F.lower(F.trim(_c(c))).isin(*NULL_TOKENS), None).otherwise(F.trim(_c(c))).alias(c)
        if c in string_cols else _c(c)
        for c in df.columns
    ])
    stats: Dict[str, object] = {"empty_rows": 0, "empty_columns": [], "column_profile": []}
    if profile:
        all_null = F.lit(True)
        for c in df.columns:
            all_null = all_null & _c(c).isNull()
        row = df.agg(
            F.count(F.lit(1)).alias("__rows"),
            F.sum(F.when(all_null, 1).otherwise(0)).alias("__empty_rows"),
            *[F.count(_c(c)).alias(f"__nn_{i}") for i, c in enumerate(df.columns)],
            *[F.approx_count_distinct(_c(c)).alias(f"__nd_{i}") for i, c in enumerate(df.columns)],
        ).collect()[0]
        total = int(row["__rows"] or 0)
        empty_rows = int(row["__empty_rows"] or 0)
        kept_rows = max(total - empty_rows, 1)
        types = {f.name: f.dataType.simpleString() for f in df.schema.fields}
        stats["empty_rows"] = empty_rows
        stats["empty_columns"] = [c for i, c in enumerate(df.columns) if row[f"__nn_{i}"] == 0]
        stats["column_profile"] = [
            {
                "column": c,
                "type": types[c],
                "non_null": int(row[f"__nn_{i}"]),
                "null_pct": round(100 * (1 - row[f"__nn_{i}"] / kept_rows), 2),
                "distinct_approx": int(row[f"__nd_{i}"]),
            }
            for i, c in enumerate(df.columns)
        ]
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
    """Return UTF-8 when the file decodes cleanly, else ISO-8859-1 (UCI and DataCo CSVs use Latin-1)."""
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


def is_multi_table_folder(p: Path) -> bool:
    return p.is_dir() and (
        (p / "olist_orders_dataset.csv").exists() or ((p / "orders.csv").exists() and (p / "products.csv").exists())
    )


def load_raw_dataset(spark: SparkSession, path: str | Path) -> Tuple[DataFrame, str, Dict[str, object]]:
    """Read any supported source, pre-clean it and map it to the canonical schema.

    Returns (canonical_df, schema_label, prep_report); prep_report holds the column mapping,
    the kept extra columns, the per-column profile and the empty rows/columns removed.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Bronze source not found: {p}")

    if p.is_dir() and (p / "olist_orders_dataset.csv").exists():
        df, tables = _load_olist_folder(spark, p)
        return df, "olist_brazil", {"column_mapping": {"source": f"Olist {tables}-table join"}}
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
            raise SchemaMappingError(f"Unsupported file type `{suffix}`. Use .csv, .csv.gz, .xlsx, .parquet or .zip.")

    clean_df, stats = clean_bronze(raw_df)
    if not clean_df.columns:
        raise SchemaMappingError(f"`{p.name}` has no non-empty columns.")
    mapping = map_columns(clean_df)
    _validate_mapping(mapping, clean_df.columns, p.name)
    extras = extra_columns(clean_df, mapping)
    report = {"column_mapping": mapping, "extra_columns": extras, **stats}
    return normalize_dataframe(clean_df, mapping, extras), detect_schema_type(clean_df.columns), report


def _validate_mapping(mapping: Dict[str, str], columns: List[str], source_name: str) -> None:
    missing = []
    if "invoice_date" not in mapping:
        missing.append("a date/time column (e.g. InvoiceDate, order_date, created_at)")
    if "unit_price" not in mapping and "line_total" not in mapping:
        missing.append("a price or amount column (e.g. UnitPrice, price, total, sales)")
    if missing:
        raise SchemaMappingError(
            f"Could not recognise the columns in `{source_name}`. Missing {' and '.join(missing)}. "
            f"Found columns: {', '.join(columns[:30])}."
        )


# --------------------------------------------------------------------------------------
# Canonical normalisation
# --------------------------------------------------------------------------------------
def normalize_dataframe(df: DataFrame, mapping: Dict[str, str], extras: Dict[str, str] | None = None) -> DataFrame:
    """Map any single-table retail, supply-chain or clickstream source into the canonical schema."""
    dtypes = {f.name: f.dataType for f in df.schema.fields}

    def col(field: str) -> Column:
        return _c(mapping[field])

    def dtype(field: str):
        return dtypes.get(mapping.get(field))

    def num(field: str) -> Column:
        return _num(col(field), dtype(field)) if field in mapping else F.lit(None).cast(DoubleType())

    ts = _ts(col("invoice_date"), dtype("invoice_date"))
    customer = _id(col("customer_id")) if "customer_id" in mapping else F.lit(None).cast(StringType())
    stock = _id(col("stock_code")) if "stock_code" in mapping else F.lit(None).cast(StringType())

    # Basket key: invoice column, else the clickstream session, else customer + timestamp
    invoice_cols = [_id(col(f)) for f in ("invoice_no", "session") if f in mapping]
    invoice = F.coalesce(*invoice_cols) if invoice_cols else F.concat_ws("_", customer, ts.cast(StringType()))

    qty = num("quantity") if "quantity" in mapping else F.lit(1.0)
    qty_int = F.when(F.abs(qty) < 2_000_000_000, F.round(qty)).cast("int")
    total = num("line_total")
    price = F.coalesce(num("unit_price"), F.when(qty_int != 0, total / F.abs(qty_int)))

    if "event_type" in mapping:
        raw_event = F.lower(_str(col("event_type")))
        event = raw_event
        for canonical, words in EVENT_SYNONYMS.items():
            event = F.when(raw_event.isin(*words), F.lit(canonical)).otherwise(event)
    else:
        event = F.lit("purchase")

    def text(field: str) -> Column:
        return _str(col(field)) if field in mapping else F.lit(None).cast(StringType())

    return df.select(
        invoice.alias("invoice_no"),
        stock.alias("stock_code"),
        F.coalesce(text("description"), text("category"), stock, F.lit("Unknown")).alias("description"),
        qty_int.alias("quantity"),
        ts.alias("invoice_date"),
        price.alias("unit_price"),
        customer.alias("customer_id"),
        F.coalesce(text("country"), F.lit("Unknown")).alias("country"),
        event.alias("event_type"),
        F.coalesce(text("category"), F.lit("uncategorized")).alias("category"),
        F.lower(text("status")).alias("order_status"),
        num("profit").alias("profit"),
        num("discount").alias("discount"),
        total.alias("source_total"),
        *[_c(src).alias(name) for name, src in (extras or {}).items()],
    )


# --------------------------------------------------------------------------------------
# Multi-table sources (distributed joins)
# --------------------------------------------------------------------------------------
def _load_olist_folder(spark: SparkSession, folder: Path) -> Tuple[DataFrame, int]:
    """Join Olist's relational CSV tables; payments and reviews are joined when present."""
    orders = _csv(spark, folder / "olist_orders_dataset.csv")
    items = _csv(spark, folder / "olist_order_items_dataset.csv")
    customers = _csv(spark, folder / "olist_customers_dataset.csv")
    products = _csv(spark, folder / "olist_products_dataset.csv")
    joined = (
        items.join(orders, on="order_id", how="inner")
        .join(customers, on="customer_id", how="left")
        .join(F.broadcast(products), on="product_id", how="left")
    )
    tables = 4

    payments_path = folder / "olist_order_payments_dataset.csv"
    if payments_path.exists():
        pay = _csv(spark, payments_path)
        pay_dt = {f.name: f.dataType for f in pay.schema.fields}
        joined = joined.join(
            pay.groupBy("order_id").agg(
                F.first(_str(_c("payment_type")), ignorenulls=True).alias("x_payment_type"),
                F.max(_num(_c("payment_installments"), pay_dt.get("payment_installments"))).alias("x_payment_installments"),
            ),
            on="order_id", how="left",
        )
        tables += 1
    reviews_path = folder / "olist_order_reviews_dataset.csv"
    if reviews_path.exists():
        rev = _csv(spark, reviews_path)
        rev_dt = {f.name: f.dataType for f in rev.schema.fields}
        joined = joined.join(
            rev.groupBy("order_id").agg(F.avg(_num(_c("review_score"), rev_dt.get("review_score"))).alias("x_review_score")),
            on="order_id", how="left",
        )
        tables += 1

    dtypes = {f.name: f.dataType for f in joined.schema.fields}
    purchased = _ts(_c("order_purchase_timestamp"), dtypes["order_purchase_timestamp"])
    status = F.lower(_str(_c("order_status")))
    is_canceled = status.isin("canceled", "unavailable")
    order_id = _str(_c("order_id"))
    category = _str(_c("product_category_name")) if "product_category_name" in dtypes else F.lit(None).cast(StringType())

    extras = [F.col(c) for c in ("x_payment_type", "x_payment_installments", "x_review_score") if c in dtypes]
    for src, name in (("customer_city", "x_customer_city"), ("customer_state", "x_customer_state"), ("seller_id", "x_seller_id")):
        if src in dtypes:
            extras.append(_str(_c(src)).alias(name))
    if "freight_value" in dtypes:
        extras.append(_num(_c("freight_value"), dtypes["freight_value"]).alias("x_freight_value"))
    if "order_delivered_customer_date" in dtypes:
        delivered = _ts(_c("order_delivered_customer_date"), dtypes["order_delivered_customer_date"])
        extras.append(F.datediff(delivered, purchased).alias("x_delivery_days"))
        if "order_estimated_delivery_date" in dtypes:
            estimated = _ts(_c("order_estimated_delivery_date"), dtypes["order_estimated_delivery_date"])
            extras.append(F.when(delivered.isNotNull(), (delivered > estimated).cast("int")).alias("x_late_delivery"))

    df = joined.select(
        F.when(is_canceled, F.concat(F.lit("C"), order_id)).otherwise(order_id).alias("invoice_no"),
        _str(_c("product_id")).alias("stock_code"),
        F.coalesce(category, _str(_c("product_id"))).alias("description"),
        F.when(is_canceled, F.lit(-1)).otherwise(F.lit(1)).alias("quantity"),
        purchased.alias("invoice_date"),
        _num(_c("price"), dtypes["price"]).alias("unit_price"),
        _str(_c("customer_unique_id")).alias("customer_id"),
        F.concat(F.lit("Brazil - "), F.coalesce(_str(_c("customer_state")), F.lit("UNK"))).alias("country"),
        F.lit("purchase").alias("event_type"),
        F.coalesce(category, F.lit("general")).alias("category"),
        status.alias("order_status"),
        F.lit(None).cast(DoubleType()).alias("profit"),
        F.lit(None).cast(DoubleType()).alias("discount"),
        F.lit(None).cast(DoubleType()).alias("source_total"),
        *extras,
    )
    return df, tables


def _load_instacart_folder(spark: SparkSession, folder: Path) -> DataFrame:
    """Join Instacart basket tables (broadcasting products and departments)."""
    orders = _csv(spark, folder / "orders.csv")
    prior_path = folder / "order_products__prior.csv"
    items = _csv(spark, prior_path if prior_path.exists() else folder / "order_products__train.csv")
    products = _csv(spark, folder / "products.csv")

    depts_path = folder / "departments.csv"
    if depts_path.exists():
        products = products.join(F.broadcast(_csv(spark, depts_path)), on="department_id", how="left")
        category = F.coalesce(_str(_c("department")), F.lit("grocery"))
    else:
        category = F.lit("grocery")

    joined = items.join(orders, on="order_id", how="inner").join(F.broadcast(products), on="product_id", how="left")

    # Instacart only gives a relative day/hour; place orders on a reference timeline
    synthetic_ts = F.expr(
        "timestampadd(HOUR, try_cast(order_hour_of_day as int), "
        "timestampadd(DAY, try_cast(order_number as int) * 7 + try_cast(order_dow as int), timestamp'2024-01-01 00:00:00'))"
    )
    return joined.select(
        _str(_c("order_id")).alias("invoice_no"),
        _str(_c("product_id")).alias("stock_code"),
        F.coalesce(_str(_c("product_name")), _str(_c("product_id"))).alias("description"),
        F.lit(1).alias("quantity"),
        synthetic_ts.alias("invoice_date"),
        F.lit(4.50).alias("unit_price"),
        _str(_c("user_id")).alias("customer_id"),
        F.lit("United States").alias("country"),
        F.lit("purchase").alias("event_type"),
        category.alias("category"),
        F.lit(None).cast(StringType()).alias("order_status"),
        F.lit(None).cast(DoubleType()).alias("profit"),
        F.lit(None).cast(DoubleType()).alias("discount"),
        F.lit(None).cast(DoubleType()).alias("source_total"),
    )
