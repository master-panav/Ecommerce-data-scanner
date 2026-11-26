"""Multi-dataset schema detection and normalization adapters for PySpark.

Normalizes five public Big Data e-commerce schemas into a unified canonical
transaction schema so downstream Silver/Gold analytics and Spark MLlib models
work consistently across datasets:
    invoice_no, stock_code, description, quantity, invoice_date,
    unit_price, customer_id, country, event_type, category
"""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from pyspark.sql import DataFrame, SparkSession, functions as F
from pyspark.sql.types import DoubleType, IntegerType, StringType

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


def detect_schema_type(columns: list[str]) -> str:
    """Identify which supported Big Data schema a file matches."""
    cols = {c.strip().lower() for c in columns}
    if {"event_type", "user_session", "product_id"}.issubset(cols):
        return "rees46_clickstream"
    if {"order_id", "product_id", "add_to_cart_order"}.issubset(cols):
        return "instacart_basket"
    if {"order_id", "order_item_id", "price"}.issubset(cols):
        return "olist_brazil"
    if {"invoice", "price", "customer id"}.issubset(cols):
        return "uci_online_retail_ii"
    if {"invoiceno", "unitprice", "customerid"}.issubset(cols):
        return "uci_online_retail"
    if {"invoice_no", "unit_price", "customer_id"}.issubset(cols):
        return "canonical"
    return "generic_retail"


class SchemaMappingError(ValueError):
    """Raised when an uploaded file lacks the columns needed for retail analytics."""


# Accepted source column names (lower-cased) for each canonical field of a retail file.
RETAIL_COLUMN_ALIASES: Dict[str, Tuple[str, ...]] = {
    "invoice_no": ("invoiceno", "invoice", "invoice_no", "invoice_id", "transactionno", "transaction_id", "order_id", "orderid"),
    "stock_code": ("stockcode", "stock_code", "productno", "product_id", "productid", "sku"),
    "description": ("description", "product", "product_name", "productname", "item", "item_name"),
    "quantity": ("quantity", "qty", "units"),
    "invoice_date": ("invoicedate", "invoice_date", "date", "order_date", "orderdate", "transaction_date", "timestamp"),
    "unit_price": ("unitprice", "price", "unit_price", "item_price"),
    "customer_id": ("customerid", "customer id", "customer_id", "customerno", "user_id"),
    "country": ("country", "region", "state"),
    "category": ("category", "category_code", "department"),
}
REQUIRED_RETAIL_FIELDS = ("invoice_no", "invoice_date", "unit_price")


def _read_excel_to_spark(spark: SparkSession, file_path: Path) -> DataFrame:
    """Load single or multi-sheet Excel workbooks into a Spark DataFrame.

    Text/mixed columns (e.g. UCI ``InvoiceNo`` holding both 536365 and "C536379")
    are converted to clean strings with real nulls, so Arrow can ship them to Spark.
    """
    sheets = pd.read_excel(file_path, sheet_name=None, engine="openpyxl")
    frames = [df for df in sheets.values() if not df.empty]
    if not frames:
        raise SchemaMappingError(f"`{file_path.name}` has no non-empty sheets.")
    pdf = pd.concat(frames, ignore_index=True)
    pdf.columns = [str(c).strip() for c in pdf.columns]
    pdf = pdf.loc[:, ~pdf.columns.str.startswith("Unnamed")]

    for col in pdf.columns:
        series = pdf[col]
        if pd.api.types.is_datetime64_any_dtype(series) or pd.api.types.is_bool_dtype(series):
            continue
        if pd.api.types.is_numeric_dtype(series):
            continue
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
        # A multi-byte char cut off at the end of the sample is still valid UTF-8.
        if exc.start < len(sample) - 4:
            return "ISO-8859-1"
    return "UTF-8"


def load_raw_dataset(spark: SparkSession, path: str | Path) -> Tuple[DataFrame, str]:
    """Read a CSV, CSV.GZ, Parquet, Excel file, or multi-table folder into Spark.

    Returns (normalized_spark_df, detected_schema_key).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Bronze source not found: {p}")

    # Multi-table directory support (Olist or Instacart folders)
    if p.is_dir():
        if (p / "olist_orders_dataset.csv").exists():
            return _load_olist_folder(spark, p), "olist_brazil"
        if (p / "orders.csv").exists() and (p / "products.csv").exists():
            return _load_instacart_folder(spark, p), "instacart_basket"
        # Otherwise treat directory as Parquet or multi-CSV partition folder
        if any(p.rglob("*.parquet")):
            raw_df = read_parquet(spark, p)
        else:
            raw_df = read_csv(spark, p)
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

    schema_type = detect_schema_type(raw_df.columns)
    if schema_type in {"canonical", "generic_retail"}:
        _validate_retail_columns(raw_df.columns, p.name)
    return normalize_dataframe(raw_df, schema_type), schema_type


def _validate_retail_columns(columns: list[str], source_name: str) -> None:
    cols = {c.strip().lower() for c in columns}
    missing = [f for f in REQUIRED_RETAIL_FIELDS if not cols.intersection(RETAIL_COLUMN_ALIASES[f])]
    if missing:
        hints = "; ".join(f"**{f}** (any of: {', '.join(RETAIL_COLUMN_ALIASES[f][:5])})" for f in missing)
        raise SchemaMappingError(
            f"Could not recognise the columns in `{source_name}`. Missing {hints}. "
            f"Found columns: {', '.join(columns[:25])}."
        )


def _csv(spark: SparkSession, path: Path) -> DataFrame:
    return read_csv(spark, path, encoding=_detect_text_encoding(path))


def _load_olist_folder(spark: SparkSession, folder: Path) -> DataFrame:
    """Perform distributed Spark Joins across Olist's relational CSV tables."""
    orders = _csv(spark, folder / "olist_orders_dataset.csv")
    items = _csv(spark, folder / "olist_order_items_dataset.csv")
    customers = _csv(spark, folder / "olist_customers_dataset.csv")
    products = _csv(spark, folder / "olist_products_dataset.csv")

    joined = (
        items.join(orders, on="order_id", how="inner")
        .join(customers, on="customer_id", how="left")
        .join(F.broadcast(products), on="product_id", how="left")
    )

    is_canceled = F.col("order_status").isin("canceled", "unavailable")
    return joined.select(
        F.when(is_canceled, F.concat(F.lit("C"), F.col("order_id").cast(StringType())))
        .otherwise(F.col("order_id").cast(StringType()))
        .alias("invoice_no"),
        F.col("product_id").cast(StringType()).alias("stock_code"),
        F.coalesce(F.col("product_category_name"), F.col("product_id")).cast(StringType()).alias("description"),
        F.when(is_canceled, F.lit(-1)).otherwise(F.lit(1)).cast(IntegerType()).alias("quantity"),
        F.to_timestamp(F.col("order_purchase_timestamp")).alias("invoice_date"),
        F.col("price").cast(DoubleType()).alias("unit_price"),
        F.col("customer_unique_id").cast(StringType()).alias("customer_id"),
        F.concat(F.lit("Brazil - "), F.coalesce(F.col("customer_state"), F.lit("UNK"))).alias("country"),
        F.lit("purchase").alias("event_type"),
        F.coalesce(F.col("product_category_name"), F.lit("general")).cast(StringType()).alias("category"),
    )


def _load_instacart_folder(spark: SparkSession, folder: Path) -> DataFrame:
    """Perform distributed Spark Joins across Instacart basket tables."""
    orders = _csv(spark, folder / "orders.csv")
    prior_path = folder / "order_products__prior.csv"
    items_path = prior_path if prior_path.exists() else folder / "order_products__train.csv"
    items = _csv(spark, items_path)
    products = _csv(spark, folder / "products.csv")

    depts_path = folder / "departments.csv"
    if depts_path.exists():
        depts = _csv(spark, depts_path)
        products = products.join(F.broadcast(depts), on="department_id", how="left")
        cat_col = F.coalesce(F.col("department"), F.lit("grocery"))
    else:
        cat_col = F.lit("grocery")

    joined = items.join(orders, on="order_id", how="inner").join(F.broadcast(products), on="product_id", how="left")

    # Instacart provides relative order_dow + order_hour_of_day; map onto a clean reference timeline
    synthetic_ts = F.expr(
        "timestampadd(HOUR, cast(order_hour_of_day as int), "
        "timestampadd(DAY, cast(order_number as int) * 7 + cast(order_dow as int), timestamp'2024-01-01 00:00:00'))"
    )
    return joined.select(
        F.col("order_id").cast(StringType()).alias("invoice_no"),
        F.col("product_id").cast(StringType()).alias("stock_code"),
        F.coalesce(F.col("product_name"), F.col("product_id")).cast(StringType()).alias("description"),
        F.lit(1).cast(IntegerType()).alias("quantity"),
        synthetic_ts.alias("invoice_date"),
        F.lit(4.50).cast(DoubleType()).alias("unit_price"),
        F.col("user_id").cast(StringType()).alias("customer_id"),
        F.lit("United States").alias("country"),
        F.lit("purchase").alias("event_type"),
        cat_col.cast(StringType()).alias("category"),
    )


def normalize_dataframe(df: DataFrame, schema_type: str) -> DataFrame:
    """Map single-file datasets into the canonical Big Data transaction schema."""
    col_map = {c.strip().lower(): c for c in df.columns}

    def pick(*names: str, default=None):
        for n in names:
            if n in col_map:
                return F.col(f"`{col_map[n]}`")
        return F.lit(default)

    if schema_type == "rees46_clickstream":
        return df.select(
            F.coalesce(pick("user_session"), F.concat(pick("user_id"), F.lit("_"), pick("event_time")))
            .cast(StringType()).alias("invoice_no"),
            pick("product_id").cast(StringType()).alias("stock_code"),
            F.coalesce(pick("brand"), pick("category_code"), pick("product_id")).cast(StringType()).alias("description"),
            F.lit(1).cast(IntegerType()).alias("quantity"),
            F.to_timestamp(F.regexp_replace(pick("event_time").cast(StringType()), " UTC$", "")).alias("invoice_date"),
            pick("price").cast(DoubleType()).alias("unit_price"),
            pick("user_id").cast(StringType()).alias("customer_id"),
            F.lit("Global").alias("country"),
            F.lower(pick("event_type", default="purchase").cast(StringType())).alias("event_type"),
            F.coalesce(pick("category_code"), F.lit("uncategorized")).cast(StringType()).alias("category"),
        )

    # UCI Online Retail I, UCI Online Retail II, canonical, or generic retail CSV
    a = RETAIL_COLUMN_ALIASES
    raw_date = pick(*a["invoice_date"]).cast(StringType())
    parsed_date = F.coalesce(
        F.to_timestamp(raw_date),
        F.to_timestamp(raw_date, "M/d/yyyy H:mm"),
        F.to_timestamp(raw_date, "d/M/yyyy H:mm"),
        F.to_timestamp(raw_date, "M/d/yyyy H:mm:ss"),
        F.to_timestamp(raw_date, "d-M-yyyy H:mm"),
        F.to_timestamp(raw_date, "yyyy-MM-dd HH:mm:ss"),
        F.to_timestamp(raw_date, "yyyy-MM-dd"),
        F.to_timestamp(raw_date, "M/d/yyyy"),
        F.to_timestamp(raw_date, "d/M/yyyy"),
    )

    return df.select(
        F.trim(pick(*a["invoice_no"]).cast(StringType())).alias("invoice_no"),
        F.trim(pick(*a["stock_code"]).cast(StringType())).alias("stock_code"),
        F.trim(pick(*a["description"], default="Unknown").cast(StringType())).alias("description"),
        pick(*a["quantity"], default=1).cast(DoubleType()).cast(IntegerType()).alias("quantity"),
        parsed_date.alias("invoice_date"),
        pick(*a["unit_price"], default=0.0).cast(DoubleType()).alias("unit_price"),
        F.regexp_replace(F.trim(pick(*a["customer_id"]).cast(StringType())), r"\.0$", "").alias("customer_id"),
        F.coalesce(pick(*a["country"], default="Unknown").cast(StringType()), F.lit("Unknown")).alias("country"),
        F.lower(pick("event_type", default="purchase").cast(StringType())).alias("event_type"),
        F.coalesce(pick(*a["category"], default="retail").cast(StringType()), F.lit("retail")).alias("category"),
    )
