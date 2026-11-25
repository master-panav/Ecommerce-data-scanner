"""Multi-dataset schema detection and normalization adapters for PySpark.

Normalizes five public Big Data e-commerce schemas into a unified canonical
transaction schema so downstream Silver/Gold analytics and Spark MLlib models
work consistently across datasets:
    invoice_no, stock_code, description, quantity, invoice_date,
    unit_price, customer_id, country, event_type, category
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from pyspark.sql import DataFrame, SparkSession, functions as F
from pyspark.sql.types import DoubleType, IntegerType, StringType

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


def _read_excel_to_spark(spark: SparkSession, file_path: Path) -> DataFrame:
    """Load single or multi-sheet Excel workbooks into a Spark DataFrame."""
    excel = pd.ExcelFile(file_path, engine="openpyxl")
    frames = [excel.parse(sheet) for sheet in excel.sheet_names]
    pdf = pd.concat(frames, ignore_index=True)
    for col in pdf.columns:
        if pdf[col].dtype == "object":
            pdf[col] = pdf[col].astype(str).replace({"nan": None, "None": None})
    return spark.createDataFrame(pdf)


def load_raw_dataset(spark: SparkSession, path: str | Path) -> Tuple[DataFrame, str]:
    """Read a CSV, CSV.GZ, Parquet, Excel file, or multi-table folder into Spark.

    Returns (normalized_spark_df, detected_schema_key).
    """
    p = Path(path)

    # Multi-table directory support (Olist or Instacart folders)
    if p.is_dir():
        if (p / "olist_orders_dataset.csv").exists():
            return _load_olist_folder(spark, p), "olist_brazil"
        if (p / "orders.csv").exists() and (p / "products.csv").exists():
            return _load_instacart_folder(spark, p), "instacart_basket"
        # Otherwise treat directory as Parquet or multi-CSV partition folder
        parquet_files = list(p.glob("*.parquet"))
        if parquet_files:
            df = spark.read.parquet(str(p))
            schema_type = detect_schema_type(df.columns)
            return normalize_dataframe(df, schema_type), schema_type
        df = spark.read.option("header", "true").option("inferSchema", "true").csv(str(p))
        schema_type = detect_schema_type(df.columns)
        return normalize_dataframe(df, schema_type), schema_type

    suffix = p.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        raw_df = _read_excel_to_spark(spark, p)
    elif suffix == ".parquet":
        raw_df = spark.read.parquet(str(p))
    else:
        raw_df = (
            spark.read
            .option("header", "true")
            .option("inferSchema", "true")
            .option("escape", '"')
            .csv(str(p))
        )

    schema_type = detect_schema_type(raw_df.columns)
    return normalize_dataframe(raw_df, schema_type), schema_type


def _load_olist_folder(spark: SparkSession, folder: Path) -> DataFrame:
    """Perform distributed Spark Joins across Olist's relational CSV tables."""
    orders = spark.read.option("header", "true").option("inferSchema", "true").csv(str(folder / "olist_orders_dataset.csv"))
    items = spark.read.option("header", "true").option("inferSchema", "true").csv(str(folder / "olist_order_items_dataset.csv"))
    customers = spark.read.option("header", "true").option("inferSchema", "true").csv(str(folder / "olist_customers_dataset.csv"))
    products = spark.read.option("header", "true").option("inferSchema", "true").csv(str(folder / "olist_products_dataset.csv"))

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
    orders = spark.read.option("header", "true").option("inferSchema", "true").csv(str(folder / "orders.csv"))
    prior_path = folder / "order_products__prior.csv"
    items_path = prior_path if prior_path.exists() else folder / "order_products__train.csv"
    items = spark.read.option("header", "true").option("inferSchema", "true").csv(str(items_path))
    products = spark.read.option("header", "true").option("inferSchema", "true").csv(str(folder / "products.csv"))

    depts_path = folder / "departments.csv"
    if depts_path.exists():
        depts = spark.read.option("header", "true").option("inferSchema", "true").csv(str(depts_path))
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
    raw_date = pick("invoicedate", "invoice_date", "date", "order_date").cast(StringType())
    parsed_date = F.coalesce(
        F.to_timestamp(raw_date),
        F.to_timestamp(raw_date, "M/d/yyyy H:mm"),
        F.to_timestamp(raw_date, "d/M/yyyy H:mm"),
        F.to_timestamp(raw_date, "yyyy-MM-dd HH:mm:ss"),
        F.to_timestamp(raw_date, "yyyy-MM-dd"),
    )

    return df.select(
        pick("invoiceno", "invoice", "invoice_no", "transactionno", "order_id").cast(StringType()).alias("invoice_no"),
        pick("stockcode", "stock_code", "productno", "product_id").cast(StringType()).alias("stock_code"),
        F.trim(pick("description", "product", "product_name", default="Unknown").cast(StringType())).alias("description"),
        pick("quantity", "qty", default=1).cast(IntegerType()).alias("quantity"),
        parsed_date.alias("invoice_date"),
        pick("unitprice", "price", "unit_price", default=0.0).cast(DoubleType()).alias("unit_price"),
        F.regexp_replace(pick("customerid", "customer id", "customer_id", "customerno").cast(StringType()), r"\.0$", "").alias("customer_id"),
        F.coalesce(pick("country", default="Unknown").cast(StringType()), F.lit("Unknown")).alias("country"),
        F.lower(pick("event_type", default="purchase").cast(StringType())).alias("event_type"),
        F.coalesce(pick("category", "category_code", default="retail").cast(StringType()), F.lit("retail")).alias("category"),
    )
