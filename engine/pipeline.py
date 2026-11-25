"""Medallion Architecture ETL Pipeline (Bronze -> Silver Parquet) + Benchmark."""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import pyarrow as pa
import pyarrow.parquet as pq
from pyspark.sql import DataFrame, SparkSession, functions as F

from engine.schemas import load_raw_dataset


@dataclass
class PipelineAudit:
    dataset_name: str
    schema_type: str
    raw_rows: int
    raw_partitions: int
    duplicate_rows: int
    cancellation_rows: int
    return_rows: int
    invalid_price_rows: int
    missing_customer_rows: int
    null_date_rows: int
    silver_valid_rows: int
    silver_partitions: int
    source_size_mb: float
    parquet_size_mb: float
    compression_ratio_pct: float
    raw_read_sec: float
    parquet_read_sec: float
    etl_total_sec: float
    silver_path: str
    events_path: str | None


def _dir_size_mb(path: Path) -> float:
    if path.is_file():
        return path.stat().st_size / (1024 * 1024)
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / (1024 * 1024)


def _write_parquet_safe(df: DataFrame, out_dir: Path, partition_cols: list[str] | None = None) -> None:
    """Write Snappy Parquet via Spark; fall back to PyArrow on Windows without winutils.exe."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    try:
        writer = df.write.mode("overwrite")
        if partition_cols:
            writer = writer.partitionBy(*partition_cols)
        writer.parquet(str(out_dir))
    except Exception as exc:
        if "HADOOP_HOME" not in str(exc) and "winutils" not in str(exc).lower():
            raise
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        pdf = df.toPandas()
        table = pa.Table.from_pandas(pdf, preserve_index=False)
        if partition_cols:
            pq.write_to_dataset(table, root_path=str(out_dir), partition_cols=partition_cols, compression="snappy")
        else:
            pq.write_table(table, str(out_dir / "part-00000.snappy.parquet"), compression="snappy")


def run_medallion_etl(
    spark: SparkSession,
    source_path: str | Path,
    dataset_name: str,
    lake_root: str | Path = "data_lake",
) -> PipelineAudit:
    """Execute Bronze -> Silver PySpark ETL, persist partitioned Parquet, and benchmark."""
    t0 = time.perf_counter()
    src = Path(source_path)
    lake = Path(lake_root)
    slug = "".join(c if c.isalnum() else "_" for c in dataset_name.lower()).strip("_")

    silver_dir = lake / "silver" / f"{slug}_sales.parquet"
    events_dir = lake / "silver" / f"{slug}_events.parquet"

    # 1. Bronze Read + Timing
    t_read = time.perf_counter()
    raw_df, schema_type = load_raw_dataset(spark, src)
    raw_df = raw_df.persist()
    raw_rows = raw_df.count()
    raw_read_sec = time.perf_counter() - t_read
    raw_partitions = raw_df.rdd.getNumPartitions()

    # Preserve clickstream events separately if REES46 (view / cart / remove_from_cart / purchase)
    events_path_str = None
    if schema_type == "rees46_clickstream":
        _write_parquet_safe(
            raw_df.select("event_type", "invoice_date", "stock_code", "category", "unit_price", "customer_id", "invoice_no"),
            events_dir,
        )
        events_path_str = str(events_dir)
        tx_df = raw_df.filter(F.col("event_type") == "purchase")
    else:
        tx_df = raw_df

    # 2. Data Quality Profiling (single distributed aggregation pass)
    audit_row = tx_df.agg(
        F.sum(F.when(F.col("invoice_no").startswith("C") | F.col("invoice_no").startswith("c"), 1).otherwise(0)).alias("cancellations"),
        F.sum(F.when(F.col("quantity") < 0, 1).otherwise(0)).alias("returns"),
        F.sum(F.when(F.col("unit_price").isNull() | (F.col("unit_price") <= 0), 1).otherwise(0)).alias("invalid_price"),
        F.sum(F.when(F.col("customer_id").isNull() | (F.trim(F.col("customer_id")) == ""), 1).otherwise(0)).alias("missing_cust"),
        F.sum(F.when(F.col("invoice_date").isNull(), 1).otherwise(0)).alias("null_date"),
    ).collect()[0]

    # Exact duplicate count
    tx_count = tx_df.count()
    dedup_df = tx_df.dropDuplicates()
    dedup_count = dedup_df.count()
    duplicate_rows = tx_count - dedup_count

    # 3. Silver Cleaning Rules (transparent flags, keep returns for Gross vs Net analysis)
    silver_df = (
        dedup_df
        .filter(F.col("invoice_date").isNotNull())
        .filter(F.col("unit_price").isNotNull() & (F.col("unit_price") > 0))
        .filter(F.col("quantity").isNotNull() & (F.col("quantity") != 0))
        .withColumn("is_cancellation", F.col("invoice_no").startswith("C") | F.col("invoice_no").startswith("c"))
        .withColumn("is_return", F.col("quantity") < 0)
        .withColumn("has_customer_id", F.col("customer_id").isNotNull() & (F.trim(F.col("customer_id")) != ""))
        .withColumn("line_total", F.round(F.col("quantity") * F.col("unit_price"), 2))
        .withColumn("year", F.year("invoice_date"))
        .withColumn("month", F.month("invoice_date"))
        .withColumn("year_month", F.date_format("invoice_date", "yyyy-MM"))
        .withColumn("day_of_week", F.date_format("invoice_date", "E"))
        .withColumn("hour", F.hour("invoice_date"))
    )

    # 4. Write Partitioned Silver Parquet Lake (partitioned by year, month)
    _write_parquet_safe(silver_df, silver_dir, partition_cols=["year", "month"])

    raw_df.unpersist()

    # 5. Benchmark Parquet Read Speed & Compression
    t_pq = time.perf_counter()
    pq_df = spark.read.parquet(str(silver_dir))
    silver_valid_rows = pq_df.count()
    parquet_read_sec = time.perf_counter() - t_pq
    silver_partitions = pq_df.rdd.getNumPartitions()

    source_size_mb = round(_dir_size_mb(src), 3)
    parquet_size_mb = round(_dir_size_mb(silver_dir), 3)
    compression_ratio_pct = round((1 - (parquet_size_mb / source_size_mb)) * 100, 1) if source_size_mb > 0 else 0.0

    return PipelineAudit(
        dataset_name=dataset_name,
        schema_type=schema_type,
        raw_rows=raw_rows,
        raw_partitions=raw_partitions,
        duplicate_rows=int(duplicate_rows),
        cancellation_rows=int(audit_row["cancellations"] or 0),
        return_rows=int(audit_row["returns"] or 0),
        invalid_price_rows=int(audit_row["invalid_price"] or 0),
        missing_customer_rows=int(audit_row["missing_cust"] or 0),
        null_date_rows=int(audit_row["null_date"] or 0),
        silver_valid_rows=int(silver_valid_rows),
        silver_partitions=int(silver_partitions),
        source_size_mb=source_size_mb,
        parquet_size_mb=parquet_size_mb,
        compression_ratio_pct=compression_ratio_pct,
        raw_read_sec=round(raw_read_sec, 3),
        parquet_read_sec=round(parquet_read_sec, 3),
        etl_total_sec=round(time.perf_counter() - t0, 3),
        silver_path=str(silver_dir),
        events_path=events_path_str,
    )


def load_silver(spark: SparkSession, silver_path: str) -> DataFrame:
    """Load the cleaned Silver Parquet table."""
    return spark.read.parquet(silver_path)


def audit_to_dict(audit: PipelineAudit) -> Dict[str, object]:
    return audit.__dict__
