"""Medallion Architecture ETL Pipeline (Bronze -> Silver Parquet) + Benchmark."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Dict

from pyspark.sql import DataFrame, SparkSession, functions as F

from engine.io_utils import read_parquet, write_parquet
from engine.schemas import load_raw_dataset

AUDIT_FILE = "_audit.json"


class EmptySilverError(ValueError):
    """Raised when cleaning rules leave no valid rows (usually a column-mapping problem)."""


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
    empty_rows_removed: int = 0
    empty_columns_removed: list = field(default_factory=list)
    malformed_quantity_rows: int = 0
    column_mapping: dict = field(default_factory=dict)


def dataset_slug(dataset_name: str) -> str:
    """Filesystem-safe, collision-free folder name ("Sales.csv" and "sales csv" must not share a Silver table)."""
    readable = "".join(c if c.isascii() and c.isalnum() else "_" for c in dataset_name.lower()).strip("_")[:60] or "dataset"
    digest = hashlib.sha1(dataset_name.encode("utf-8")).hexdigest()[:8]
    return f"{readable}_{digest}"


def _dir_size_mb(path: Path) -> float:
    if path.is_file():
        return path.stat().st_size / (1024 * 1024)
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file() and not f.name.startswith("_"))
    return total / (1024 * 1024)


def save_audit(audit: PipelineAudit) -> None:
    Path(audit.silver_path, AUDIT_FILE).write_text(json.dumps(asdict(audit), indent=2), encoding="utf-8")


def load_saved_audits(lake_root: str | Path = "data_lake") -> Dict[str, PipelineAudit]:
    """Restore previously processed Silver datasets so a browser refresh doesn't lose them."""
    audits: Dict[str, PipelineAudit] = {}
    known = {f.name for f in fields(PipelineAudit)}
    for audit_file in sorted(Path(lake_root, "silver").glob(f"*/{AUDIT_FILE}")):
        try:
            data = json.loads(audit_file.read_text(encoding="utf-8"))
            audit = PipelineAudit(**{k: v for k, v in data.items() if k in known})
        except (OSError, ValueError, TypeError):
            continue
        # Re-anchor paths to where the audit file actually is, so moving/renaming the
        # project folder doesn't orphan previously processed datasets.
        silver_dir = audit_file.parent
        audit.silver_path = str(silver_dir)
        if audit.events_path:
            events_dir = silver_dir.parent / Path(audit.events_path).name
            audit.events_path = str(events_dir) if events_dir.exists() else None
        audits[audit.dataset_name] = audit
    return audits


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
    slug = dataset_slug(dataset_name)

    silver_dir = lake / "silver" / f"{slug}_sales.parquet"
    events_dir = lake / "silver" / f"{slug}_events.parquet"

    # 1. Bronze Read + Timing
    t_read = time.perf_counter()
    raw_df, schema_type, prep = load_raw_dataset(spark, src)
    raw_df = raw_df.persist()
    empty_rows = int(prep.get("empty_rows", 0))
    raw_rows = raw_df.count() + empty_rows
    raw_read_sec = time.perf_counter() - t_read
    raw_partitions = raw_df.rdd.getNumPartitions()

    if raw_rows == empty_rows:
        raw_df.unpersist()
        raise EmptySilverError(f"`{dataset_name}` contains no data rows.")

    try:
        # Preserve clickstream events separately if REES46 (view / cart / remove_from_cart / purchase)
        events_path_str = None
        # Clickstream (any source with view/cart/purchase events): keep all events for the funnel
        is_clickstream = "event_type" in prep.get("column_mapping", {}) or schema_type == "rees46_clickstream"
        if is_clickstream:
            write_parquet(
                raw_df.select("event_type", "invoice_date", "stock_code", "category", "unit_price", "customer_id", "invoice_no"),
                events_dir,
            )
            events_path_str = str(events_dir)
            tx_df = raw_df.filter(F.col("event_type") == "purchase")
        else:
            tx_df = raw_df

        is_cancel = F.upper(F.coalesce(F.col("invoice_no"), F.lit(""))).startswith("C")
        has_customer = F.col("customer_id").isNotNull() & (F.trim(F.col("customer_id")) != "")

        # 2. Data Quality Profiling (single distributed aggregation pass)
        audit_row = tx_df.agg(
            F.count("*").alias("tx_count"),
            F.sum(F.when(is_cancel, 1).otherwise(0)).alias("cancellations"),
            F.sum(F.when(F.col("quantity") < 0, 1).otherwise(0)).alias("returns"),
            F.sum(F.when(F.col("unit_price").isNull() | (F.col("unit_price") <= 0), 1).otherwise(0)).alias("invalid_price"),
            F.sum(F.when(~has_customer, 1).otherwise(0)).alias("missing_cust"),
            F.sum(F.when(F.col("invoice_date").isNull(), 1).otherwise(0)).alias("null_date"),
            F.sum(F.when(F.col("quantity").isNull() | (F.col("quantity") == 0), 1).otherwise(0)).alias("bad_qty"),
        ).collect()[0]

        # Exact duplicate count
        tx_count = int(audit_row["tx_count"] or 0)
        dedup_df = tx_df.dropDuplicates()
        duplicate_rows = tx_count - dedup_df.count()

        # 3. Silver Cleaning Rules (transparent flags, keep returns for Gross vs Net analysis)
        silver_df = (
            dedup_df
            .filter(F.col("invoice_date").isNotNull())
            .filter(F.col("unit_price").isNotNull() & (F.col("unit_price") > 0))
            .filter(F.col("quantity").isNotNull() & (F.col("quantity") != 0))
            .withColumn("is_cancellation", is_cancel)
            .withColumn("is_return", F.col("quantity") < 0)
            .withColumn("has_customer_id", has_customer)
            .withColumn("line_total", F.round(F.col("quantity") * F.col("unit_price"), 2))
            .withColumn("year", F.year("invoice_date"))
            .withColumn("month", F.month("invoice_date"))
            .withColumn("year_month", F.date_format("invoice_date", "yyyy-MM"))
            .withColumn("day_of_week", F.date_format("invoice_date", "E"))
            .withColumn("hour", F.hour("invoice_date"))
        )

        if silver_df.limit(1).count() == 0:
            mapped = ", ".join(f"{k} <- {v}" for k, v in prep.get("column_mapping", {}).items())
            raise EmptySilverError(
                f"No valid rows left after cleaning `{dataset_name}`: of {tx_count:,} transaction rows, "
                f"{int(audit_row['null_date'] or 0):,} had no parseable date, "
                f"{int(audit_row['invalid_price'] or 0):,} had no positive price and "
                f"{int(audit_row['bad_qty'] or 0):,} had a missing/zero quantity. Column mapping used: {mapped}."
            )

        # 4. Write Partitioned Silver Parquet Lake (partitioned by year, month)
        write_parquet(silver_df, silver_dir, partition_cols=["year", "month"])
    finally:
        raw_df.unpersist()

    # 5. Benchmark Parquet Read Speed & Compression
    t_pq = time.perf_counter()
    pq_df = read_parquet(spark, silver_dir)
    silver_valid_rows = pq_df.count()
    parquet_read_sec = time.perf_counter() - t_pq
    silver_partitions = pq_df.rdd.getNumPartitions()

    source_size_mb = round(_dir_size_mb(src), 3)
    parquet_size_mb = round(_dir_size_mb(silver_dir), 3)
    compression_ratio_pct = round((1 - (parquet_size_mb / source_size_mb)) * 100, 1) if source_size_mb > 0 else 0.0

    audit = PipelineAudit(
        dataset_name=dataset_name,
        schema_type=schema_type,
        raw_rows=int(raw_rows),
        raw_partitions=int(raw_partitions),
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
        empty_rows_removed=empty_rows,
        empty_columns_removed=list(prep.get("empty_columns", [])),
        malformed_quantity_rows=int(audit_row["bad_qty"] or 0),
        column_mapping=dict(prep.get("column_mapping", {})),
    )
    save_audit(audit)
    return audit


def load_silver(spark: SparkSession, silver_path: str) -> DataFrame:
    """Load the cleaned Silver Parquet table."""
    return read_parquet(spark, silver_path)


def audit_to_dict(audit: PipelineAudit) -> Dict[str, object]:
    return asdict(audit)
