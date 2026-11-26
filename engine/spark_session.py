"""SparkSession factory with memory tuning, Arrow support, and runtime telemetry."""

from __future__ import annotations

import os
import sys
from typing import Dict

# PySpark 3.5's pandas/Arrow bridge still imports `distutils`, removed in Python 3.12.
# setuptools ships a drop-in copy; importing it registers the shim.
try:  # pragma: no cover - depends on interpreter version
    import distutils.version  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    try:
        import setuptools  # noqa: F401
    except ModuleNotFoundError:
        pass

from pyspark.sql import SparkSession

from engine.io_utils import native_hadoop_io_available


# Ingestion is already cast-safe; ANSI off is a safety net so a stray malformed value yields NULL
# instead of aborting a job.
RUNTIME_CONF = {
    "spark.sql.session.timeZone": "UTC",  # keep source wall-clock times (REES46 is UTC)
    "spark.sql.ansi.enabled": "false",
    "spark.sql.legacy.timeParserPolicy": "CORRECTED",  # no SparkUpgradeException on odd date strings
}


def get_spark(app_name: str = "EcommerceBigDataWorkbench", shuffle_partitions: int = 8) -> SparkSession:
    """Return an active tuned local SparkSession (reuses existing session if alive)."""
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

    spark = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.execution.arrow.pyspark.fallback.enabled", "true")
        .config("spark.sql.parquet.compression.codec", "snappy")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.driver.memory", "4g")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    for key, value in RUNTIME_CONF.items():  # also applied to a reused session
        spark.conf.set(key, value)
    set_shuffle_partitions(spark, shuffle_partitions)
    return spark


def set_shuffle_partitions(spark: SparkSession, shuffle_partitions: int) -> None:
    """Shuffle partitions is a runtime SQL conf, so it can change without restarting Spark."""
    spark.conf.set("spark.sql.shuffle.partitions", str(int(shuffle_partitions)))


def get_spark_telemetry(spark: SparkSession) -> Dict[str, str]:
    """Return key Spark runtime parameters for the Big Data Monitor view."""
    sc = spark.sparkContext
    conf = sc.getConf()
    return {
        "App Name": sc.appName,
        "Spark Version": spark.version,
        "Master": sc.master,
        "Default Parallelism (Cores)": str(sc.defaultParallelism),
        "Shuffle Partitions": spark.conf.get("spark.sql.shuffle.partitions", "200"),
        "Adaptive Query Execution": spark.conf.get("spark.sql.adaptive.enabled", "true"),
        "Parquet Compression": spark.conf.get("spark.sql.parquet.compression.codec", "snappy"),
        "Driver Memory": conf.get("spark.driver.memory", "default"),
        "Parquet Writer": "Spark (native Hadoop)" if native_hadoop_io_available() else "PyArrow fallback (no winutils)",
    }
