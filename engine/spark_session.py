"""SparkSession factory with memory tuning, Arrow support, and runtime telemetry."""

from __future__ import annotations

import os
import sys
from typing import Dict

from pyspark.sql import SparkSession


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
        .config("spark.sql.parquet.compression.codec", "snappy")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.driver.memory", "4g")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def get_spark_telemetry(spark: SparkSession) -> Dict[str, str]:
    """Return key Spark runtime parameters for the Big Data Monitor view."""
    sc = spark.sparkContext
    conf = sc.getConf()
    return {
        "App Name": sc.appName,
        "Spark Version": spark.version,
        "Master": sc.master,
        "Default Parallelism (Cores)": str(sc.defaultParallelism),
        "Shuffle Partitions": conf.get("spark.sql.shuffle.partitions", "200"),
        "Adaptive Query Execution": conf.get("spark.sql.adaptive.enabled", "true"),
        "Parquet Compression": conf.get("spark.sql.parquet.compression.codec", "snappy"),
        "Driver Memory": conf.get("spark.driver.memory", "default"),
    }
