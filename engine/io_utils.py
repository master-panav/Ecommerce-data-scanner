"""Cross-platform Parquet / CSV I/O helpers.

On Windows, Spark's Hadoop layer needs ``winutils.exe`` + ``hadoop.dll`` (via
``HADOOP_HOME``) to *list directories* and to *write* files. Without them every
``spark.read.parquet(<dir>)`` fails with
``UnsatisfiedLinkError: NativeIO$Windows.access0`` and every ``df.write`` fails
with ``HADOOP_HOME and hadoop.home.dir are unset``.

These helpers keep the app working out of the box on a plain Windows machine:

* writes fall back to PyArrow (microsecond timestamps, which Spark 3.5 can read);
* reads pass Spark an explicit list of files (plus ``basePath`` so ``year=/month=``
  partition columns are still discovered), which avoids the native directory
  listing entirely.
"""

from __future__ import annotations

import functools
import os
import shutil
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from pyspark.sql import DataFrame, SparkSession

_NATIVE_IO_ERROR_MARKERS = ("hadoop_home", "winutils", "nativeio", "unsatisfiedlinkerror", "hadoop.home.dir")


def _spark_path(path: Path) -> str:
    return path.resolve().as_posix()


@functools.lru_cache(maxsize=1)
def native_hadoop_io_available() -> bool:
    """True when Spark can use Hadoop's native local filesystem calls."""
    if not sys.platform.startswith("win"):
        return True
    home = os.environ.get("HADOOP_HOME") or os.environ.get("hadoop.home.dir")
    if not home:
        return False
    bin_dir = Path(home) / "bin"
    return (bin_dir / "winutils.exe").exists() and (bin_dir / "hadoop.dll").exists()


def is_native_io_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in _NATIVE_IO_ERROR_MARKERS)


def _list_data_files(root: Path, suffixes: tuple[str, ...]) -> list[str]:
    files = [
        f for f in root.rglob("*")
        if f.is_file()
        and not any(part.startswith(("_", ".")) for part in f.relative_to(root).parts)
        and f.name.lower().endswith(suffixes)
    ]
    return sorted(_spark_path(f) for f in files)


def read_parquet(spark: SparkSession, path: str | Path) -> DataFrame:
    """Read a Parquet file or (partitioned) directory on any OS."""
    p = Path(path)
    if p.is_file() or native_hadoop_io_available():
        return spark.read.parquet(_spark_path(p))
    files = _list_data_files(p, (".parquet",))
    if not files:
        raise FileNotFoundError(f"No .parquet files found under {p}")
    return spark.read.option("basePath", _spark_path(p)).parquet(*files)


def read_csv(spark: SparkSession, path: str | Path, encoding: str = "UTF-8") -> DataFrame:
    """Read a CSV / CSV.GZ file or a folder of CSV part files on any OS."""
    p = Path(path)
    reader = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .option("escape", '"')
        .option("encoding", encoding)
    )
    if p.is_file() or native_hadoop_io_available():
        return reader.csv(_spark_path(p))
    files = _list_data_files(p, (".csv", ".csv.gz"))
    if not files:
        raise FileNotFoundError(f"No CSV files found under {p}")
    return reader.csv(files)


def _arrow_table_spark_compatible(df: DataFrame) -> pa.Table:
    """Collect a Spark DataFrame into an Arrow table with Spark-readable timestamps.

    PyArrow defaults to nanosecond timestamps, which Spark 3.5 rejects with
    ``illegalParquetTypeError``; cast them to microseconds.
    """
    table = pa.Table.from_pandas(df.toPandas(), preserve_index=False)
    fields = [
        pa.field(f.name, pa.timestamp("us", tz=f.type.tz)) if pa.types.is_timestamp(f.type) else f
        for f in table.schema
    ]
    return table.cast(pa.schema(fields), safe=False)


def write_parquet(df: DataFrame, out_dir: str | Path, partition_cols: list[str] | None = None) -> None:
    """Write Snappy Parquet via Spark; fall back to PyArrow when native Hadoop I/O is unavailable."""
    out = Path(out_dir)
    if out.exists():
        shutil.rmtree(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if native_hadoop_io_available():
        try:
            writer = df.write.mode("overwrite").option("compression", "snappy")
            if partition_cols:
                writer = writer.partitionBy(*partition_cols)
            writer.parquet(_spark_path(out))
            return
        except Exception as exc:  # pragma: no cover - depends on host setup
            if not is_native_io_error(exc):
                raise
            if out.exists():
                shutil.rmtree(out)

    out.mkdir(parents=True, exist_ok=True)
    table = _arrow_table_spark_compatible(df)
    if partition_cols:
        pq.write_to_dataset(
            table,
            root_path=str(out),
            partition_cols=partition_cols,
            compression="snappy",
            basename_template="part-{i}.snappy.parquet",
            existing_data_behavior="overwrite_or_ignore",
        )
    else:
        pq.write_table(table, str(out / "part-00000.snappy.parquet"), compression="snappy")
