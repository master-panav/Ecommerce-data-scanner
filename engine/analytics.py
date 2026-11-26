"""Gold Layer: Distributed Spark SQL Business Intelligence & Cross-Dataset Analytics."""

from __future__ import annotations

from typing import Dict

import pandas as pd
from pyspark.sql import DataFrame, SparkSession, functions as F

from engine.io_utils import read_parquet


def filter_silver(
    df: DataFrame,
    countries: list[str] | None = None,
    date_range: tuple[str, str] | None = None,
) -> DataFrame:
    """Apply optional country and date filters lazily inside Spark."""
    out = df
    if countries:
        out = out.filter(F.col("country").isin(countries))
    if date_range and len(date_range) == 2:
        out = out.filter(
            (F.to_date("invoice_date") >= F.lit(date_range[0]))
            & (F.to_date("invoice_date") <= F.lit(date_range[1]))
        )
    return out


def compute_kpis(df: DataFrame) -> Dict[str, float]:
    """Compute Gross Sales, Returns, Net Revenue, Orders, Active Customers, AOV, Return Rate."""
    completed = df.filter(~F.col("is_cancellation") & ~F.col("is_return"))
    returns = df.filter(F.col("is_cancellation") | F.col("is_return"))

    comp_row = completed.agg(
        F.coalesce(F.sum("line_total"), F.lit(0.0)).alias("gross_sales"),
        F.countDistinct("invoice_no").alias("orders"),
        F.countDistinct(F.when(F.col("has_customer_id"), F.col("customer_id"))).alias("customers"),
        F.coalesce(F.sum("quantity"), F.lit(0)).alias("units_sold"),
    ).collect()[0]

    ret_row = returns.agg(
        F.coalesce(F.sum(F.abs(F.col("line_total"))), F.lit(0.0)).alias("return_value"),
        F.countDistinct("invoice_no").alias("return_orders"),
    ).collect()[0]

    gross = float(comp_row["gross_sales"])
    ret_val = float(ret_row["return_value"])
    net = gross - ret_val
    orders = int(comp_row["orders"])
    aov = round(gross / orders, 2) if orders > 0 else 0.0
    return_rate = round((ret_val / gross) * 100, 2) if gross > 0 else 0.0

    return {
        "gross_sales": round(gross, 2),
        "return_value": round(ret_val, 2),
        "net_revenue": round(net, 2),
        "orders": orders,
        "return_orders": int(ret_row["return_orders"]),
        "customers": int(comp_row["customers"]),
        "units_sold": int(comp_row["units_sold"]),
        "aov": aov,
        "return_rate_pct": return_rate,
    }


def monthly_trends(df: DataFrame) -> pd.DataFrame:
    """Monthly gross sales, returns, net revenue, and distinct order counts."""
    agg_df = (
        df.groupBy("year_month")
        .agg(
            F.round(F.sum(F.when(~F.col("is_cancellation") & ~F.col("is_return"), F.col("line_total")).otherwise(0.0)), 2).alias("gross_sales"),
            F.round(F.sum(F.when(F.col("is_cancellation") | F.col("is_return"), F.abs(F.col("line_total"))).otherwise(0.0)), 2).alias("returns"),
            F.countDistinct(F.when(~F.col("is_cancellation") & ~F.col("is_return"), F.col("invoice_no"))).alias("orders"),
            F.countDistinct(F.when(F.col("has_customer_id") & ~F.col("is_cancellation"), F.col("customer_id"))).alias("active_customers"),
        )
        .withColumn("net_revenue", F.round(F.col("gross_sales") - F.col("returns"), 2))
        .orderBy("year_month")
    )
    return agg_df.toPandas()


def top_products(df: DataFrame, limit: int = 15) -> pd.DataFrame:
    """Top products ranked by completed revenue and quantity."""
    completed = df.filter(~F.col("is_cancellation") & ~F.col("is_return"))
    agg_df = (
        completed.groupBy("stock_code", "description")
        .agg(
            F.round(F.sum("line_total"), 2).alias("revenue"),
            F.sum("quantity").alias("units_sold"),
            F.countDistinct("invoice_no").alias("orders"),
        )
        .orderBy(F.desc("revenue"))
        .limit(limit)
    )
    return agg_df.toPandas()


def country_summary(df: DataFrame) -> pd.DataFrame:
    """Country-level revenue, order volume, and customer count."""
    completed = df.filter(~F.col("is_cancellation") & ~F.col("is_return"))
    agg_df = (
        completed.groupBy("country")
        .agg(
            F.round(F.sum("line_total"), 2).alias("revenue"),
            F.countDistinct("invoice_no").alias("orders"),
            F.countDistinct(F.when(F.col("has_customer_id"), F.col("customer_id"))).alias("customers"),
        )
        .withColumn("aov", F.round(F.col("revenue") / F.col("orders"), 2))
        .orderBy(F.desc("revenue"))
    )
    return agg_df.toPandas()


def shopping_heatmap(df: DataFrame) -> pd.DataFrame:
    """Day-of-week x Hour-of-day completed order density."""
    completed = df.filter(~F.col("is_cancellation") & ~F.col("is_return"))
    agg_df = (
        completed.groupBy("day_of_week", "hour")
        .agg(
            F.countDistinct("invoice_no").alias("orders"),
            F.round(F.sum("line_total"), 2).alias("revenue"),
        )
        .orderBy("day_of_week", "hour")
    )
    return agg_df.toPandas()


def clickstream_funnel(spark: SparkSession, events_path: str) -> pd.DataFrame:
    """Compute View -> Cart -> Remove -> Purchase conversion funnel for REES46 datasets."""
    ev = read_parquet(spark, events_path)
    agg_df = (
        ev.groupBy("event_type")
        .agg(
            F.count("*").alias("events"),
            F.countDistinct("customer_id").alias("unique_users"),
            F.countDistinct("invoice_no").alias("sessions"),
        )
    )
    pdf = agg_df.toPandas()
    order = {"view": 1, "cart": 2, "remove_from_cart": 3, "purchase": 4}
    pdf["stage_order"] = pdf["event_type"].map(lambda x: order.get(x, 99))
    return pdf.sort_values("stage_order").drop(columns=["stage_order"])


def compare_datasets(
    df_a: DataFrame,
    name_a: str,
    df_b: DataFrame,
    name_b: str,
) -> pd.DataFrame:
    """Side-by-side Gold-layer KPI comparison across any two Silver datasets."""
    kpi_a = compute_kpis(df_a)
    kpi_b = compute_kpis(df_b)

    rows = []
    labels = {
        "gross_sales": "Gross Sales",
        "return_value": "Returns & Cancellations",
        "net_revenue": "Net Revenue",
        "orders": "Completed Orders",
        "customers": "Unique Customers",
        "units_sold": "Units Sold",
        "aov": "Average Order Value (AOV)",
        "return_rate_pct": "Return Rate (%)",
    }
    for key, label in labels.items():
        va, vb = kpi_a[key], kpi_b[key]
        delta_pct = round(((vb - va) / va) * 100, 1) if va else 0.0
        rows.append({"Metric": label, name_a: va, name_b: vb, "Delta (%)": delta_pct})
    return pd.DataFrame(rows)
