"""Gold Layer: Distributed Spark SQL Business Intelligence & Cross-Dataset Analytics."""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

import pandas as pd
from pyspark.sql import DataFrame, SparkSession, functions as F
from pyspark.sql.types import BooleanType, IntegralType, NumericType, StringType

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

    has_profit = "profit" in df.columns
    comp_row = completed.agg(
        F.sum("profit").alias("profit") if has_profit else F.lit(None).alias("profit"),
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
        # None when the dataset has no profit column
        "profit": None if comp_row["profit"] is None else round(float(comp_row["profit"]), 2),
        "profit_margin_pct": None if comp_row["profit"] is None or gross <= 0
        else round(float(comp_row["profit"]) / gross * 100, 2),
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
        "profit": "Profit",
        "profit_margin_pct": "Profit Margin (%)",
    }
    for key, label in labels.items():
        va, vb = kpi_a[key], kpi_b[key]
        if va is None or vb is None:
            continue
        delta_pct = round(((vb - va) / va) * 100, 1) if va else 0.0
        rows.append({"Metric": label, name_a: va, name_b: vb, "Delta (%)": delta_pct})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------
# Slice & dice (OLAP-style) explorer over any dimension / measure of a Silver table
# ---------------------------------------------------------------------------------------
TIME_DIMENSIONS = ["year", "year_month", "month", "day_of_week", "hour"]
_EXPLORABLE = ("country", "category", "order_status", "profit", "discount")
# Numeric identifiers / coordinates are not meaningful to sum or average
_NOT_A_MEASURE = re.compile(r"(^|_)(id|ids|no|code|zip|zipcode|postal_code|latitude|longitude|lat|lon|lng|year|month)$")


def pretty(column: str) -> str:
    return column[2:].replace("_", " ").title() if column.startswith("x_") else column.replace("_", " ").title()


def explorer_fields(df: DataFrame, max_categories: int = 300, max_numeric_codes: int = 12) -> Tuple[List[str], List[str]]:
    """Return (dimensions, numeric measures) usable in the explorer, using one approx-distinct Spark pass.

    Dimensions: calendar fields, text/boolean columns with 2..max_categories values, and small integer
    codes (e.g. a 0/1 late-delivery flag). Measures: numeric source attributes such as profit,
    discount or shipping days, excluding identifiers and coordinates.
    """
    candidates = [f for f in df.schema.fields if f.name.startswith("x_") or f.name in _EXPLORABLE]
    dims: List[str] = []
    numeric: List[str] = []
    if candidates:
        row = df.agg(*[F.approx_count_distinct(F.col(f"`{f.name}`")).alias(f"d{i}")
                       for i, f in enumerate(candidates)]).collect()[0]
        for i, f in enumerate(candidates):
            distinct = row[f"d{i}"] or 0
            if isinstance(f.dataType, (StringType, BooleanType)):
                if 2 <= distinct <= max_categories:
                    dims.append(f.name)
            elif isinstance(f.dataType, NumericType):
                if isinstance(f.dataType, IntegralType) and 2 <= distinct <= max_numeric_codes:
                    dims.append(f.name)
                if distinct >= 1 and not _NOT_A_MEASURE.search(f.name):
                    numeric.append(f.name)
    return [d for d in TIME_DIMENSIONS if d in df.columns] + dims, numeric


def _measure_expr(measure: str, agg: str):
    base = {
        "Revenue": F.sum("line_total"),
        "Orders": F.countDistinct("invoice_no"),
        "Units Sold": F.sum("quantity"),
        "Customers": F.countDistinct(F.when(F.col("has_customer_id"), F.col("customer_id"))),
        "Order Lines": F.count(F.lit(1)),
    }
    if measure in base:
        return base[measure]
    col = F.col(f"`{measure}`")
    return {"Sum": F.sum(col), "Average": F.avg(col), "Min": F.min(col), "Max": F.max(col)}[agg]


def explore(
    df: DataFrame,
    dimension: str,
    measure: str,
    agg: str = "Sum",
    column_dimension: str | None = None,
    completed_only: bool = True,
    limit: int = 25,
) -> pd.DataFrame:
    """Group a Silver table by one or two dimensions and aggregate one measure in Spark."""
    data = df.filter(~F.col("is_cancellation") & ~F.col("is_return")) if completed_only else df
    keys = [dimension] + ([column_dimension] if column_dimension and column_dimension != dimension else [])
    out = data.groupBy(*[F.col(f"`{k}`") for k in keys]).agg(F.round(_measure_expr(measure, agg), 2).alias("value"))
    if len(keys) == 1:
        order = F.col(f"`{dimension}`") if dimension in TIME_DIMENSIONS else F.desc("value")
        return out.orderBy(order).limit(limit).toPandas()
    top = [r[0] for r in data.groupBy(F.col(f"`{dimension}`")).agg(_measure_expr(measure, agg).alias("v"))
           .orderBy(F.desc("v")).limit(limit).collect()]
    return out.filter(F.col(f"`{dimension}`").isin(top)).toPandas()
