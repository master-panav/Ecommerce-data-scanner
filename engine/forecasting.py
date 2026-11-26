"""Gold Layer: revenue forecasting (Spark MLlib) and customer cohort analysis.

Forecasting
    Monthly totals are aggregated in Spark. A trailing month with too few trading days is
    excluded (e.g. UCI Online Retail ends on 9 Dec 2011). Four candidate models are scored
    on the most recent `holdout` months, never seen during training:
      * Naive               - repeat the last observed month
      * Seasonal naive      - repeat the same month one year earlier (needs 12+ months)
      * Linear trend        - Spark MLlib LinearRegression on a time index
      * Trend + seasonality - LinearRegression on time index + sin/cos of month-of-year
    The model with the lowest holdout RMSE is refit on all months and projected forward
    with an approximate 95% interval (+/- 1.96 x training residual std).

Cohorts
    Customers are grouped by the month of their first completed purchase. For each later
    month we report the share of the cohort that bought again (retention) and cumulative
    revenue per customer (a simple lifetime-value curve). Periods after the end of the data
    are left empty rather than shown as 0%.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.sql import DataFrame, functions as F

MEASURES = {"Revenue": "revenue", "Orders": "orders", "Active Customers": "customers"}
TREND = ["t"]
SEASONAL = ["t", "month_sin", "month_cos"]


# ---------------------------------------------------------------------------------------
# Forecasting
# ---------------------------------------------------------------------------------------
def monthly_series(silver_df: DataFrame, measure: str = "Revenue") -> tuple[pd.DataFrame, str | None]:
    """Aggregate completed sales per calendar month in Spark; fill gaps with 0 and drop a partial last month."""
    completed = silver_df.filter(~F.col("is_cancellation") & ~F.col("is_return"))
    monthly = (
        completed.groupBy(F.trunc("invoice_date", "month").alias("month"))
        .agg(
            F.sum("line_total").alias("revenue"),
            F.countDistinct("invoice_no").alias("orders"),
            F.countDistinct(F.when(F.col("has_customer_id"), F.col("customer_id"))).alias("customers"),
            F.countDistinct(F.to_date("invoice_date")).alias("trading_days"),
        )
        .orderBy("month")
        .toPandas()
    )
    if monthly.empty:
        return pd.DataFrame(columns=["month", "y"]), None
    monthly["month"] = pd.to_datetime(monthly["month"])
    full = pd.DataFrame({"month": pd.date_range(monthly["month"].min(), monthly["month"].max(), freq="MS")})
    monthly = full.merge(monthly, on="month", how="left").fillna(0)

    dropped = None
    if len(monthly) >= 4:
        typical_days = monthly["trading_days"].iloc[:-1].median()
        if monthly["trading_days"].iloc[-1] < 0.6 * typical_days:
            dropped = monthly["month"].iloc[-1].strftime("%Y-%m")
            monthly = monthly.iloc[:-1]
    out = monthly[["month"]].copy()
    out["y"] = monthly[MEASURES[measure]].astype(float)
    return out.reset_index(drop=True), dropped


def _features(months: pd.Series, start: pd.Timestamp) -> pd.DataFrame:
    t = (months.dt.year - start.year) * 12 + (months.dt.month - start.month)
    angle = 2 * np.pi * (months.dt.month - 1) / 12
    return pd.DataFrame({"t": t.astype(float), "month_sin": np.sin(angle), "month_cos": np.cos(angle)})


def _fit_predict(spark, train: pd.DataFrame, predict_months: pd.Series, cols: List[str], start: pd.Timestamp):
    """Fit Spark MLlib LinearRegression on `cols`; return (fitted values, predictions, residual std)."""
    train_feat = _features(train["month"], start).assign(y=train["y"].values)
    assembler = VectorAssembler(inputCols=cols, outputCol="features")
    train_vec = assembler.transform(spark.createDataFrame(train_feat))
    model = LinearRegression(featuresCol="features", labelCol="y", regParam=0.0).fit(train_vec)
    fitted = model.transform(train_vec).select("prediction").toPandas()["prediction"].values
    future = assembler.transform(spark.createDataFrame(_features(predict_months, start)))
    pred = model.transform(future).select("prediction").toPandas()["prediction"].values
    resid_std = float(np.std(train["y"].values - fitted, ddof=1)) if len(train) > 2 else 0.0
    return fitted, pred, resid_std


def _baseline(name: str, history: pd.DataFrame, months: pd.Series) -> np.ndarray:
    if name == "Naive (last month)":
        return np.repeat(history["y"].iloc[-1], len(months))
    lookup = dict(zip(history["month"], history["y"]))  # seasonal naive: same month a year earlier
    values = []
    for m in months:
        prev = m - pd.DateOffset(years=1)
        while prev not in lookup:  # beyond history: step back another year until known
            prev -= pd.DateOffset(years=1)
        values.append(lookup[prev])
    return np.array(values, dtype=float)


def _errors(actual: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(np.mean((actual - pred) ** 2)))
    nonzero = actual != 0
    mape = float(np.mean(np.abs((actual[nonzero] - pred[nonzero]) / actual[nonzero])) * 100) if nonzero.any() else float("nan")
    return {"RMSE": round(rmse, 2), "MAPE (%)": round(mape, 2)}


def forecast_monthly(silver_df: DataFrame, horizon: int = 6, holdout: int = 3, measure: str = "Revenue") -> Dict[str, object]:
    """Back-test candidate models on the last `holdout` months, then forecast `horizon` months ahead."""
    spark = silver_df.sparkSession
    series, dropped = monthly_series(silver_df, measure)
    n = len(series)
    if n < 6:
        return {"error": f"Need at least 6 complete months of data to forecast; this dataset has {n}."}
    holdout = max(1, min(holdout, n - 4))  # keep at least 4 months to train on
    train, test = series.iloc[:-holdout], series.iloc[-holdout:]
    start = series["month"].iloc[0]

    candidates = ["Naive (last month)", "Linear trend (MLlib)"]
    if len(train) >= 12:
        candidates += ["Seasonal naive", "Trend + seasonality (MLlib)"]

    scores, test_preds = [], {}
    for name in candidates:
        if "MLlib" in name:
            cols = SEASONAL if "seasonality" in name else TREND
            _, pred, _ = _fit_predict(spark, train, test["month"], cols, start)
        else:
            pred = _baseline(name, train, test["month"])
        test_preds[name] = pred
        scores.append({"Model": name, **_errors(test["y"].values, pred)})
    metrics = pd.DataFrame(scores).sort_values("RMSE", kind="stable").reset_index(drop=True)
    best = metrics["Model"].iloc[0]

    future_months = pd.Series(pd.date_range(series["month"].iloc[-1] + pd.DateOffset(months=1), periods=horizon, freq="MS"))
    if "MLlib" in best:
        cols = SEASONAL if "seasonality" in best else TREND
        fitted, future, resid_std = _fit_predict(spark, series, future_months, cols, start)
    else:
        fitted = np.full(n, np.nan)
        future = _baseline(best, series, future_months)
        resid_std = float(np.std(np.diff(series["y"].values), ddof=1)) if n > 2 else 0.0
    band = 1.96 * resid_std

    history = series.rename(columns={"y": "actual"}).assign(fitted=fitted, holdout_prediction=np.nan)
    history.loc[history.index[-holdout:], "holdout_prediction"] = test_preds[best]
    forecast = pd.DataFrame({"month": future_months, "forecast": future, "lower": future - band, "upper": future + band})
    if (series["y"] >= 0).all():  # sales and counts cannot go below zero
        forecast["lower"] = forecast["lower"].clip(lower=0)
    return {
        "history": history,
        "forecast": forecast,
        "metrics": metrics,
        "best_model": best,
        "holdout_months": holdout,
        "dropped_partial_month": dropped,
        "measure": measure,
    }


# ---------------------------------------------------------------------------------------
# Cohort analysis
# ---------------------------------------------------------------------------------------
def cohort_analysis(silver_df: DataFrame, max_periods: int = 12) -> Dict[str, pd.DataFrame]:
    """Monthly acquisition cohorts: retention %, cumulative revenue per customer and cohort sizes."""
    base = silver_df.filter(~F.col("is_cancellation") & ~F.col("is_return") & F.col("has_customer_id"))
    activity = base.select("customer_id", F.trunc("invoice_date", "month").alias("month"), "line_total")
    first = activity.groupBy("customer_id").agg(F.min("month").alias("cohort"))
    agg = (
        activity.join(first, "customer_id")
        .withColumn("period", F.round(F.months_between("month", "cohort")).cast("int"))
        .filter(F.col("period") <= max_periods)
        .groupBy("cohort", "period")
        .agg(F.countDistinct("customer_id").alias("customers"), F.sum("line_total").alias("revenue"))
        .toPandas()
    )
    if agg.empty:
        return {"error": "No customers with an ID and a completed purchase, so cohorts cannot be built."}
    last = pd.Timestamp(silver_df.agg(F.max(F.trunc("invoice_date", "month"))).collect()[0][0])
    agg["cohort"] = pd.to_datetime(agg["cohort"])

    sizes = agg[agg["period"] == 0].set_index("cohort")["customers"].sort_index()
    periods = list(range(max_periods + 1))
    customers = agg.pivot_table(index="cohort", columns="period", values="customers", aggfunc="sum").reindex(
        index=sizes.index, columns=periods)
    revenue = agg.pivot_table(index="cohort", columns="period", values="revenue", aggfunc="sum").reindex(
        index=sizes.index, columns=periods)

    # Observable periods get 0 when nobody bought; periods after the data ends stay empty
    observable = pd.DataFrame(
        {p: [(last.year - c.year) * 12 + (last.month - c.month) >= p for c in sizes.index] for p in periods},
        index=sizes.index,
    )
    customers = customers.fillna(0).where(observable)
    revenue = revenue.fillna(0).where(observable)

    retention = customers.div(sizes, axis=0) * 100
    ltv = revenue.cumsum(axis=1).where(observable).div(sizes, axis=0)
    weights = observable.mul(sizes, axis=0)
    avg_retention = (customers.sum() / weights.sum() * 100).where(weights.sum() > 0)

    label = sizes.index.strftime("%Y-%m")
    retention.index = label
    ltv.index = label
    return {
        "retention_pct": retention.round(1),
        "ltv_per_customer": ltv.round(2),
        "cohort_sizes": pd.DataFrame({"cohort": label, "new_customers": sizes.values}),
        "avg_retention": pd.DataFrame({"period": periods, "retention_pct": avg_retention.round(1).values}),
    }
