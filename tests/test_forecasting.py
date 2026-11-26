"""Forecasting and cohort analysis on series with known answers."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from engine.forecasting import cohort_analysis, forecast_monthly, monthly_series


def _silver(spark, rows):
    pdf = pd.DataFrame(rows, columns=["invoice_date", "line_total", "invoice_no", "customer_id"])
    pdf["invoice_date"] = pd.to_datetime(pdf["invoice_date"])
    pdf["is_cancellation"] = False
    pdf["is_return"] = False
    pdf["has_customer_id"] = pdf["customer_id"].notna()
    return spark.createDataFrame(pdf)


def _trend_rows(months: int, partial_last_days: int | None = None):
    rows = []
    for t, month in enumerate(pd.date_range("2020-01-01", periods=months, freq="MS")):
        days = partial_last_days if (partial_last_days and t == months - 1) else 20
        for d in range(days):
            rows.append((month + pd.Timedelta(days=d), (1000 + 50 * t) / days, f"{t}-{d}", f"c{d}"))
    return rows


def test_linear_trend_is_learned_and_extrapolated(spark):
    df = _silver(spark, _trend_rows(24))
    res = forecast_monthly(df, horizon=3, holdout=3)
    assert res["best_model"] == "Linear trend (MLlib)"
    assert res["metrics"].iloc[0]["RMSE"] < 1e-3
    assert res["forecast"]["forecast"].round(1).tolist() == [2200.0, 2250.0, 2300.0]
    f = res["forecast"]
    assert (f["lower"] <= f["forecast"]).all() and (f["forecast"] <= f["upper"]).all()
    assert set(res["metrics"]["Model"]) == {
        "Naive (last month)", "Linear trend (MLlib)", "Seasonal naive", "Trend + seasonality (MLlib)"}


def test_partial_last_month_is_dropped(spark):
    df = _silver(spark, _trend_rows(10, partial_last_days=3))
    series, dropped = monthly_series(df)
    assert dropped == "2020-10"
    assert len(series) == 9


def test_too_little_history_is_reported(spark):
    res = forecast_monthly(_silver(spark, _trend_rows(4)))
    assert "at least 6 complete months" in res["error"]


def test_cohort_retention_matches_hand_calculation(spark):
    rows = [
        ("2021-01-05", 10.0, "1", "A"), ("2021-01-09", 20.0, "2", "B"),
        ("2021-02-03", 30.0, "3", "A"), ("2021-02-04", 5.0, "4", "C"),
        ("2021-03-10", 5.0, "5", "C"),
        ("2021-04-02", 40.0, "6", "A"),
    ]
    res = cohort_analysis(_silver(spark, rows), max_periods=3)
    ret = res["retention_pct"]
    assert ret.loc["2021-01"].tolist() == [100.0, 50.0, 0.0, 50.0]  # A and B joined; A returns in Feb and Apr
    feb = ret.loc["2021-02"].tolist()
    assert feb[:3] == [100.0, 100.0, 0.0] and math.isnan(feb[3])  # Feb+3 = May is after the data ends
    assert res["cohort_sizes"]["new_customers"].tolist() == [2, 1]
    ltv = res["ltv_per_customer"].loc["2021-01"].tolist()
    assert ltv == pytest.approx([15.0, 30.0, 30.0, 50.0])  # cumulative revenue / 2 customers
