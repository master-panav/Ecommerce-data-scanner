from __future__ import annotations

import pandas as pd
import pytest

from data_tools import generate_rees46_sample, generate_uci_sample
from engine.analytics import clickstream_funnel, compute_kpis, monthly_trends, shopping_heatmap, top_products
from engine.ml_models import run_fpgrowth_basket, run_kmeans_segmentation
from engine.pipeline import EmptySilverError, load_saved_audits, load_silver, run_medallion_etl


@pytest.fixture(scope="module")
def uci_audit(spark, tmp_path_factory):
    root = tmp_path_factory.mktemp("lake")
    src = generate_uci_sample(root / "bronze" / "uci sample.csv", n_rows=8000, seed=7)
    return run_medallion_etl(spark, src, src.name, root), root


def test_etl_produces_partitioned_silver(spark, uci_audit):
    audit, _ = uci_audit
    assert audit.schema_type == "uci_online_retail"
    assert audit.raw_rows > audit.silver_valid_rows > 0
    assert audit.duplicate_rows > 0
    assert audit.cancellation_rows > 0
    df = load_silver(spark, audit.silver_path)
    assert {"year", "month", "line_total", "is_return", "has_customer_id"} <= set(df.columns)


def test_audits_survive_restart(uci_audit):
    audit, root = uci_audit
    restored = load_saved_audits(root)
    assert restored[audit.dataset_name] == audit


def test_gold_analytics(spark, uci_audit):
    audit, _ = uci_audit
    df = load_silver(spark, audit.silver_path)
    kpis = compute_kpis(df)
    assert kpis["gross_sales"] > 0 and kpis["orders"] > 0
    assert not monthly_trends(df).empty
    assert len(top_products(df, limit=5)) == 5
    assert not shopping_heatmap(df).empty


def test_ml_models(spark, uci_audit):
    audit, _ = uci_audit
    df = load_silver(spark, audit.silver_path)
    km = run_kmeans_segmentation(df, k_selected=3)
    assert "error" not in km
    assert km["profile_df"]["customers"].sum() == km["customer_count"]
    fp = run_fpgrowth_basket(df, min_support=0.02, min_confidence=0.2)
    assert "error" not in fp and not fp["rules_df"].empty


def test_rees46_funnel(spark, tmp_path):
    src = generate_rees46_sample(tmp_path / "bronze" / "clicks.csv", n_events=3000)
    audit = run_medallion_etl(spark, src, src.name, tmp_path)
    funnel = clickstream_funnel(spark, audit.events_path)
    assert funnel["event_type"].iloc[0] == "view"
    assert "purchase" in set(funnel["event_type"])


def test_utc_event_times_are_not_shifted(spark, tmp_path):
    src = tmp_path / "rees.csv"
    pd.DataFrame(
        {"event_time": ["2019-10-01 00:30:00 UTC"], "event_type": ["purchase"], "product_id": [1], "category_code": ["a"],
         "brand": ["b"], "price": [1.0], "user_id": [1], "user_session": ["s"]}
    ).to_csv(src, index=False)
    audit = run_medallion_etl(spark, src, src.name, tmp_path)
    row = load_silver(spark, audit.silver_path).select("hour").collect()[0]
    assert row["hour"] == 0


def test_similar_names_get_separate_silver_tables():
    from engine.pipeline import dataset_slug

    slugs = {dataset_slug(n) for n in ["Sales.csv", "sales_csv", "sales csv"]}
    assert len(slugs) == 3


def test_saved_audits_survive_moving_the_project(spark, uci_audit, tmp_path):
    import shutil

    audit, root = uci_audit
    moved = tmp_path / "moved_lake"
    shutil.copytree(root / "silver", moved / "silver")
    restored = load_saved_audits(moved)[audit.dataset_name]
    assert restored.silver_path.startswith(str(moved))
    assert load_silver(spark, restored.silver_path).count() == audit.silver_valid_rows


def test_rows_without_valid_dates_fail_loudly(spark, tmp_path):
    src = tmp_path / "bad_dates.csv"
    pd.DataFrame(
        {"InvoiceNo": ["1", "2"], "InvoiceDate": ["not a date", "??"], "UnitPrice": [1.0, 2.0], "Quantity": [1, 1]}
    ).to_csv(src, index=False)
    with pytest.raises(EmptySilverError, match="no parseable date"):
        run_medallion_etl(spark, src, src.name, tmp_path)
