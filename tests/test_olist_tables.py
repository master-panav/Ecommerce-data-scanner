"""Olist ships as separate related CSVs: single tables, the full set and the translation lookup."""
from __future__ import annotations

import pandas as pd
import pytest

from data_tools import generate_olist_sample
from engine.schemas import SchemaMappingError, load_raw_dataset


def _translation(path):
    pd.DataFrame({
        "product_category_name": ["bed_bath_table", "health_beauty"],
        "product_category_name_english": ["Bed Bath Table", "Health Beauty"],
    }).to_csv(path / "product_category_name_translation.csv", index=False)


def test_lone_translation_table_names_the_missing_tables(spark, tmp_path):
    _translation(tmp_path)
    with pytest.raises(SchemaMappingError, match="olist_orders_dataset.csv"):
        load_raw_dataset(spark, tmp_path / "product_category_name_translation.csv")


def test_any_olist_table_loads_the_whole_set_with_english_categories(spark, tmp_path):
    generate_olist_sample(tmp_path, n_orders=200, seed=1)
    _translation(tmp_path)

    df, label, report = load_raw_dataset(spark, tmp_path / "product_category_name_translation.csv")
    assert label == "olist_brazil"
    assert "5-table join" in report["column_mapping"]["source"]
    categories = {r[0] for r in df.select("category").distinct().collect()}
    assert {"Bed Bath Table", "Health Beauty"} <= categories  # translated
    assert "sports_leisure" in categories  # untranslated names are kept, not dropped
