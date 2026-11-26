"""Ingestion robustness: unusual headers, dirty values and mixed types (run under ANSI mode)."""

from __future__ import annotations

import pandas as pd

from data_tools import generate_rees46_sample
from engine.pipeline import load_silver, run_medallion_etl
from engine.schemas import load_raw_dataset


def test_rees46_text_brand_with_numeric_product_id(spark, tmp_path):
    # Regression: coalesce(brand "apple", product_id BIGINT) raised CAST_INVALID_INPUT under ANSI
    src = generate_rees46_sample(tmp_path / "bronze" / "rees.csv", n_events=3000)
    audit = run_medallion_etl(spark, src, src.name, tmp_path)
    assert audit.silver_valid_rows > 0
    assert audit.events_path is not None


def test_dirty_retail_csv_is_cleaned_before_processing(spark, tmp_path):
    src = tmp_path / "dirty.csv"
    rows = [
        ["Invoice No", "Order Date (UTC)", "Item Name", "Qty", "Unit Price", "Customer", "Region", "Notes"],
        ["1001", "2011-01-05T10:00:00", "Mug", "2", "£1,234.50", "C1", "UK", ""],
        ["1001", "2011-01-05T10:00:00", "Mug", "2", "£1,234.50", "C1", "UK", ""],  # exact duplicate
        ["1002", "05/01/2011 11:30", "Lamp", "abc", "3.00", "N/A", "France", ""],  # malformed qty, placeholder id
        ["", "", "", "", "", "", "", ""],  # fully empty row
        ["1003", "not a date", "Cup", "1", "2.50", "C2", "UK", ""],
        ["1004", "2011-02-01 09:00", "Plate", "3", "12,50", "C3", "null", ""],  # decimal comma, null token
    ]
    src.write_text("﻿" + "\n".join(",".join(f'"{v}"' for v in r) for r in rows), encoding="utf-8")

    audit = run_medallion_etl(spark, src, src.name, tmp_path)
    assert audit.empty_rows_removed == 1
    assert audit.empty_columns_removed == ["Notes"]
    assert audit.duplicate_rows == 1
    assert audit.malformed_quantity_rows == 1
    assert audit.null_date_rows == 1
    assert audit.missing_customer_rows == 1
    assert audit.column_mapping["invoice_date"] == "Order Date (UTC)"

    silver = {r["invoice_no"]: r for r in load_silver(spark, audit.silver_path).collect()}
    assert set(silver) == {"1001", "1004"}
    assert silver["1001"]["unit_price"] == 1234.5
    assert silver["1004"]["unit_price"] == 12.5
    assert silver["1004"]["country"] == "Unknown"


def test_amount_only_file_derives_unit_price(spark, tmp_path):
    src = tmp_path / "amounts.csv"
    pd.DataFrame(
        {"order_id": ["A", "B"], "date": ["2012-03-01", "2012-03-02"], "quantity": [4, 2], "sales": [20.0, 9.0]}
    ).to_csv(src, index=False)
    df, _, prep = load_raw_dataset(spark, src)
    prices = sorted(r["unit_price"] for r in df.collect())
    assert prices == [4.5, 5.0]
    assert prep["column_mapping"]["line_total"] == "sales"


def test_generic_clickstream_without_invoice_column(spark, tmp_path):
    src = tmp_path / "events.csv"
    pd.DataFrame(
        {
            "timestamp": ["2020-01-01 10:00:00", "2020-01-01 10:01:00", "2020-01-01 10:02:00"],
            "action": ["page_view", "add_to_cart", "buy"],
            "user": [1, 1, 1],
            "sku": [10, 10, 10],
            "price": ["9.99", "9.99", "9.99"],
        }
    ).to_csv(src, index=False)
    audit = run_medallion_etl(spark, src, src.name, tmp_path)
    assert audit.schema_type == "clickstream"
    assert audit.silver_valid_rows == 1  # only the purchase becomes a sales transaction
    events = {r["event_type"] for r in load_silver(spark, audit.events_path).collect()}
    assert events == {"view", "cart", "purchase"}


def test_duplicate_and_blank_headers(spark, tmp_path):
    src = tmp_path / "headers.csv"
    src.write_text("InvoiceNo,InvoiceDate,Price,Price,\n1,2011-01-01,2.0,3.0,x\n", encoding="utf-8")
    df, _, prep = load_raw_dataset(spark, src)
    assert df.count() == 1
    assert prep["column_mapping"]["unit_price"].startswith("Price")  # Spark renames duplicates to Price2/Price3
