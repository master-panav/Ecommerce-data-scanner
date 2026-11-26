from __future__ import annotations

import pandas as pd
import pytest

from engine.schemas import SchemaMappingError, _detect_text_encoding, detect_schema_type, load_raw_dataset


@pytest.mark.parametrize(
    "columns, expected",
    [
        (["InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID", "Country"], "uci_online_retail"),
        (["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "Customer ID", "Country"], "uci_online_retail_ii"),
        (["event_time", "event_type", "product_id", "user_id", "user_session", "price"], "rees46_clickstream"),
        (["invoice_no", "unit_price", "customer_id", "invoice_date"], "canonical"),
        (["order_date", "sku", "qty", "price", "order_id"], "generic_retail"),
    ],
)
def test_detect_schema_type(columns, expected):
    assert detect_schema_type(columns) == expected


def test_latin1_csv_is_detected(tmp_path):
    path = tmp_path / "latin1.csv"
    path.write_bytes("InvoiceNo,Description\n1,CAF\xc9 MUG\n".encode("latin-1"))
    assert _detect_text_encoding(path) == "ISO-8859-1"


def test_utf8_csv_is_detected(tmp_path):
    path = tmp_path / "utf8.csv"
    path.write_text("InvoiceNo,Description\n1,CAFÉ MUG\n", encoding="utf-8")
    assert _detect_text_encoding(path) == "UTF-8"


def test_unmappable_upload_gives_clear_error(spark, tmp_path):
    path = tmp_path / "not_sales.csv"
    pd.DataFrame({"name": ["a", "b"], "score": [1, 2]}).to_csv(path, index=False)
    with pytest.raises(SchemaMappingError, match="Could not recognise the columns"):
        load_raw_dataset(spark, path)


def test_lone_data_dictionary_gives_clear_error(spark, tmp_path):
    path = tmp_path / "DescriptionDataCoSupplyChain.csv"
    pd.DataFrame({"FIELDS": ["Type", "Days for shipping (real)"], "DESCRIPTION": ["Type of transaction", "Actual days"]}).to_csv(path, index=False)
    with pytest.raises(SchemaMappingError, match="data dictionary"):
        load_raw_dataset(spark, path)


def test_data_dictionary_redirects_to_the_dataset_it_describes(spark, tmp_path):
    fields = ["Type", "order date (DateOrders)", "Order Id", "Order Item Quantity", "Sales", "Order Customer Id", "Order Status"]
    pd.DataFrame({"FIELDS": fields, "DESCRIPTION": ["..."] * len(fields)}).to_csv(tmp_path / "DescriptionDataCoSupplyChain.csv", index=False)
    pd.DataFrame({
        "Type": ["DEBIT", "CASH"], "order date (DateOrders)": ["1/31/2018 22:56", "2/1/2018 10:00"], "Order Id": [1, 2],
        "Order Item Quantity": [1, 2], "Sales": [327.75, 99.5], "Order Customer Id": [20755, 19492], "Order Status": ["COMPLETE", "CANCELED"],
    }).to_csv(tmp_path / "DataCoSupplyChainDataset.csv", index=False)
    # A bigger sibling that is not the documented dataset must not be picked
    pd.DataFrame({"Product": ["x"] * 500, "Date": ["9/1/2017 6:00"] * 500, "ip": ["1.1.1.1"] * 500}).to_csv(tmp_path / "tokenized_access_logs.csv", index=False)

    df, _, report = load_raw_dataset(spark, tmp_path / "DescriptionDataCoSupplyChain.csv")
    assert report["redirected_from"] == "DescriptionDataCoSupplyChain.csv"
    assert report["source_file"] == "DataCoSupplyChainDataset.csv"
    assert df.count() == 2


def test_excel_with_mixed_invoice_column(spark, tmp_path):
    path = tmp_path / "mixed.xlsx"
    pd.DataFrame(
        {
            "InvoiceNo": [536365, "C536379", 536366],
            "StockCode": ["85123A", 71053, "22423"],
            "Description": ["MUG", None, "CAKESTAND"],
            "Quantity": [6, -1, 2],
            "InvoiceDate": pd.to_datetime(["2010-12-01 08:26", "2010-12-01 09:41", "2010-12-01 10:00"]),
            "UnitPrice": [2.55, 27.5, 12.75],
            "CustomerID": [17850.0, None, 13047.0],
            "Country": ["United Kingdom", "United Kingdom", "France"],
        }
    ).to_excel(path, index=False)

    df, schema, _ = load_raw_dataset(spark, path)
    rows = {r["invoice_no"]: r for r in df.collect()}

    assert schema == "uci_online_retail"
    assert set(rows) == {"536365", "C536379", "536366"}
    assert rows["536365"]["customer_id"] == "17850"
    assert rows["C536379"]["customer_id"] is None
    assert rows["536366"]["invoice_date"] is not None


def test_excel_numeric_ids_with_blanks_are_not_floats(spark, tmp_path):
    path = tmp_path / "blank_ids.xlsx"
    pd.DataFrame(
        {
            "InvoiceNo": [536365, None, 536366],
            "InvoiceDate": pd.to_datetime(["2011-01-01"] * 3),
            "StockCode": [85123, 85124, None],
            "Quantity": [1, 2, 3],
            "UnitPrice": [1.0, 2.0, 3.0],
            "CustomerID": [17850.0, None, 13047.0],
            "Country": ["UK"] * 3,
        }
    ).to_excel(path, index=False)
    df, _, _ = load_raw_dataset(spark, path)
    rows = df.orderBy("unit_price").collect()
    assert [r["invoice_no"] for r in rows] == ["536365", None, "536366"]
    assert [r["stock_code"] for r in rows] == ["85123", "85124", None]
    assert [r["customer_id"] for r in rows] == ["17850", None, "13047"]
