"""Wide (20-50+ column) public datasets: column mapping, cleaning and BI on header-faithful samples."""

from __future__ import annotations

import pandas as pd
import pytest

from engine.analytics import compute_kpis, explore, explorer_fields
from engine.pipeline import load_silver, run_medallion_etl
from engine.schemas import load_raw_dataset

DATACO_COLUMNS = [
    "Type", "Days for shipping (real)", "Days for shipment (scheduled)", "Benefit per order", "Sales per customer",
    "Delivery Status", "Late_delivery_risk", "Category Id", "Category Name", "Customer City", "Customer Country",
    "Customer Email", "Customer Fname", "Customer Id", "Customer Lname", "Customer Password", "Customer Segment",
    "Customer State", "Customer Street", "Customer Zipcode", "Department Id", "Department Name", "Latitude",
    "Longitude", "Market", "Order City", "Order Country", "Order Customer Id", "order date (DateOrders)", "Order Id",
    "Order Item Cardprod Id", "Order Item Discount", "Order Item Discount Rate", "Order Item Id",
    "Order Item Product Price", "Order Item Profit Ratio", "Order Item Quantity", "Sales", "Order Item Total",
    "Order Profit Per Order", "Order Region", "Order State", "Order Status", "Order Zipcode", "Product Card Id",
    "Product Category Id", "Product Description", "Product Image", "Product Name", "Product Price", "Product Status",
    "shipping date (DateOrders)", "Shipping Mode",
]


def _dataco_rows(n: int = 60) -> pd.DataFrame:
    rows = []
    for i in range(n):
        qty, price, disc = 1 + i % 4, 49.99 + i, 5.0
        status = "CANCELED" if i % 10 == 0 else "COMPLETE"
        rows.append({
            "Type": "DEBIT", "Days for shipping (real)": 3 + i % 3, "Days for shipment (scheduled)": 4,
            "Benefit per order": 20.5, "Sales per customer": qty * price - disc,
            "Delivery Status": "Late delivery" if i % 3 else "Advance shipping", "Late_delivery_risk": int(i % 3 > 0),
            "Category Id": 17, "Category Name": ["Cleats", "Men's Footwear", "Women's Apparel"][i % 3],
            "Customer City": "Caguas", "Customer Country": "EE. UU.", "Customer Email": "XXXXXXXXX",
            "Customer Fname": "Cally", "Customer Id": 20000 + i % 15, "Customer Lname": "Holloway",
            "Customer Password": "XXXXXXXXX", "Customer Segment": ["Consumer", "Corporate", "Home Office"][i % 3],
            "Customer State": "PR", "Customer Street": "5365 Noble Nectar Island", "Customer Zipcode": 725,
            "Department Id": 2, "Department Name": "Fitness", "Latitude": 18.25, "Longitude": -66.04,
            "Market": ["Pacific Asia", "LATAM", "Europe"][i % 3], "Order City": "Bekasi",
            "Order Country": ["Indonesia", "México", "Francia"][i % 3], "Order Customer Id": 20000 + i % 15,
            "order date (DateOrders)": f"{1 + i % 12}/{1 + i % 27}/2017 {i % 24}:{i % 60:02d}",
            "Order Id": 77000 + i // 2, "Order Item Cardprod Id": 1360, "Order Item Discount": disc,
            "Order Item Discount Rate": 0.04, "Order Item Id": 180000 + i, "Order Item Product Price": price,
            "Order Item Profit Ratio": 0.29, "Order Item Quantity": qty, "Sales": qty * price,
            "Order Item Total": qty * price - disc, "Order Profit Per Order": 20.5,
            "Order Region": "Southeast Asia", "Order State": "Java Occidental", "Order Status": status,
            "Order Zipcode": None, "Product Card Id": 1360 + i % 5, "Product Category Id": 73,
            "Product Description": None, "Product Image": "http://images.example/1360.jpg",
            "Product Name": f"Smart watch {i % 5}", "Product Price": price, "Product Status": 0,
            "shipping date (DateOrders)": "2/3/2018 22:56", "Shipping Mode": "Standard Class",
        })
    return pd.DataFrame(rows, columns=DATACO_COLUMNS)


@pytest.fixture(scope="module")
def dataco_audit(spark, tmp_path_factory):
    root = tmp_path_factory.mktemp("dataco")
    src = root / "DataCoSupplyChainDataset.csv"
    _dataco_rows().to_csv(src, index=False, encoding="latin-1")  # the real file is Latin-1 encoded
    return run_medallion_etl(spark, src, src.name, root)


def test_dataco_53_columns_are_mapped(dataco_audit):
    m = dataco_audit.column_mapping
    assert dataco_audit.schema_type == "dataco_supply_chain"
    assert m["invoice_date"] == "order date (DateOrders)"
    assert m["invoice_no"] == "Order Id"
    assert m["customer_id"] == "Customer Id"
    assert m["stock_code"] == "Product Card Id"
    assert m["quantity"] == "Order Item Quantity"
    assert m["unit_price"] == "Order Item Product Price"
    assert m["line_total"] == "Order Item Total"
    assert m["profit"] == "Order Profit Per Order"
    assert m["status"] == "Order Status"
    assert m["country"] == "Order Country"
    assert m["category"] == "Category Name"
    assert m["description"] == "Product Name"
    assert dataco_audit.empty_columns_removed == ["Order Zipcode", "Product Description"]
    assert len(dataco_audit.column_profile) == 53
    assert "x_shipping_mode" in dataco_audit.extra_columns
    assert "x_days_for_shipping_real" in dataco_audit.extra_columns


def test_dataco_cancellations_profit_and_discounted_revenue(spark, dataco_audit):
    assert dataco_audit.status_cancellation_rows == 6  # every 10th row is CANCELED
    df = load_silver(spark, dataco_audit.silver_path)
    kpis = compute_kpis(df)
    completed = _dataco_rows()
    completed = completed[completed["Order Status"] != "CANCELED"]
    assert kpis["gross_sales"] == pytest.approx(completed["Order Item Total"].sum(), abs=0.05)
    assert kpis["profit"] == pytest.approx(completed["Order Profit Per Order"].sum(), abs=0.05)


def test_dataco_slice_and_dice(spark, dataco_audit):
    df = load_silver(spark, dataco_audit.silver_path)
    dims, numeric = explorer_fields(df)
    assert {"x_customer_segment", "x_delivery_status", "x_market"} <= set(dims)
    assert {"profit", "x_days_for_shipping_real", "x_late_delivery_risk"} <= set(numeric)
    late = explore(df, "x_market", "x_late_delivery_risk", "Average")
    assert set(late["x_market"]) == {"Pacific Asia", "LATAM", "Europe"}
    pivot = explore(df, "x_customer_segment", "Revenue", column_dimension="x_market")
    assert len(pivot) == 3  # each segment appears with exactly one market in this sample


def test_pakistan_ecommerce_layout(spark, tmp_path):
    header = ["item_id", "status", "created_at", "sku", "price", "qty_ordered", "grand_total", "increment_id",
              "category_name_1", "sales_commission_code", "discount_amount", "payment_method", "Working Date",
              "BI Status", " MV ", "Year", "Month", "Customer Since", "M-Y", "FY", "Customer ID", "", "", "", "", ""]
    lines = [",".join(header)]
    statuses = ["complete", "canceled", "order_refunded", "received", "\\N"]
    for i in range(40):
        lines.append(",".join([
            str(211131 + i), statuses[i % 5], f"7/{1 + i % 28}/2016", f"kreations_YI {i % 6}", "1950", str(1 + i % 2),
            "1950", str(100147443 + i), ["Women's Fashion", "Beauty & Grooming", "Mobiles & Tablets"][i % 3], "\\N", "0",
            ["cod", "ublcreditcard", "Easypay"][i % 3], "7/1/2016", "#REF!", '" 1,950 "', "2016", "7", "2016-7",
            "7-2016", "FY17", str(1 + i % 9), "", "", "", "", "",
        ]))
    lines += [",,,,,,,,,,,,,,,,,,,,,,,,,"] * 15  # the real file ends with many fully empty rows
    src = tmp_path / "Pakistan Largest Ecommerce Dataset.csv"
    src.write_text("\n".join(lines), encoding="utf-8")

    audit = run_medallion_etl(spark, src, src.name, tmp_path)
    m = audit.column_mapping
    assert audit.schema_type == "pakistan_ecommerce"
    assert (m["invoice_no"], m["invoice_date"], m["stock_code"], m["quantity"]) == (
        "increment_id", "created_at", "sku", "qty_ordered")
    assert m["category"] == "category_name_1" and m["status"] == "status" and m["customer_id"] == "Customer ID"
    assert audit.empty_rows_removed == 15
    assert len(audit.empty_columns_removed) == 5  # the 5 blank-header columns
    assert audit.status_cancellation_rows == 16  # canceled + order_refunded
    df = load_silver(spark, audit.silver_path)
    dims, _ = explorer_fields(df)
    assert "x_payment_method" in dims


@pytest.mark.parametrize("style", ["snake", "title"])
def test_superstore_layout_derives_price_from_sales(spark, tmp_path, style):
    snake = ["order_id", "order_date", "ship_date", "ship_mode", "customer_name", "segment", "state", "country",
             "market", "region", "product_id", "category", "sub_category", "product_name", "sales", "quantity",
             "discount", "profit", "shipping_cost", "order_priority", "year"]
    title = ["Order ID", "Order Date", "Ship Date", "Ship Mode", "Customer Name", "Segment", "State", "Country",
             "Market", "Region", "Product ID", "Category", "Sub-Category", "Product Name", "Sales", "Quantity",
             "Discount", "Profit", "Shipping Cost", "Order Priority", "Year"]
    rows = []
    for i in range(30):
        rows.append([f"AG-2011-{2000 + i // 2}", f"{1 + i % 28}-0{1 + i % 9}-2011", "06-01-2011", "Standard Class",
                     f"Customer {i % 7}", ["Consumer", "Corporate"][i % 2], "Constantine", "Algeria", "Africa", "Africa",
                     f"OFF-TEN-{i % 4}", ["Office Supplies", "Technology"][i % 2], "Storage", f"Tenex Lockers {i % 4}",
                     f"{1000 + i:,}", str(2 + i % 3), "0.1", "106.14", "35.46", "Medium", "2011"])
    src = tmp_path / f"superstore_{style}.csv"
    pd.DataFrame(rows, columns=snake if style == "snake" else title).to_csv(src, index=False)

    df, schema, prep = load_raw_dataset(spark, src)
    m = prep["column_mapping"]
    assert schema == "superstore_sales"
    assert {"invoice_date", "invoice_no", "customer_id", "quantity", "line_total", "profit", "discount"} <= set(m)
    assert "unit_price" not in m
    first = df.orderBy("invoice_no", "stock_code").first()
    assert first["source_total"] == 1000.0  # "1,000" parsed despite the thousands comma
    assert first["unit_price"] == pytest.approx(1000.0 / first["quantity"])
    assert first["invoice_date"].month == 1 and first["invoice_date"].day == 1  # d-M-yyyy
