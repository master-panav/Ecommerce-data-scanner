"""Sample Big Data generator (for instant offline testing) & UCI dataset downloader."""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

UCI_URLS = {
    "online_retail.xlsx": "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx",
    "online_retail_II.xlsx": "https://archive.ics.uci.edu/ml/machine-learning-databases/00502/online_retail_II.xlsx",
}

PRODUCTS = [
    ("85123A", "WHITE HANGING HEART T-LIGHT HOLDER", 2.95, "Home Decor"),
    ("71053", "WHITE METAL LANTERN", 3.75, "Lighting"),
    ("84406B", "CREAM CUPID HEARTS COAT HANGER", 4.15, "Home Decor"),
    ("22423", "REGENCY CAKESTAND 3 TIER", 12.75, "Kitchen"),
    ("47566", "PARTY BUNTING", 4.95, "Party"),
    ("20725", "LUNCH BAG RED RETROSPOT", 1.65, "Bags"),
    ("22720", "SET OF 3 CAKE TINS PANTRY DESIGN", 4.95, "Kitchen"),
    ("21212", "PACK OF 72 RETROSPOT CAKE CASES", 0.55, "Baking"),
    ("22197", "POPCORN HOLDER", 0.85, "Party"),
    ("23203", "JUMBO BAG VINTAGE DOILY", 2.08, "Bags"),
    ("22960", "JAM MAKING SET WITH JARS", 4.25, "Kitchen"),
    ("22633", "HAND WARMER UNION JACK", 2.10, "Gifts"),
]

COUNTRIES = ["United Kingdom", "Germany", "France", "EIRE", "Spain", "Netherlands", "Belgium", "Switzerland", "Australia"]
COUNTRY_WEIGHTS = [0.76, 0.05, 0.05, 0.03, 0.025, 0.025, 0.02, 0.02, 0.02]


def generate_uci_sample(out_path: Path, n_rows: int = 50000, year: int = 2011, retail_ii_schema: bool = False, seed: int = 42) -> Path:
    """Generate a realistic UCI Online Retail I or II style CSV with cancellations, returns, and baskets."""
    rng = np.random.default_rng(seed)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_invoices = max(n_rows // 4, 2000)
    customers = [f"{12345 + i}" for i in range(1800)]
    start = pd.Timestamp(f"{year}-01-01 08:00:00")
    end = pd.Timestamp(f"{year}-12-20 18:00:00")
    span_sec = int((end - start).total_seconds())

    rows = []
    for inv_idx in range(n_invoices):
        is_cancel = rng.random() < 0.035
        inv_no = f"C{540000 + inv_idx}" if is_cancel else f"{540000 + inv_idx}"
        ts = start + pd.Timedelta(seconds=int(rng.integers(0, span_sec)))
        # Realistic daytime shopping bias
        ts = ts.replace(hour=int(rng.choice(range(8, 20), p=[0.04, 0.08, 0.11, 0.13, 0.14, 0.12, 0.11, 0.10, 0.08, 0.05, 0.025, 0.015])))
        cust_id = None if rng.random() < 0.12 else rng.choice(customers)
        country = rng.choice(COUNTRIES, p=COUNTRY_WEIGHTS)

        basket_size = int(rng.choice([1, 2, 3, 4, 5, 6], p=[0.18, 0.25, 0.24, 0.16, 0.10, 0.07]))
        picked = rng.choice(len(PRODUCTS), size=basket_size, replace=False)

        # Inject realistic co-occurrence for FP-Growth (Kitchen & Baking pairs)
        if 3 in picked and rng.random() < 0.65 and 6 not in picked:
            picked = np.append(picked, 6)
        if 5 in picked and rng.random() < 0.60 and 9 not in picked:
            picked = np.append(picked, 9)

        for p_idx in picked:
            code, desc, price, _ = PRODUCTS[p_idx]
            qty = -int(rng.integers(1, 6)) if is_cancel else int(rng.integers(1, 18))
            unit_p = 0.0 if rng.random() < 0.005 else round(price * rng.uniform(0.95, 1.08), 2)
            rows.append((inv_no, code, desc, qty, ts.strftime("%Y-%m-%d %H:%M:%S"), unit_p, cust_id, country))

    cols = (
        ["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "Customer ID", "Country"]
        if retail_ii_schema
        else ["InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID", "Country"]
    )
    df = pd.DataFrame(rows[:n_rows], columns=cols)
    # Inject ~1% exact duplicates to test audit log
    dups = df.sample(frac=0.01, random_state=seed)
    df = pd.concat([df, dups], ignore_index=True)
    df.to_csv(out_path, index=False)
    return out_path


def generate_rees46_sample(out_path: Path, n_events: int = 60000, seed: int = 99) -> Path:
    """Generate a realistic REES46 clickstream funnel CSV (view -> cart -> remove -> purchase)."""
    rng = np.random.default_rng(seed)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    categories = ["electronics.smartphone", "electronics.audio.headphone", "appliances.kitchen.kettle", "computers.notebook", "apparel.shoes"]
    brands = ["samsung", "apple", "sony", "xiaomi", "bosch", "nike", "asus"]
    start = pd.Timestamp("2019-10-01 00:00:00")

    n_sessions = n_events // 3
    rows = []
    for s in range(n_sessions):
        uid = int(rng.integers(500000000, 500003000))
        sess = f"sess-{s:07d}"
        base_ts = start + pd.Timedelta(seconds=int(rng.integers(0, 60 * 86400)))
        prod_id = int(rng.integers(100001, 100120))
        cat = rng.choice(categories)
        brand = rng.choice(brands)
        price = round(float(rng.uniform(25, 950)), 2)

        rows.append((base_ts.strftime("%Y-%m-%d %H:%M:%S UTC"), "view", prod_id, 2000 + (prod_id % 10), cat, brand, price, uid, sess))
        if rng.random() < 0.28:
            rows.append(((base_ts + pd.Timedelta(minutes=2)).strftime("%Y-%m-%d %H:%M:%S UTC"), "cart", prod_id, 2000 + (prod_id % 10), cat, brand, price, uid, sess))
            if rng.random() < 0.22:
                rows.append(((base_ts + pd.Timedelta(minutes=4)).strftime("%Y-%m-%d %H:%M:%S UTC"), "remove_from_cart", prod_id, 2000 + (prod_id % 10), cat, brand, price, uid, sess))
            elif rng.random() < 0.55:
                rows.append(((base_ts + pd.Timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S UTC"), "purchase", prod_id, 2000 + (prod_id % 10), cat, brand, price, uid, sess))

    df = pd.DataFrame(rows[:n_events], columns=[
        "event_time", "event_type", "product_id", "category_id", "category_code", "brand", "price", "user_id", "user_session"
    ])
    df.to_csv(out_path, index=False)
    return out_path


def generate_olist_sample(out_dir: Path, n_orders: int = 12000, seed: int = 77) -> Path:
    """Generate a 4-table Olist relational CSV folder to demonstrate distributed Spark Joins."""
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    states = ["SP", "RJ", "MG", "RS", "PR", "BA", "SC"]
    cats = ["bed_bath_table", "health_beauty", "sports_leisure", "computers_accessories", "furniture_decor", "watches_gifts"]

    prod_ids = [f"prod_{i:04d}" for i in range(150)]
    products = pd.DataFrame({
        "product_id": prod_ids,
        "product_category_name": rng.choice(cats, size=len(prod_ids)),
    })

    unique_custs = [f"u_{i:05d}" for i in range(4500)]
    order_ids = [f"ord_{i:06d}" for i in range(n_orders)]
    cust_order_ids = [f"c_{i:06d}" for i in range(n_orders)]
    start = pd.Timestamp("2017-01-01")
    ts_list = [(start + pd.Timedelta(seconds=int(rng.integers(0, 600 * 86400)))).strftime("%Y-%m-%d %H:%M:%S") for _ in range(n_orders)]

    orders = pd.DataFrame({
        "order_id": order_ids,
        "customer_id": cust_order_ids,
        "order_status": rng.choice(["delivered", "shipped", "canceled"], size=n_orders, p=[0.93, 0.04, 0.03]),
        "order_purchase_timestamp": ts_list,
    })
    customers = pd.DataFrame({
        "customer_id": cust_order_ids,
        "customer_unique_id": rng.choice(unique_custs, size=n_orders),
        "customer_state": rng.choice(states, size=n_orders, p=[0.42, 0.13, 0.12, 0.09, 0.08, 0.08, 0.08]),
    })

    item_rows = []
    for oid in order_ids:
        for idx in range(1, int(rng.choice([1, 2, 3], p=[0.7, 0.22, 0.08])) + 1):
            item_rows.append((oid, idx, rng.choice(prod_ids), round(float(rng.uniform(19.9, 490.0)), 2)))
    items = pd.DataFrame(item_rows, columns=["order_id", "order_item_id", "product_id", "price"])

    orders.to_csv(out_dir / "olist_orders_dataset.csv", index=False)
    customers.to_csv(out_dir / "olist_customers_dataset.csv", index=False)
    products.to_csv(out_dir / "olist_products_dataset.csv", index=False)
    items.to_csv(out_dir / "olist_order_items_dataset.csv", index=False)
    return out_dir


def download_uci_datasets(target_dir: Path = Path("data_lake/bronze")) -> None:
    """Download the official UCI Online Retail & Online Retail II Excel workbooks."""
    target_dir.mkdir(parents=True, exist_ok=True)
    for fname, url in UCI_URLS.items():
        dest = target_dir / fname
        if dest.exists():
            print(f"[skip] {dest} already exists.")
            continue
        print(f"[downloading] {url} -> {dest}")
        # Download to a temp name first so an interrupted download isn't mistaken for a finished file.
        partial = dest.with_name(dest.name + ".part")
        try:
            urllib.request.urlretrieve(url, partial)
            partial.replace(dest)
        finally:
            partial.unlink(missing_ok=True)
        print(f"[done] {dest} ({dest.stat().st_size / (1024 * 1024):.1f} MB)")


def generate_all_samples(base_dir: Path = Path("data_lake/bronze")) -> None:
    """Create all four multi-schema sample datasets for instant local testing."""
    generate_uci_sample(base_dir / "sample_uci_online_retail_2011.csv", n_rows=50000, year=2011, retail_ii_schema=False, seed=42)
    generate_uci_sample(base_dir / "sample_uci_online_retail_II_2010.csv", n_rows=65000, year=2010, retail_ii_schema=True, seed=24)
    generate_rees46_sample(base_dir / "sample_rees46_clickstream.csv", n_events=60000, seed=99)
    generate_olist_sample(base_dir / "sample_olist_brazil", n_orders=12000, seed=77)
    print(f"Generated multi-schema sample datasets in {base_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Big Data dataset helper")
    parser.add_argument("--generate-samples", action="store_true", help="Generate realistic multi-schema sample datasets")
    parser.add_argument("--download-uci", action="store_true", help="Download real UCI Online Retail I & II datasets")
    args = parser.parse_args()
    if args.generate_samples:
        generate_all_samples()
    if args.download_uci:
        download_uci_datasets()
    if not args.generate_samples and not args.download_uci:
        parser.print_help()
