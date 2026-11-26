# E-Commerce Customer Intelligence & Sales Analytics (Big Data Workbench)

**Course:** Big Data Analytics and Business Intelligence — Bennett University (B.Tech)

A PySpark-powered Big Data software application implementing a **Medallion Architecture (Bronze → Silver Parquet → Gold Analytics & Spark MLlib)** with an interactive 6-page workbench UI.

## Supported Public Big Data Datasets

| Dataset | Scale | Format | Source |
| :--- | :--- | :--- | :--- |
| **UCI Online Retail (2010–2011)** | 541,909 rows | 22.6 MB `.xlsx` | [UCI ML Repository](https://archive.ics.uci.edu/dataset/352/online+retail) |
| **UCI Online Retail II (2009–2011)** | 1,067,371 rows | 43.5 MB `.xlsx` | [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/online+retail+II) |
| **Olist Brazilian E-Commerce** | ~100,000+ orders (9 tables) | ~45 MB CSVs | [Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) |
| **REES46 Multi-Category Clickstream** | 285M events (7 months) | 1.6–2.7 GB/mo `.csv.gz` | [Kaggle](https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store) |
| **Instacart Market Basket** | 3M+ orders / 30M+ items | ~700 MB CSVs | [Kaggle](https://www.kaggle.com/datasets/yasserh/instacart-online-grocery-basket-analysis-dataset) |

## Quick Start (Local Machine)

### 1. Prerequisites
- Python 3.10 – 3.12
- Java 8, 11 or 17 (required by Apache Spark 3.5) — check with `java -version`

### 2. Install Dependencies
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

> `requirements.txt` pins **PySpark 3.5** and **pandas < 3**, and installs **setuptools** because PySpark 3.5 still imports
> `distutils`, which Python 3.12 removed. Installing unpinned versions is what caused the
> `ModuleNotFoundError: No module named 'distutils'` crash when a dataset was uploaded.

### 3. Add Datasets (Option A: Instant Demo, Option B: Real UCI/Kaggle Data)
- **Option A (Instant local test — generates UCI I, UCI II, REES46 & Olist samples):**
  ```bash
  python data_tools.py --generate-samples
  ```
- **Option B (Download real UCI Online Retail I & II automatically):**
  ```bash
  python data_tools.py --download-uci
  ```
- **Option C (Kaggle Big Data sets):** Place any `.csv` / `.csv.gz` (REES46) or multi-table folder (Olist / Instacart) directly inside `data_lake/bronze/`.
- **Option D (Upload in the app):** CSV, CSV.GZ, XLSX or Parquet up to 2 GB (set in `.streamlit/config.toml`).
  Any retail file works as long as it has an invoice/order id, a date and a price column — common names such as
  `InvoiceNo` / `order_id`, `InvoiceDate` / `order_date`, `UnitPrice` / `price` are recognised automatically.

### 4. Launch the Workbench
Run from the project folder so `.streamlit/config.toml` is picked up:
```bash
streamlit run app.py
```
Processed Silver datasets are remembered (`data_lake/silver/*/_audit.json`), so refreshing the browser does not lose them.

### 5. Run the Tests
```bash
pip install -r requirements-dev.txt
pytest -q
```

## Windows Notes
Spark normally needs Hadoop's `winutils.exe` + `hadoop.dll` on Windows. **You do not need them here**: without
`HADOOP_HOME`, the workbench writes Parquet with PyArrow and reads it back through an explicit file list, which avoids
Hadoop's native calls (the telemetry panel shows `Parquet Writer: PyArrow fallback`). If you do set `HADOOP_HOME`
with both files in `bin/`, Spark's native distributed writer is used instead.

## Troubleshooting
| Error | Fix |
| :--- | :--- |
| `No module named 'distutils'` | `pip install -r requirements.txt` (adds setuptools, pins PySpark 3.5 / pandas < 3) |
| `UnsatisfiedLinkError: NativeIO$Windows.access0` | Fixed in this version — update the code; no winutils needed |
| `UnsupportedClassVersionError` / Spark won't start | You have PySpark 4.x with Java < 17. Reinstall from `requirements.txt` |
| "Could not recognise the columns" | The file lacks an invoice/order id, date or price column — rename the header |
| "No valid rows left after cleaning" | Dates could not be parsed or all prices were ≤ 0 |

## Architecture Overview
- `engine/io_utils.py` — Cross-platform Parquet/CSV I/O (Spark native writer, or PyArrow fallback on Windows without winutils).
- `engine/spark_session.py` — Local Spark cluster configuration, Adaptive Query Execution (AQE), Snappy Parquet compression, and live telemetry.
- `engine/schemas.py` — Automatic schema detection and PySpark normalization adapters (including distributed `broadcast` joins for Olist & Instacart).
- `engine/pipeline.py` — Bronze → Silver ETL, full data quality audit, `Year/Month` partitioned Parquet writer, and CSV/Excel vs. Parquet read-speed benchmark.
- `engine/analytics.py` — Gold-layer Spark SQL aggregations (Gross/Returns/Net KPIs, monthly trends, top products, country breakdown, day×hour heatmap, REES46 clickstream funnel, cross-dataset comparison).
- `engine/ml_models.py` — Spark MLlib **RFM + K-Means** (`k=2..6`, Silhouette & Elbow evaluation, cluster profiling, churn-risk lookup) and **FP-Growth** Market Basket association rules.
- `app.py` — 6-page interactive Streamlit + Plotly workbench with CSV export for reports.
- `tests/` — pytest suite covering schema detection, Excel/CSV ingestion, the full ETL, Gold analytics and MLlib models.
