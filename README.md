# E-Commerce Customer Intelligence & Sales Analytics (Big Data Workbench)

**Course:** Big Data Analytics and Business Intelligence — Bennett University (B.Tech)

A PySpark-powered Big Data software application implementing a **Medallion Architecture (Bronze → Silver Parquet → Gold Analytics & Spark MLlib)** with an interactive 7-page workbench UI.

## Supported Public Big Data Datasets

| Dataset | Scale | Format | Source |
| :--- | :--- | :--- | :--- |
| **UCI Online Retail (2010–2011)** | 541,909 rows | 22.6 MB `.xlsx` | [UCI ML Repository](https://archive.ics.uci.edu/dataset/352/online+retail) |
| **UCI Online Retail II (2009–2011)** | 1,067,371 rows | 43.5 MB `.xlsx` | [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/online+retail+II) |
| **Olist Brazilian E-Commerce** | ~100,000+ orders (9 tables) | ~45 MB CSVs | [Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) |
| **DataCo Smart Supply Chain** | ~180,000 rows × 53 columns | ~95 MB `.csv` | [Kaggle](https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis) |
| **Pakistan's Largest E-Commerce** | ~580,000 orders × 26 columns | ~100 MB `.csv` | [Kaggle](https://www.kaggle.com/datasets/zusmani/pakistans-largest-ecommerce-dataset) |
| **SuperStore Sales** | ~50,000 rows × 21 columns | ~8 MB `.csv` | [Kaggle](https://www.kaggle.com/datasets/laibaanwer/superstore-sales-dataset) |
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

> `requirements.txt` pins **PySpark 3.5** (not 4.x) and **pandas < 3**, and installs **setuptools** because PySpark 3.5 still imports
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
- **Option D (Upload in the app):** CSV, CSV.GZ, XLSX, Parquet or the **Kaggle `.zip` as downloaded**, up to 2 GB
  (set in `.streamlit/config.toml`). ZIPs are extracted into `data_lake/bronze/<name>/`; multi-table Olist is detected
  as one source, other ZIPs list each data file separately.
  Any retail, supply-chain or clickstream file works as long as it has a **date column** and a **price or amount
  column**, however many columns it has. Columns are matched by name regardless of case, spaces or punctuation
  (`Invoice No`, `order date (DateOrders)`, `Order Item Quantity`, `qty_ordered`, `Sub-Category`…); if no date column is
  recognised by name, the column whose values parse as dates is used. Order status (`canceled`, `refunded`…),
  profit and discount columns are picked up automatically, and **every other column is kept** in Silver as an `x_*`
  attribute for the Slice & Dice Explorer. The Data Quality page shows the mapping, kept columns and a per-column profile.

### How uploaded data is cleaned before analysis
1. **Bronze pre-clean:** BOM / blank / duplicate headers fixed, text trimmed, placeholders (`""`, `NaN`, `null`, `N/A`, `-`)
   turned into NULL, fully empty rows and columns removed.
2. **Cast-safe normalisation:** numbers with currency symbols, thousands separators or decimal commas are parsed;
   malformed numbers and dates become NULL instead of crashing (even with Spark's strict ANSI mode on).
3. **Silver rules:** exact duplicates removed; rows with no date, no positive price or a missing/zero quantity dropped;
   cancellations (`C` invoice prefix or a cancelled/refunded/returned status), returns and missing customer IDs flagged.
   Revenue uses the source's own line total when there is one (so discounts are respected). Every count is shown on
   the Data Quality page.

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
| `CAST_INVALID_INPUT` (e.g. `'apple'` to `BIGINT`) / Spark 4.x in telemetry | PySpark 4 is installed. Run `pip install "pyspark>=3.5.3,<4"` |
| "Could not recognise the columns" | The file has no date column or no price/amount column — rename the header |
| "No valid rows left after cleaning" | The message lists how many rows lacked a date, price or quantity, and the column mapping used |

## Architecture Overview
- `engine/io_utils.py` — Cross-platform Parquet/CSV I/O (Spark native writer, or PyArrow fallback on Windows without winutils).
- `engine/spark_session.py` — Local Spark cluster configuration, Adaptive Query Execution (AQE), Snappy Parquet compression, and live telemetry.
- `engine/schemas.py` — Automatic schema detection and PySpark normalization adapters (including distributed `broadcast` joins for Olist & Instacart).
- `engine/pipeline.py` — Bronze → Silver ETL, full data quality audit, `Year/Month` partitioned Parquet writer, and CSV/Excel vs. Parquet read-speed benchmark.
- `engine/analytics.py` — Gold-layer Spark SQL aggregations (Gross/Returns/Net KPIs, monthly trends, top products, country breakdown, day×hour heatmap, REES46 clickstream funnel, cross-dataset comparison).
- `engine/ml_models.py` — Spark MLlib **RFM + K-Means** (`k=2..6`, Silhouette & Elbow evaluation, cluster profiling, churn-risk lookup) and **FP-Growth** Market Basket association rules.
- `app.py` — 7-page interactive Streamlit + Plotly workbench with CSV export for reports. Page 7 is an OLAP-style
  Slice & Dice Explorer: any measure (revenue, profit, shipping days…) by any dimension, with an optional pivot.
- `tests/` — pytest suite covering schema detection, Excel/CSV ingestion, the full ETL, Gold analytics and MLlib models.
