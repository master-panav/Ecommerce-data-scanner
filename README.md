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
- Python 3.10+
- Java 17 (required by Apache Spark)

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

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

### 4. Launch the Workbench
```bash
streamlit run app.py
```

## Architecture Overview
- `engine/spark_session.py` — Local Spark cluster configuration, Adaptive Query Execution (AQE), Snappy Parquet compression, and live telemetry.
- `engine/schemas.py` — Automatic schema detection and PySpark normalization adapters (including distributed `broadcast` joins for Olist & Instacart).
- `engine/pipeline.py` — Bronze → Silver ETL, full data quality audit, `Year/Month` partitioned Parquet writer, and CSV/Excel vs. Parquet read-speed benchmark.
- `engine/analytics.py` — Gold-layer Spark SQL aggregations (Gross/Returns/Net KPIs, monthly trends, top products, country breakdown, day×hour heatmap, REES46 clickstream funnel, cross-dataset comparison).
- `engine/ml_models.py` — Spark MLlib **RFM + K-Means** (`k=2..6`, Silhouette & Elbow evaluation, cluster profiling, churn-risk lookup) and **FP-Growth** Market Basket association rules.
- `app.py` — 6-page interactive Streamlit + Plotly workbench with CSV export for reports.
