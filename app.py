"""E-Commerce Customer Intelligence & Sales Analytics — PySpark Big Data Workbench.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import re
import shutil
import traceback
import zipfile
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_tools import generate_all_samples
from engine.analytics import (
    clickstream_funnel,
    compare_datasets,
    compute_kpis,
    country_summary,
    explore,
    explorer_fields,
    filter_silver,
    monthly_trends,
    pretty,
    shopping_heatmap,
    top_products,
)
from engine.forecasting import MEASURES, cohort_analysis, forecast_monthly
from engine.ml_models import run_fpgrowth_basket, run_kmeans_segmentation
from engine.pipeline import EmptySilverError, PipelineAudit, load_saved_audits, load_silver, run_medallion_etl
from engine.schemas import OLIST_FILE_RE, DATASET_CATALOG, SchemaMappingError, best_sales_file, is_multi_table_folder
from engine.spark_session import get_spark, get_spark_telemetry, set_shuffle_partitions

APP_DIR = Path(__file__).resolve().parent
LAKE_ROOT = APP_DIR / "data_lake"
BRONZE_DIR = LAKE_ROOT / "bronze"
SUPPORTED_UPLOAD_TYPES = ["csv", "gz", "xlsx", "parquet", "zip"]
DATA_SUFFIXES = {".csv", ".xlsx", ".parquet", ".gz"}

st.set_page_config(
    page_title="Big Data E-Commerce Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Subtle professional styling
st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    div[data-testid="stMetric"] {
        background: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 10px;
        padding: 12px 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Starting local Spark cluster...")
def init_spark():
    return get_spark()


def _visible(item: Path) -> bool:
    return not item.name.startswith((".", "~$", "_")) and not item.name.endswith(".part")  # hidden, Excel lock, partial


def discover_bronze_sources() -> dict[str, Path]:
    """List data files, multi-table folders (Olist/Instacart) and Parquet folders in data_lake/bronze.

    Other folders (e.g. an extracted Kaggle ZIP) are opened and each data file inside is listed separately.
    """
    BRONZE_DIR.mkdir(parents=True, exist_ok=True)
    sources: dict[str, Path] = {}
    for item in sorted(filter(_visible, BRONZE_DIR.iterdir()), key=lambda x: x.name.lower()):
        if item.is_file() and item.suffix.lower() in DATA_SUFFIXES:
            sources[item.name] = item
        elif item.is_dir() and (is_multi_table_folder(item) or any(item.glob("*.parquet"))):
            sources[item.name] = item
        elif item.is_dir():
            for f in sorted(filter(_visible, item.rglob("*")), key=lambda x: str(x).lower()):
                if f.is_file() and f.suffix.lower() in DATA_SUFFIXES:
                    sources[f.relative_to(BRONZE_DIR).as_posix()] = f
    return sources


def extract_zip(zip_path: Path) -> str:
    """Extract a Kaggle-style ZIP into a Bronze folder (guarding against path traversal)."""
    target = BRONZE_DIR / zip_path.stem
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            dest = (target / member.filename).resolve()
            if target.resolve() not in dest.parents and dest != target.resolve():
                raise ValueError(f"Unsafe path in ZIP: {member.filename}")
        zf.extractall(target)
    zip_path.unlink()
    # A ZIP that wraps everything in one top-level folder: flatten it
    inner = [p for p in target.iterdir() if _visible(p)]
    if len(inner) == 1 and inner[0].is_dir():
        for child in inner[0].iterdir():
            child.rename(target / child.name)
        inner[0].rmdir()
    return target.name


def safe_upload_name(name: str) -> str:
    """Strip any path components and unsafe characters from an uploaded file name."""
    base = Path(name.replace("\\", "/")).name
    cleaned = re.sub(r"[^A-Za-z0-9._ -]+", "_", base).strip(" .")
    return cleaned or "uploaded_dataset.csv"


def save_upload(uploaded) -> str:
    """Persist an uploaded file to Bronze exactly once (Streamlit reruns the script on every click)."""
    upload_id = getattr(uploaded, "file_id", None) or f"{uploaded.name}:{uploaded.size}"
    name = safe_upload_name(uploaded.name)
    saved = st.session_state.setdefault("saved_upload_ids", set())
    if upload_id not in saved:
        # Olist ships as 9 related CSVs: collect them in one folder so they are joined as one source
        olist_dir = BRONZE_DIR / "olist"
        dest = (olist_dir if OLIST_FILE_RE.match(name) else BRONZE_DIR) / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        with open(tmp, "wb") as fh:
            fh.write(uploaded.getbuffer())
        tmp.replace(dest)
        saved.add(upload_id)
        if dest.parent == olist_dir:
            name = "olist" if is_multi_table_folder(olist_dir) else f"olist/{name}"
            st.toast(f"Saved `{dest.name}` to `data_lake/bronze/olist/`.")
        elif dest.suffix.lower() == ".zip":
            try:
                folder = extract_zip(dest)
            except (zipfile.BadZipFile, ValueError) as exc:
                dest.unlink(missing_ok=True)
                st.error(f"Could not extract `{name}`: {exc}")
                return name
            sources = discover_bronze_sources()
            name = folder if folder in sources else next((k for k in sources if k.startswith(folder + "/")), folder)
            st.toast(f"Extracted `{dest.name}` into `data_lake/bronze/{folder}/`.")
        else:
            st.toast(f"Saved `{name}` to the Bronze layer.")
        st.session_state["bronze_choice"] = name
    return name


def run_etl_with_feedback(spark, source: Path, name: str) -> PipelineAudit | None:
    """Run ETL and turn failures into readable messages instead of a crashed page."""
    try:
        with st.spinner(f"Running Spark ETL on {name}..."):
            return run_medallion_etl(spark, source, name, LAKE_ROOT)
    except (SchemaMappingError, EmptySilverError) as exc:
        st.error(str(exc))
    except Exception as exc:  # noqa: BLE001 - surface any Spark/IO failure to the user
        first_line = (str(exc).splitlines() or [""])[0][:400]
        st.error(f"ETL failed for `{name}`: {type(exc).__name__}: {first_line}")
        with st.expander("Technical details"):
            st.code(traceback.format_exc())
    return None


# Session state for processed Silver datasets (restored from disk after a refresh)
if "audits" not in st.session_state:
    st.session_state["audits"] = load_saved_audits(LAKE_ROOT)  # dataset_name -> PipelineAudit
if "active_dataset" not in st.session_state:
    st.session_state["active_dataset"] = next(iter(st.session_state["audits"]), None)


# ---------------- Sidebar ----------------
with st.sidebar:
    st.title("⚡ Big Data Workbench")
    st.caption("Apache Spark · Medallion Lake · Spark MLlib")

    page = st.radio(
        "Workspace",
        [
            "1. Pipeline & Spark Monitor",
            "2. Data Quality Audit",
            "3. Sales & Market BI",
            "4. RFM & K-Means Clustering",
            "5. Market Basket (FP-Growth)",
            "6. Cross-Dataset Comparison",
            "7. Slice & Dice Explorer",
            "8. Forecasting & Cohorts",
        ],
    )

    st.divider()
    shuffle_parts = st.slider("Spark Shuffle Partitions", min_value=2, max_value=32, value=8, step=2)
    try:
        spark = init_spark()
    except Exception as exc:  # noqa: BLE001
        first_line = (str(exc).splitlines() or [""])[0][:300]
        st.error(
            "Could not start Spark. Make sure Java 8, 11 or 17 is installed and `java -version` works."
            f"\n\n`{type(exc).__name__}: {first_line}`"
        )
        st.stop()
    set_shuffle_partitions(spark, shuffle_parts)

    if st.session_state["audits"]:
        names = list(st.session_state["audits"].keys())
        default_idx = names.index(st.session_state["active_dataset"]) if st.session_state["active_dataset"] in names else 0
        st.session_state["active_dataset"] = st.selectbox("Active Silver Dataset", names, index=default_idx)
    else:
        st.info("Run the Medallion ETL on Page 1 to load a Silver dataset.")


def require_active_dataset() -> tuple[str, PipelineAudit]:
    active = st.session_state.get("active_dataset")
    if not active or active not in st.session_state["audits"]:
        st.warning("No Silver dataset is loaded yet. Go to **1. Pipeline & Spark Monitor** and run ETL on a dataset first.")
        st.stop()
    return active, st.session_state["audits"][active]


# ---------------- Page 1: Pipeline & Spark Monitor ----------------
if page == "1. Pipeline & Spark Monitor":
    st.header("Medallion ETL Pipeline & Spark Monitor")
    st.caption("Ingest raw Bronze datasets, clean & partition into Silver Parquet, and benchmark distributed Spark I/O.")

    col_left, col_right = st.columns([1.3, 1])

    with col_left:
        st.subheader("Ingest Dataset (Bronze → Silver)")
        sources = discover_bronze_sources()

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Generate Multi-Schema Samples", width="stretch"):
                with st.spinner("Generating UCI I, UCI II, REES46 & Olist sample datasets..."):
                    generate_all_samples(BRONZE_DIR)
                st.success("Generated 4 sample datasets in `data_lake/bronze/`.")
                st.rerun()
        with c2:
            uploads = st.file_uploader(
                "Upload CSV / CSV.GZ / XLSX / Parquet / ZIP",
                type=SUPPORTED_UPLOAD_TYPES,
                accept_multiple_files=True,
                label_visibility="collapsed",
                help="Saved into data_lake/bronze/. Kaggle ZIPs are extracted; Olist CSVs can be uploaded together "
                "and are joined automatically. For multi-GB files, copy them there directly.",
            )
            already_saved = set(st.session_state.get("saved_upload_ids", set()))
            new_names = []
            for uploaded in uploads or []:
                name = save_upload(uploaded)
                if (getattr(uploaded, "file_id", None) or f"{uploaded.name}:{uploaded.size}") not in already_saved:
                    new_names.append(name)
            if uploads:
                sources = discover_bronze_sources()
                # Several plain files added at once (e.g. a Kaggle download's CSVs): select the main
                # sales table rather than whichever file happened to be last
                batch = [sources[n] for n in dict.fromkeys(new_names) if n in sources and sources[n].is_file()]
                if len(batch) > 1 and st.session_state.get("bronze_choice") in new_names:
                    best = best_sales_file(batch)
                    st.session_state["bronze_choice"] = next(n for n in new_names if sources.get(n) == best)

        if sources:
            names = list(sources.keys())
            if st.session_state.get("bronze_choice") not in names:
                st.session_state["bronze_choice"] = names[0]
            selected_name = st.selectbox("Select Bronze Source", names, key="bronze_choice")
            if st.button("Run PySpark Medallion ETL", type="primary", width="stretch"):
                audit = run_etl_with_feedback(spark, sources[selected_name], selected_name)
                if audit is not None:
                    st.session_state["audits"][selected_name] = audit
                    st.session_state["active_dataset"] = selected_name
                    # Drop model results computed on an older version of this dataset
                    stale = [k for k in st.session_state if str(k).startswith((f"kmeans_{selected_name}_", f"fp_{selected_name}"))]
                    for key in stale:
                        del st.session_state[key]
                    st.session_state.pop("comp_df", None)
                    st.session_state["etl_message"] = (
                        f"Silver Parquet ready for `{selected_name}` in {audit.etl_total_sec}s ({audit.silver_valid_rows:,} rows)."
                    )
                    st.rerun()  # refresh the sidebar's dataset selector
            if msg := st.session_state.pop("etl_message", None):
                st.success(msg)
        else:
            st.info("No files in `data_lake/bronze/` yet. Click **Generate Multi-Schema Samples** or drop your UCI/Kaggle files there.")

    with col_right:
        st.subheader("Live Spark Engine Telemetry")
        telem = get_spark_telemetry(spark)
        st.dataframe(pd.DataFrame(telem.items(), columns=["Spark Parameter", "Value"]), width="stretch", hide_index=True)

    # Benchmark summary if datasets have been processed
    if st.session_state["audits"]:
        st.subheader("Parquet Storage & Read-Speed Benchmarks")
        bench_rows = []
        for a in st.session_state["audits"].values():
            bench_rows.append({
                "Dataset": a.dataset_name,
                "Detected Schema": a.schema_type,
                "Raw Rows": f"{a.raw_rows:,}",
                "Silver Rows": f"{a.silver_valid_rows:,}",
                "Source (MB)": a.source_size_mb,
                "Parquet (MB)": a.parquet_size_mb,
                "Compression (%)": a.compression_ratio_pct,
                "Raw Read (s)": a.raw_read_sec,
                "Parquet Read (s)": a.parquet_read_sec,
                "Partitions": a.silver_partitions,
            })
        st.dataframe(pd.DataFrame(bench_rows), width="stretch", hide_index=True)

    with st.expander("Supported Public Big Data Catalog (UCI & Kaggle)"):
        cat_df = pd.DataFrame([
            {"Dataset": v["name"], "Scale": v["rows"], "File Size": v["size"], "Big Data Focus": v["focus"], "Source URL": v["url"]}
            for v in DATASET_CATALOG.values()
        ])
        st.dataframe(cat_df, width="stretch", hide_index=True)


# ---------------- Page 2: Data Quality Audit ----------------
elif page == "2. Data Quality Audit":
    name, audit = require_active_dataset()
    st.header(f"Data Quality Audit — {name}")
    st.caption(f"Detected Schema: `{audit.schema_type}` · Silver Parquet Path: `{audit.silver_path}`")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Raw Rows Read", f"{audit.raw_rows:,}")
    k2.metric("Silver Valid Rows", f"{audit.silver_valid_rows:,}")
    k3.metric("Duplicates Removed", f"{audit.duplicate_rows:,}")
    k4.metric("Parquet Compression", f"{audit.compression_ratio_pct}%")

    empty_cols = audit.empty_columns_removed
    audit_table = pd.DataFrame([
        {"Check": "Fully Empty Rows", "Row Count": audit.empty_rows_removed, "Handling": "Removed before processing (Bronze pre-clean)"},
        {"Check": "Fully Empty Columns", "Row Count": len(empty_cols), "Handling": "Dropped: " + (", ".join(empty_cols) if empty_cols else "none")},
        {"Check": "Cancelled / Refunded Orders", "Row Count": audit.cancellation_rows,
         "Handling": f"Flagged (`is_cancellation`): 'C' invoice prefix or order status ({audit.status_cancellation_rows:,} by status)"},
        {"Check": "Negative Quantity / Returns", "Row Count": audit.return_rows, "Handling": "Flagged (`is_return`), excluded from completed sales"},
        {"Check": "Missing CustomerID", "Row Count": audit.missing_customer_rows, "Handling": "Kept for aggregate BI; excluded from customer RFM"},
        {"Check": "Invalid / Non-Positive Unit Price", "Row Count": audit.invalid_price_rows, "Handling": "Dropped in Silver layer"},
        {"Check": "Null / Unparseable Timestamp", "Row Count": audit.null_date_rows, "Handling": "Dropped in Silver layer"},
        {"Check": "Missing / Malformed / Zero Quantity", "Row Count": audit.malformed_quantity_rows, "Handling": "Dropped in Silver layer"},
        {"Check": "Exact Duplicate Rows", "Row Count": audit.duplicate_rows, "Handling": "Deduplicated via Spark `dropDuplicates()`"},
    ])
    st.caption("Placeholder values such as blank, `NaN`, `null`, `N/A` are converted to NULL, and malformed numbers or "
               "dates become NULL instead of failing the job, so they are counted in the checks below.")

    col_a, col_b = st.columns([1.2, 1])
    with col_a:
        st.subheader("Cleaning Audit Log")
        st.dataframe(audit_table, width="stretch", hide_index=True)
    with col_b:
        st.subheader("Data Quality Issue Distribution")
        fig = px.bar(audit_table, x="Row Count", y="Check", orientation="h", text_auto=True)
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="")
        st.plotly_chart(fig, width="stretch")

    if audit.column_mapping:
        map_col, extra_col = st.columns(2)
        with map_col:
            st.subheader("Detected Column Mapping")
            st.dataframe(
                pd.DataFrame(list(audit.column_mapping.items()), columns=["Canonical Field", "Source Column"]),
                width="stretch", hide_index=True,
            )
        with extra_col:
            st.subheader(f"Extra Columns Kept ({len(audit.extra_columns)})")
            st.caption("Unmapped columns are stored in Silver as `x_*` attributes for the Slice & Dice Explorer.")
            st.dataframe(
                pd.DataFrame(list(audit.extra_columns.items()), columns=["Silver Column", "Source Column"]),
                width="stretch", hide_index=True, height=260,
            )

    if audit.column_profile:
        st.subheader(f"Bronze Column Profile ({len(audit.column_profile)} columns)")
        profile_df = pd.DataFrame(audit.column_profile).rename(columns={
            "column": "Column", "type": "Spark Type", "non_null": "Non-Null", "null_pct": "Null %",
            "distinct_approx": "Distinct (approx)"})
        st.dataframe(
            profile_df, width="stretch", hide_index=True,
            column_config={"Null %": st.column_config.ProgressColumn("Null %", min_value=0, max_value=100, format="%.1f%%")},
        )

    st.subheader("Silver Parquet Sample Preview (First 20 Rows)")
    silver_df = load_silver(spark, audit.silver_path)
    st.dataframe(silver_df.limit(20).toPandas(), width="stretch", hide_index=True)


# ---------------- Page 3: Sales & Market BI ----------------
elif page == "3. Sales & Market BI":
    name, audit = require_active_dataset()
    st.header(f"Gold-Layer Sales & Market BI — {name}")

    silver_df = load_silver(spark, audit.silver_path)
    all_countries = [r["country"] for r in silver_df.select("country").distinct().orderBy("country").collect()]

    selected_countries = st.multiselect("Filter Countries (leave empty for all)", all_countries)
    filtered_df = filter_silver(silver_df, countries=selected_countries or None)

    kpis = compute_kpis(filtered_df)
    cards = st.columns(6 if kpis["profit"] is not None else 5)
    cards[0].metric("Net Revenue", f"{kpis['net_revenue']:,.2f}")
    cards[1].metric("Gross Sales", f"{kpis['gross_sales']:,.2f}")
    cards[2].metric("Completed Orders", f"{kpis['orders']:,}")
    cards[3].metric("Average Order Value", f"{kpis['aov']:,.2f}")
    cards[4].metric("Return / Cancel Rate", f"{kpis['return_rate_pct']}%")
    if kpis["profit"] is not None:
        cards[5].metric("Profit", f"{kpis['profit']:,.2f}", f"{kpis['profit_margin_pct']}% margin")

    # Clickstream funnel if REES46
    if audit.events_path:
        st.subheader("Clickstream Conversion Funnel (View → Cart → Purchase)")
        funnel_pdf = clickstream_funnel(spark, audit.events_path)
        fc1, fc2 = st.columns([1.3, 1])
        with fc1:
            fig_f = px.funnel(funnel_pdf, x="events", y="event_type", text="unique_users")
            fig_f.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_f, width="stretch")
        with fc2:
            st.dataframe(funnel_pdf, width="stretch", hide_index=True)

    # Monthly trend
    trend_pdf = monthly_trends(filtered_df)
    st.subheader("Monthly Gross Sales, Returns & Net Revenue")
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(x=trend_pdf["year_month"], y=trend_pdf["gross_sales"], mode="lines+markers", name="Gross Sales"))
    fig_trend.add_trace(go.Scatter(x=trend_pdf["year_month"], y=trend_pdf["net_revenue"], mode="lines+markers", name="Net Revenue"))
    fig_trend.add_trace(go.Bar(x=trend_pdf["year_month"], y=trend_pdf["returns"], name="Returns", opacity=0.45))
    fig_trend.update_layout(height=360, barmode="overlay", xaxis_title="Month", yaxis_title="Value")
    st.plotly_chart(fig_trend, width="stretch")

    c_prod, c_geo = st.columns(2)
    with c_prod:
        st.subheader("Top 12 Products by Revenue")
        prod_pdf = top_products(filtered_df, limit=12)
        fig_prod = px.bar(prod_pdf.sort_values("revenue"), x="revenue", y="description", orientation="h", hover_data=["units_sold", "orders"])
        fig_prod.update_layout(height=400, yaxis_title="", xaxis_title="Revenue")
        st.plotly_chart(fig_prod, width="stretch")

    with c_geo:
        st.subheader("Revenue by Country / Region")
        geo_pdf = country_summary(filtered_df)
        fig_geo = px.bar(geo_pdf.head(12).sort_values("revenue"), x="revenue", y="country", orientation="h", hover_data=["orders", "aov"])
        fig_geo.update_layout(height=400, yaxis_title="", xaxis_title="Revenue")
        st.plotly_chart(fig_geo, width="stretch")

    st.subheader("Peak Shopping Activity (Day of Week × Hour)")
    heat_pdf = shopping_heatmap(filtered_df)
    if not heat_pdf.empty:
        day_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        pivot = heat_pdf.pivot(index="day_of_week", columns="hour", values="orders").fillna(0)
        pivot = pivot.reindex([d for d in day_order if d in pivot.index])
        fig_h = px.imshow(pivot, aspect="auto", color_continuous_scale="Blues", labels=dict(x="Hour of Day", y="Day", color="Orders"))
        fig_h.update_layout(height=320)
        st.plotly_chart(fig_h, width="stretch")

    st.download_button("Download Monthly BI Summary (CSV)", trend_pdf.to_csv(index=False), f"{name}_monthly_bi.csv", "text/csv")


# ---------------- Page 4: RFM & K-Means Clustering ----------------
elif page == "4. RFM & K-Means Clustering":
    name, audit = require_active_dataset()
    st.header(f"Customer Intelligence: RFM + Spark MLlib K-Means — {name}")

    k_val = st.slider("Number of Customer Segments (k)", min_value=2, max_value=6, value=4)
    if st.button("Train Spark MLlib K-Means Model", type="primary"):
        with st.spinner("Computing RFM features & evaluating K-Means (k=2..6) in Spark..."):
            silver_df = load_silver(spark, audit.silver_path)
            res = run_kmeans_segmentation(silver_df, k_selected=k_val)
            st.session_state[f"kmeans_{name}_{k_val}"] = res

    res = st.session_state.get(f"kmeans_{name}_{k_val}")
    if not res:
        st.info("Choose `k` and click **Train Spark MLlib K-Means Model** to segment customers.")
    elif "error" in res:
        st.error(res["error"])
    else:
        st.caption(f"Reference Date: `{res['reference_date']}` · Clustered Customers: `{res['customer_count']:,}`")

        eval_df = res["eval_df"]
        prof_df = res["profile_df"]
        cust_df = res["customers_df"]

        ec1, ec2 = st.columns(2)
        with ec1:
            st.subheader("Silhouette Score by k (Higher = Better Separation)")
            fig_sil = px.line(eval_df, x="k", y="silhouette", markers=True)
            fig_sil.update_layout(height=300, xaxis=dict(dtick=1))
            st.plotly_chart(fig_sil, width="stretch")
        with ec2:
            st.subheader("Elbow Curve (WSSSE Training Cost)")
            fig_elb = px.line(eval_df, x="k", y="wssse", markers=True)
            fig_elb.update_layout(height=300, xaxis=dict(dtick=1))
            st.plotly_chart(fig_elb, width="stretch")

        st.subheader("Cluster Profiles (Observed Means)")
        st.dataframe(prof_df[["cluster", "segment", "customers", "share_pct", "avg_recency", "avg_frequency", "avg_monetary", "total_revenue"]], width="stretch", hide_index=True)

        st.subheader("3D RFM Cluster Visualization")
        sample_plot = cust_df.sample(min(len(cust_df), 4000), random_state=42)
        fig_3d = px.scatter_3d(
            sample_plot,
            x="recency",
            y="frequency",
            z="monetary",
            color="segment",
            opacity=0.7,
            hover_data=["customer_id", "churn_risk"],
        )
        fig_3d.update_layout(height=480, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_3d, width="stretch")

        st.subheader("Individual Customer Lookup & Churn Risk")
        search_id = st.text_input("Enter Customer ID (partial or exact)")
        if search_id:
            matches = cust_df[cust_df["customer_id"].astype(str).str.contains(search_id.strip(), case=False, na=False, regex=False)].head(25)
            st.dataframe(matches, width="stretch", hide_index=True)

        st.download_button("Download Customer Segments (CSV)", cust_df.to_csv(index=False), f"{name}_rfm_segments.csv", "text/csv")


# ---------------- Page 5: Market Basket (FP-Growth) ----------------
elif page == "5. Market Basket (FP-Growth)":
    name, audit = require_active_dataset()
    st.header(f"Market Basket Analysis: Spark MLlib FP-Growth — {name}")

    fc1, fc2 = st.columns(2)
    min_sup = fc1.slider("Minimum Support", min_value=0.005, max_value=0.10, value=0.015, step=0.005, format="%.3f")
    min_conf = fc2.slider("Minimum Confidence", min_value=0.10, max_value=0.80, value=0.25, step=0.05)

    if st.button("Run Spark FP-Growth Mining", type="primary"):
        with st.spinner("Mining frequent itemsets & association rules in Spark..."):
            silver_df = load_silver(spark, audit.silver_path)
            res = run_fpgrowth_basket(silver_df, min_support=min_sup, min_confidence=min_conf)
            res["params"] = (min_sup, min_conf)
            st.session_state[f"fp_{name}"] = res

    fp_res = st.session_state.get(f"fp_{name}")
    if fp_res and fp_res.get("params") != (min_sup, min_conf):
        run_sup, run_conf = fp_res.get("params", (None, None))
        st.info(
            f"Showing results for support={run_sup}, confidence={run_conf}. "
            "Click **Run Spark FP-Growth Mining** to apply the new thresholds."
        )
    if not fp_res:
        st.info("Adjust Support/Confidence and click **Run Spark FP-Growth Mining**.")
    elif "error" in fp_res:
        st.error(fp_res["error"])
    else:
        rules_df = fp_res["rules_df"]
        st.caption(f"Analyzed `{fp_res['basket_count']:,}` multi-item invoices · Discovered `{len(rules_df)}` top rules.")
        if rules_df.empty:
            st.warning("No rules found at these thresholds. Lower Minimum Support or Confidence and try again.")
        else:
            item_filter = st.text_input("Filter rules by product name")
            if item_filter:
                mask = rules_df["If Customer Buys"].str.contains(item_filter, case=False, na=False, regex=False) | rules_df[
                    "Also Likely To Buy"
                ].str.contains(item_filter, case=False, na=False, regex=False)
                view_df = rules_df[mask]
            else:
                view_df = rules_df
            st.dataframe(view_df, width="stretch", hide_index=True)

            fig_fp = px.scatter(
                view_df,
                x="Support",
                y="Confidence",
                size="Lift",
                color="Lift",
                hover_data=["If Customer Buys", "Also Likely To Buy"],
                color_continuous_scale="Viridis",
            )
            fig_fp.update_layout(height=380)
            st.plotly_chart(fig_fp, width="stretch")
            st.download_button("Download Association Rules (CSV)", view_df.to_csv(index=False), f"{name}_fpgrowth_rules.csv", "text/csv")


# ---------------- Page 6: Cross-Dataset Comparison ----------------
elif page == "6. Cross-Dataset Comparison":
    st.header("Cross-Dataset Comparison (Reproducibility Check)")
    audits = st.session_state["audits"]
    if len(audits) < 2:
        st.info("Process at least **2 datasets** on Page 1 (e.g., Online Retail I & Online Retail II) to compare them side by side.")
    else:
        names = list(audits.keys())
        c1, c2 = st.columns(2)
        ds_a = c1.selectbox("Dataset A", names, index=0)
        ds_b = c2.selectbox("Dataset B", names, index=1)

        if ds_a == ds_b:
            st.warning("Pick two different datasets to compare.")
        elif st.button("Compare Datasets in Spark", type="primary"):
            with st.spinner("Running side-by-side Gold-layer comparison..."):
                df_a = load_silver(spark, audits[ds_a].silver_path)
                df_b = load_silver(spark, audits[ds_b].silver_path)
                st.session_state["comp_df"] = compare_datasets(df_a, ds_a, df_b, ds_b)
                st.session_state["comp_pair"] = (ds_a, ds_b)

        # Only show a result that matches the currently selected pair
        if "comp_df" in st.session_state and st.session_state.get("comp_pair") == (ds_a, ds_b):
            comp_df = st.session_state["comp_df"]
            st.dataframe(comp_df, width="stretch", hide_index=True)
            st.download_button("Download Comparison Table (CSV)", comp_df.to_csv(index=False), "cross_dataset_comparison.csv", "text/csv")


# ---------------- Page 7: Slice & Dice Explorer ----------------
elif page == "7. Slice & Dice Explorer":
    name, audit = require_active_dataset()
    st.header(f"Slice & Dice Explorer (OLAP) — {name}")
    st.caption("Group any measure by any dimension of the Silver table, including every extra column kept from the "
               "source file. All aggregation runs in Spark; only the summary is collected.")

    silver_df = load_silver(spark, audit.silver_path)
    cache_key = f"explorer_fields_{name}_{audit.etl_total_sec}"
    if cache_key not in st.session_state:
        with st.spinner("Profiling dimensions in Spark..."):
            st.session_state[cache_key] = explorer_fields(silver_df)
    dims, numeric = st.session_state[cache_key]

    base_measures = ["Revenue", "Orders", "Units Sold", "Customers", "Order Lines"]
    c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.2, 0.8])
    dimension = c1.selectbox("Rows (dimension)", dims, format_func=pretty)
    column_dim = c2.selectbox("Columns (optional pivot)", ["(none)"] + [d for d in dims if d != dimension], format_func=pretty)
    measure = c3.selectbox("Measure", base_measures + numeric, format_func=pretty)
    agg = c4.selectbox("Aggregation", ["Sum", "Average", "Min", "Max"], disabled=measure in base_measures)
    completed_only = st.toggle("Completed sales only (exclude cancellations & returns)", value=True)

    pivot_dim = None if column_dim == "(none)" else column_dim
    result = explore(silver_df, dimension, measure, agg, pivot_dim, completed_only)
    label = measure if measure in base_measures else f"{agg} of {pretty(measure)}"
    if result.empty:
        st.info("No rows for this selection.")
    elif pivot_dim is None:
        fig = px.bar(result, x=dimension, y="value", labels={dimension: pretty(dimension), "value": label})
        fig.update_layout(height=420, xaxis_type="category")
        st.plotly_chart(fig, width="stretch")
        st.dataframe(result.rename(columns={dimension: pretty(dimension), "value": label}), width="stretch", hide_index=True)
    else:
        pivot = result.pivot_table(index=dimension, columns=pivot_dim, values="value", aggfunc="sum").fillna(0)
        fig = px.imshow(pivot, aspect="auto", color_continuous_scale="Blues", text_auto=".3s",
                        labels=dict(x=pretty(pivot_dim), y=pretty(dimension), color=label))
        fig.update_layout(height=520)
        st.plotly_chart(fig, width="stretch")
        st.dataframe(pivot.reset_index(), width="stretch", hide_index=True)
    if not result.empty:
        st.download_button("Download Result (CSV)", result.to_csv(index=False), f"{name}_explorer.csv", "text/csv")


# ---------------- Page 8: Forecasting & Cohorts ----------------
elif page == "8. Forecasting & Cohorts":
    name, audit = require_active_dataset()
    st.header(f"Sales Forecasting & Customer Cohorts — {name}")
    tab_fc, tab_co = st.tabs(["Revenue Forecast (Spark MLlib)", "Cohort Retention & LTV"])

    with tab_fc:
        st.caption("Monthly totals are aggregated in Spark. Candidate models are scored on the most recent months "
                   "(held out from training); the most accurate one is refit on all months and projected forward.")
        f1, f2, f3 = st.columns(3)
        fc_measure = f1.selectbox("Measure", list(MEASURES))
        horizon = f2.slider("Months to forecast", 1, 12, 6)
        holdout = f3.slider("Months held out for testing", 1, 6, 3)
        fc_key = f"forecast_{name}_{fc_measure}_{horizon}_{holdout}"
        if st.button("Run Forecast", type="primary"):
            with st.spinner("Aggregating monthly series and training Spark MLlib models..."):
                st.session_state[fc_key] = forecast_monthly(load_silver(spark, audit.silver_path), horizon, holdout, fc_measure)
        fc = st.session_state.get(fc_key)
        if not fc:
            st.info("Choose the settings and click **Run Forecast**.")
        elif "error" in fc:
            st.warning(fc["error"])
        else:
            hist, fut, metrics = fc["history"], fc["forecast"], fc["metrics"]
            best = metrics.iloc[0]
            k1, k2, k3 = st.columns(3)
            k1.metric("Selected Model", fc["best_model"])
            k2.metric(f"Holdout RMSE ({fc['holdout_months']} months)", f"{best['RMSE']:,.2f}")
            k3.metric("Holdout MAPE", "n/a" if pd.isna(best["MAPE (%)"]) else f"{best['MAPE (%)']:.1f}%")
            if fc["dropped_partial_month"]:
                st.caption(f"Excluded `{fc['dropped_partial_month']}`: the data stops part-way through that month.")

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fut["month"].tolist() + fut["month"].tolist()[::-1],
                                     y=fut["upper"].tolist() + fut["lower"].tolist()[::-1], fill="toself",
                                     line=dict(width=0), opacity=0.25, name="95% interval", hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=hist["month"], y=hist["actual"], mode="lines+markers", name="Actual"))
            if hist["fitted"].notna().any():
                fig.add_trace(go.Scatter(x=hist["month"], y=hist["fitted"], mode="lines", name="Model fit",
                                         line=dict(dash="dot")))
            fig.add_trace(go.Scatter(x=hist["month"], y=hist["holdout_prediction"], mode="markers",
                                     name="Holdout prediction", marker=dict(symbol="x", size=10)))
            fig.add_trace(go.Scatter(x=fut["month"], y=fut["forecast"], mode="lines+markers", name="Forecast",
                                     line=dict(dash="dash")))
            fig.add_vrect(x0=hist["month"].iloc[-fc["holdout_months"]], x1=hist["month"].iloc[-1],
                          opacity=0.08, line_width=0, annotation_text="holdout", annotation_position="top left")
            fig.update_layout(height=440, xaxis_title="Month", yaxis_title=fc["measure"], hovermode="x unified")
            st.plotly_chart(fig, width="stretch")

            m1, m2 = st.columns([1, 1.2])
            with m1:
                st.subheader("Model Comparison (holdout)")
                st.dataframe(metrics, width="stretch", hide_index=True)
            with m2:
                st.subheader("Forecast")
                show = fut.assign(month=fut["month"].dt.strftime("%Y-%m")).round(2)
                st.dataframe(show, width="stretch", hide_index=True)
            st.download_button("Download Forecast (CSV)", show.to_csv(index=False), f"{name}_forecast.csv", "text/csv")

    with tab_co:
        st.caption("Customers are grouped by the month of their first completed purchase. Each cell shows how that "
                   "cohort behaved N months later; blank cells are months after the data ends.")
        max_periods = st.slider("Months after first purchase", 3, 24, 12)
        co_key = f"cohorts_{name}_{max_periods}"
        if st.button("Build Cohorts", type="primary"):
            with st.spinner("Computing cohorts in Spark..."):
                st.session_state[co_key] = cohort_analysis(load_silver(spark, audit.silver_path), max_periods)
        co = st.session_state.get(co_key)
        if not co:
            st.info("Click **Build Cohorts** to compute retention and lifetime value.")
        elif "error" in co:
            st.warning(co["error"])
        else:
            view = st.radio("Show", ["Retention %", "Cumulative revenue per customer"], horizontal=True)
            matrix = co["retention_pct"] if view == "Retention %" else co["ltv_per_customer"]
            fig_c = px.imshow(matrix, aspect="auto", color_continuous_scale="Blues", text_auto=".0f",
                              labels=dict(x="Months since first purchase", y="Cohort (first purchase)", color=view))
            fig_c.update_layout(height=max(320, 26 * len(matrix) + 120), xaxis=dict(dtick=1))
            st.plotly_chart(fig_c, width="stretch")

            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Average Retention Curve")
                fig_r = px.line(co["avg_retention"], x="period", y="retention_pct", markers=True,
                                labels={"period": "Months since first purchase", "retention_pct": "Retention %"})
                fig_r.update_layout(height=320, xaxis=dict(dtick=1))
                st.plotly_chart(fig_r, width="stretch")
            with c2:
                st.subheader("New Customers per Cohort")
                fig_s = px.bar(co["cohort_sizes"], x="cohort", y="new_customers",
                               labels={"cohort": "Cohort", "new_customers": "New customers"})
                fig_s.update_layout(height=320)
                st.plotly_chart(fig_s, width="stretch")
            st.download_button("Download Cohort Matrix (CSV)", matrix.to_csv(), f"{name}_cohorts.csv", "text/csv")
