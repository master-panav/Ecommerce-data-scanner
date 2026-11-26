"""E-Commerce Customer Intelligence & Sales Analytics — PySpark Big Data Workbench.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import re
import traceback
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
    filter_silver,
    monthly_trends,
    shopping_heatmap,
    top_products,
)
from engine.ml_models import run_fpgrowth_basket, run_kmeans_segmentation
from engine.pipeline import EmptySilverError, PipelineAudit, load_saved_audits, load_silver, run_medallion_etl
from engine.schemas import DATASET_CATALOG, SchemaMappingError
from engine.spark_session import get_spark, get_spark_telemetry, set_shuffle_partitions

APP_DIR = Path(__file__).resolve().parent
LAKE_ROOT = APP_DIR / "data_lake"
BRONZE_DIR = LAKE_ROOT / "bronze"
SUPPORTED_UPLOAD_TYPES = ["csv", "gz", "xlsx", "parquet"]

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


def discover_bronze_sources() -> dict[str, Path]:
    """List all CSV, XLSX, Parquet files and multi-table folders inside data_lake/bronze."""
    BRONZE_DIR.mkdir(parents=True, exist_ok=True)
    sources: dict[str, Path] = {}
    for item in sorted(BRONZE_DIR.iterdir(), key=lambda x: x.name.lower()):
        if item.name.startswith((".", "~$", "_")) or item.name.endswith(".part"):  # hidden, Excel lock, partial uploads
            continue
        if item.is_dir() or item.suffix.lower() in {".csv", ".xlsx", ".parquet", ".gz"}:
            sources[item.name] = item
    return sources


def safe_upload_name(name: str) -> str:
    """Strip any path components and unsafe characters from an uploaded file name."""
    base = Path(name.replace("\\", "/")).name
    cleaned = re.sub(r"[^A-Za-z0-9._ -]+", "_", base).strip(" .")
    return cleaned or "uploaded_dataset.csv"


def save_upload(uploaded) -> str:
    """Persist an uploaded file to Bronze exactly once (Streamlit reruns the script on every click)."""
    upload_id = getattr(uploaded, "file_id", None) or f"{uploaded.name}:{uploaded.size}"
    name = safe_upload_name(uploaded.name)
    if st.session_state.get("last_upload_id") != upload_id:
        dest = BRONZE_DIR / name
        tmp = dest.with_name(dest.name + ".part")
        with open(tmp, "wb") as fh:
            fh.write(uploaded.getbuffer())
        tmp.replace(dest)
        st.session_state["last_upload_id"] = upload_id
        st.session_state["bronze_choice"] = name
        st.toast(f"Saved `{name}` to the Bronze layer.")
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
            uploaded = st.file_uploader(
                "Upload CSV / CSV.GZ / XLSX / Parquet",
                type=SUPPORTED_UPLOAD_TYPES,
                label_visibility="collapsed",
                help="Files are saved into data_lake/bronze/. For multi-GB files, copy them there directly.",
            )
            if uploaded is not None:
                save_upload(uploaded)
                sources = discover_bronze_sources()

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

    audit_table = pd.DataFrame([
        {"Check": "Cancellation Invoices (Prefix 'C')", "Row Count": audit.cancellation_rows, "Handling": "Flagged (`is_cancellation`) for Gross vs. Net reporting"},
        {"Check": "Negative Quantity / Returns", "Row Count": audit.return_rows, "Handling": "Flagged (`is_return`), excluded from completed sales"},
        {"Check": "Missing CustomerID", "Row Count": audit.missing_customer_rows, "Handling": "Kept for aggregate BI; excluded from customer RFM"},
        {"Check": "Invalid / Non-Positive Unit Price", "Row Count": audit.invalid_price_rows, "Handling": "Dropped in Silver layer"},
        {"Check": "Null / Unparseable Timestamp", "Row Count": audit.null_date_rows, "Handling": "Dropped in Silver layer"},
        {"Check": "Exact Duplicate Rows", "Row Count": audit.duplicate_rows, "Handling": "Deduplicated via Spark `dropDuplicates()`"},
    ])

    col_a, col_b = st.columns([1.2, 1])
    with col_a:
        st.subheader("Cleaning Audit Log")
        st.dataframe(audit_table, width="stretch", hide_index=True)
    with col_b:
        st.subheader("Data Quality Issue Distribution")
        fig = px.bar(audit_table, x="Row Count", y="Check", orientation="h", text_auto=True)
        fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="")
        st.plotly_chart(fig, width="stretch")

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
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Net Revenue", f"{kpis['net_revenue']:,.2f}")
    m2.metric("Gross Sales", f"{kpis['gross_sales']:,.2f}")
    m3.metric("Completed Orders", f"{kpis['orders']:,}")
    m4.metric("Average Order Value", f"{kpis['aov']:,.2f}")
    m5.metric("Return Rate", f"{kpis['return_rate_pct']}%")

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
