"""Gold Layer: Distributed Spark MLlib Customer Segmentation (RFM K-Means) & Market Basket (FP-Growth)."""

from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd
from pyspark.ml import Pipeline
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.ml.fpm import FPGrowth
from pyspark.sql import DataFrame, functions as F


def build_rfm_table(silver_df: DataFrame) -> Tuple[DataFrame, str]:
    """Compute customer-level Recency, Frequency, Monetary features in Spark.

    Excludes cancellations, returns, and rows with missing CustomerID.
    Reference date is declared as max(invoice_date) + 1 day.
    """
    eligible = silver_df.filter(
        ~F.col("is_cancellation")
        & ~F.col("is_return")
        & F.col("has_customer_id")
        & (F.col("line_total") > 0)
    )

    max_date = eligible.agg(F.max(F.to_date("invoice_date")).alias("m")).collect()[0]["m"]
    ref_date_col = F.date_add(F.lit(max_date), 1)

    rfm = (
        eligible.groupBy("customer_id")
        .agg(
            F.datediff(ref_date_col, F.max(F.to_date("invoice_date"))).alias("recency"),
            F.countDistinct("invoice_no").alias("frequency"),
            F.round(F.sum("line_total"), 2).alias("monetary"),
            F.first("country").alias("country"),
        )
        .filter((F.col("recency") >= 0) & (F.col("frequency") > 0) & (F.col("monetary") > 0))
        .withColumn("log_recency", F.log1p(F.col("recency")))
        .withColumn("log_frequency", F.log1p(F.col("frequency")))
        .withColumn("log_monetary", F.log1p(F.col("monetary")))
    )
    return rfm, str(max_date)


def _assign_segment_labels(profile_pdf: pd.DataFrame) -> Dict[int, str]:
    """Derive interpretable segment labels strictly from observed cluster RFM stats."""
    ranked = profile_pdf.copy()
    ranked["score"] = (
        ranked["avg_monetary"].rank(pct=True) * 0.45
        + ranked["avg_frequency"].rank(pct=True) * 0.35
        + (1.0 - ranked["avg_recency"].rank(pct=True)) * 0.20
    )
    ranked = ranked.sort_values("score", ascending=False).reset_index(drop=True)

    palette_names = [
        "Champions (High Value, Frequent)",
        "Loyal Regulars",
        "Potential Loyalists",
        "Occasional Buyers",
        "At-Risk Customers",
        "Hibernating / Dormant",
    ]
    n = len(ranked)
    labels: Dict[int, str] = {}
    for idx, row in ranked.iterrows():
        cid = int(row["cluster"])
        if n <= len(palette_names):
            # Spread across top-to-bottom labels proportionally
            pick = round(idx * (len(palette_names) - 1) / max(n - 1, 1))
            labels[cid] = palette_names[pick]
        else:
            labels[cid] = f"Segment {cid + 1}"
    return labels


def run_kmeans_segmentation(
    silver_df: DataFrame,
    k_selected: int = 4,
    k_range: range = range(2, 7),
    seed: int = 42,
) -> Dict[str, object]:
    """Evaluate Spark ML K-Means over candidate k values and return profiles for k_selected."""
    rfm_df, ref_date = build_rfm_table(silver_df)
    rfm_df = rfm_df.persist()
    customer_count = rfm_df.count()

    if customer_count < max(k_range):
        rfm_df.unpersist()
        return {"error": f"Not enough unique customers ({customer_count}) for clustering."}

    assembler = VectorAssembler(
        inputCols=["log_recency", "log_frequency", "log_monetary"],
        outputCol="raw_features",
    )
    scaler = StandardScaler(inputCol="raw_features", outputCol="features", withMean=True, withStd=True)
    prep_pipeline = Pipeline(stages=[assembler, scaler])
    scaled_df = prep_pipeline.fit(rfm_df).transform(rfm_df).persist()

    evaluator = ClusteringEvaluator(featuresCol="features", predictionCol="cluster", metricName="silhouette")

    eval_rows = []
    selected_predictions = None
    for k in k_range:
        if k >= customer_count:
            continue
        km = KMeans(k=k, seed=seed, featuresCol="features", predictionCol="cluster", maxIter=30)
        model = km.fit(scaled_df)
        preds = model.transform(scaled_df)
        sil = float(evaluator.evaluate(preds))
        wssse = float(model.summary.trainingCost)
        eval_rows.append({"k": k, "silhouette": round(sil, 4), "wssse": round(wssse, 2)})
        if k == k_selected:
            selected_predictions = preds

    if selected_predictions is None:
        km = KMeans(k=k_selected, seed=seed, featuresCol="features", predictionCol="cluster", maxIter=30)
        selected_predictions = km.fit(scaled_df).transform(scaled_df)

    # Cluster summary profiles
    profile_df = (
        selected_predictions.groupBy("cluster")
        .agg(
            F.count("*").alias("customers"),
            F.round(F.avg("recency"), 1).alias("avg_recency"),
            F.round(F.avg("frequency"), 2).alias("avg_frequency"),
            F.round(F.avg("monetary"), 2).alias("avg_monetary"),
            F.round(F.sum("monetary"), 2).alias("total_revenue"),
        )
        .orderBy("cluster")
    )
    profile_pdf = profile_df.toPandas()
    total_cust = profile_pdf["customers"].sum()
    profile_pdf["share_pct"] = (profile_pdf["customers"] / total_cust * 100).round(1)

    label_map = _assign_segment_labels(profile_pdf)
    profile_pdf["segment"] = profile_pdf["cluster"].map(label_map)

    # Customer-level table (capped to 10,000 for responsive scatter plotting, full lookup via Spark/pandas)
    customers_pdf = (
        selected_predictions
        .select("customer_id", "country", "recency", "frequency", "monetary", "cluster")
        .toPandas()
    )
    customers_pdf["segment"] = customers_pdf["cluster"].map(label_map)

    # Simple transparent churn-risk indicator based on observed recency percentile
    rec_p75 = customers_pdf["recency"].quantile(0.75)
    rec_p50 = customers_pdf["recency"].quantile(0.50)
    customers_pdf["churn_risk"] = customers_pdf["recency"].apply(
        lambda r: "High" if r >= rec_p75 else ("Medium" if r >= rec_p50 else "Low")
    )

    scaled_df.unpersist()
    rfm_df.unpersist()

    return {
        "reference_date": ref_date,
        "customer_count": customer_count,
        "eval_df": pd.DataFrame(eval_rows),
        "profile_df": profile_pdf,
        "customers_df": customers_pdf,
    }


def run_fpgrowth_basket(
    silver_df: DataFrame,
    min_support: float = 0.015,
    min_confidence: float = 0.25,
    max_rules: int = 50,
) -> Dict[str, object]:
    """Mine invoice-level product baskets using Spark MLlib FP-Growth."""
    eligible = (
        silver_df
        .filter(~F.col("is_cancellation") & ~F.col("is_return"))
        .filter(F.col("description").isNotNull() & (F.trim(F.col("description")) != ""))
    )

    baskets = (
        eligible.groupBy("invoice_no")
        .agg(F.collect_set("description").alias("items"))
        .filter(F.size("items") >= 2)
        .persist()
    )
    basket_count = baskets.count()
    if basket_count < 10:
        baskets.unpersist()
        return {"error": "Not enough multi-item invoices for FP-Growth.", "basket_count": basket_count}

    fp = FPGrowth(itemsCol="items", minSupport=min_support, minConfidence=min_confidence)
    model = fp.fit(baskets)

    rules_df = (
        model.associationRules
        .withColumn("antecedent_str", F.array_join("antecedent", " + "))
        .withColumn("consequent_str", F.array_join("consequent", " + "))
        .select(
            F.col("antecedent_str").alias("If Customer Buys"),
            F.col("consequent_str").alias("Also Likely To Buy"),
            F.round("support", 4).alias("Support"),
            F.round("confidence", 4).alias("Confidence"),
            F.round("lift", 3).alias("Lift"),
        )
        .orderBy(F.desc("Lift"), F.desc("Confidence"))
        .limit(max_rules)
    )
    rules_pdf = rules_df.toPandas()
    baskets.unpersist()

    return {
        "basket_count": basket_count,
        "rules_df": rules_pdf,
    }
