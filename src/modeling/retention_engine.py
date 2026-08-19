# src/modeling/retention_engine.py

from __future__ import annotations
import os
import pandas as pd
import numpy as np
import joblib


DATA_PATH = "data/processed/netflix_segmented.csv"
MODEL_PATH = "models/churn_model.pkl"
OUT_PATH = "data/processed/netflix_retention_recommendations.csv"


def ensure_dirs():
    os.makedirs("data/processed", exist_ok=True)


def load_inputs():
    df = pd.read_csv(DATA_PATH)
    model = joblib.load(MODEL_PATH)
    return df, model


def predict_churn_probability(df: pd.DataFrame, pipeline) -> pd.Series:
    """
    Uses the saved sklearn Pipeline to generate churn probability.
    """
    X = df.drop(columns=["churned"], errors="ignore")
    proba = pipeline.predict_proba(X)[:, 1]
    return pd.Series(proba, index=df.index, name="churn_probability")


def assign_risk_bucket(p: float) -> str:
    if p >= 0.75:
        return "high"
    if p >= 0.40:
        return "medium"
    return "low"


def compute_clv_proxy(row) -> float:
    """
    Simple CLV proxy (not perfect but realistic):
    CLV ≈ monthly_spend * (tenure_months_final + 1) * engagement_multiplier

    This helps prioritize whom to save first.
    """
    spend = float(row.get("monthly_spend", row.get("monthly_fee", 0.0)))
    tenure = float(row.get("tenure_months_final", row.get("tenure_months", 0.0)))
    engagement = float(row.get("engagement_score", 0.0))

    # convert engagement score to a multiplier roughly in [0.8, 1.3]
    mult = 1.0 + np.tanh(engagement) * 0.3
    return spend * (tenure + 1) * mult


def label_segments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Give human-readable names to numeric clusters.
    This uses simple heuristics based on spend + engagement + inactivity.
    """
    df = df.copy()

    # Compute segment-level means
    grp = df.groupby("segment").agg(
        avg_spend=("monthly_spend", "mean"),
        avg_eng=("engagement_score", "mean"),
        avg_inact=("inactivity_score", "mean"),
        churn_rate=("churned", "mean") if "churned" in df.columns else ("segment", "size")
    ).reset_index()

    # Rank segments by spend and engagement
    spend_rank = grp["avg_spend"].rank(method="dense")
    eng_rank = grp["avg_eng"].rank(method="dense")
    inact_rank = grp["avg_inact"].rank(method="dense")

    # Map segment id -> label
    seg_map = {}
    for i, r in grp.iterrows():
        s = int(r["segment"])

        high_value = spend_rank[i] >= spend_rank.max()
        high_eng = eng_rank[i] >= eng_rank.max()
        high_inact = inact_rank[i] >= inact_rank.max()

        if high_value and high_eng and not high_inact:
            seg_map[s] = "Loyal High-Value"
        elif high_value and high_inact:
            seg_map[s] = "High-Value At-Risk"
        elif (not high_value) and high_eng:
            seg_map[s] = "Engaged Low-Spend"
        else:
            seg_map[s] = "Low Engagement / Budget"

    df["segment_label"] = df["segment"].map(seg_map).fillna("Unknown Segment")
    return df


def recommend_action(row) -> str:
    """
    Rule-based retention playbook (simple but effective).
    Uses:
      - risk_bucket
      - segment_label
      - price sensitivity flags
      - inactivity / engagement
    """
    risk = row["risk_bucket"]
    seg = row["segment_label"]

    price_sensitive = int(row.get("is_price_sensitive", 0)) == 1
    high_inactivity = int(row.get("is_high_inactivity", 0)) == 1
    low_engagement = int(row.get("is_low_engagement", 0)) == 1

    # High risk actions
    if risk == "high":
        if seg == "High-Value At-Risk":
            return "Personal outreach + 2-month discount + curated recommendations"
        if price_sensitive:
            return "Offer plan downgrade/discount + highlight value (downloads, multi-device)"
        if high_inactivity:
            return "Re-engagement campaign: 'Continue watching' + push/email reminders"
        return "Strong retention offer: personalized content + limited-time discount"

    # Medium risk actions
    if risk == "medium":
        if seg == "Loyal High-Value":
            return "Premium retention: early-access content + upgrade trial"
        if low_engagement:
            return "Personalized content emails + genre-based recommendations"
        if price_sensitive:
            return "Bundle/annual plan suggestion + small discount"
        return "Nudge campaign: top picks + reminders + optimize onboarding"

    # Low risk actions
    if seg == "Loyal High-Value":
        return "Upsell: premium upgrade + refer-a-friend"
    if seg == "Engaged Low-Spend":
        return "Cross-sell: suggest Standard/Premium trial"
    return "Maintain: periodic recommendations + new releases notifications"


def build_retention_table(df: pd.DataFrame, churn_prob: pd.Series) -> pd.DataFrame:
    out = df.copy()
    out["churn_probability"] = churn_prob
    out["risk_bucket"] = out["churn_probability"].apply(assign_risk_bucket)

    # CLV proxy + priority score
    out["clv_proxy"] = out.apply(compute_clv_proxy, axis=1)

    # Priority: high churn prob + high CLV
    out["retention_priority"] = out["churn_probability"] * (out["clv_proxy"] / (out["clv_proxy"].median() + 1e-6))

    # Segment labels
    out = label_segments(out)

    # Recommended action
    out["recommended_action"] = out.apply(recommend_action, axis=1)

    # Sort: highest priority first
    out = out.sort_values(by="retention_priority", ascending=False)

    return out


def main():
    ensure_dirs()

    df, pipeline = load_inputs()

    churn_prob = predict_churn_probability(df, pipeline)
    retention_df = build_retention_table(df, churn_prob)

    retention_df.to_csv(OUT_PATH, index=False)

    print("✅ Retention recommendations generated:", OUT_PATH)
    print("Top 5 customers by retention priority:")
    cols = ["churn_probability", "risk_bucket", "segment", "segment_label", "clv_proxy", "retention_priority", "recommended_action"]
    existing = [c for c in cols if c in retention_df.columns]
    print(retention_df[existing].head(5))


if __name__ == "__main__":
    main()
