# src/features/build_features.py

from __future__ import annotations
import os
import pandas as pd
import numpy as np


CHURN_CLEAN_PATH = "data/interim/churn_clean.csv"
TITLES_CLEAN_PATH = "data/interim/titles_clean.csv"
OUT_PATH = "data/processed/netflix_final_ml_ready.csv"


def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


# --------------------------
# 1) Customer Feature Engineering
# --------------------------
def engineer_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates Netflix-style customer features.
    Works even if some columns are missing (it checks before using).
    """
    df = df.copy()

    # Helper to safely get a column or default
    def col(name: str, default=0.0):
        return df[name] if name in df.columns else default

    # Identify likely spend column
    # Prefer 'monthly_fee' then 'monthly_charges'
    if "monthly_fee" in df.columns:
        spend = df["monthly_fee"].astype(float)
        spend_col = "monthly_fee"
    elif "monthly_charges" in df.columns:
        spend = df["monthly_charges"].astype(float)
        spend_col = "monthly_charges"
    else:
        spend = pd.Series([np.nan] * len(df))
        spend_col = None

    # Identify likely tenure column
    if "tenure_months" in df.columns:
        tenure = df["tenure_months"].astype(float)
    elif "tenure" in df.columns:
        tenure = df["tenure"].astype(float)
    else:
        tenure = pd.Series([np.nan] * len(df))

    # Optional usage columns
    watch_hours = col("watch_hours", 0.0)
    avg_watch_time_per_day = col("avg_watch_time_per_day", 0.0)
    last_login_days = col("last_login_days", 0.0)
    profiles = col("number_of_profiles", 1.0)

    # Safe denominators
    last_login_safe = pd.to_numeric(last_login_days, errors="coerce").fillna(0).replace(0, 0.5)
    spend_safe = pd.to_numeric(spend, errors="coerce")
    spend_safe = spend_safe.fillna(spend_safe.median() if spend_safe.notna().any() else 1.0).replace(0, 1.0)
    profiles_safe = pd.to_numeric(profiles, errors="coerce").fillna(1).replace(0, 1)

    # Core features
    df["tenure_months_final"] = pd.to_numeric(tenure, errors="coerce").fillna(tenure.median() if pd.Series(tenure).notna().any() else 0)
    df["monthly_spend"] = spend_safe

    df["binge_index"] = pd.to_numeric(watch_hours, errors="coerce").fillna(0) / (last_login_safe + 1)
    df["value_score"] = pd.to_numeric(watch_hours, errors="coerce").fillna(0) / spend_safe
    df["price_per_profile"] = spend_safe / profiles_safe

    df["inactivity_score"] = np.log1p(last_login_safe)

    # Watch intensity proxy
    df["watch_intensity"] = (
        pd.to_numeric(avg_watch_time_per_day, errors="coerce").fillna(0)
        * pd.to_numeric(watch_hours, errors="coerce").fillna(0)
    )

    # Composite engagement score (tunable weights)
    df["engagement_score"] = (
        (np.log1p(pd.to_numeric(watch_hours, errors="coerce").fillna(0)) * 0.45) +
        (np.log1p(pd.to_numeric(avg_watch_time_per_day, errors="coerce").fillna(0)) * 0.35) -
        (np.log1p(last_login_safe) * 0.20)
    )

    # Optional: subscription tier mapping if subscription_type exists
    if "subscription_type" in df.columns:
        plan_map = {"basic": 1, "standard": 2, "premium": 3, "unknown": 2}
        df["subscription_tier"] = df["subscription_type"].astype(str).str.lower().map(plan_map).fillna(2).astype(int)

    # Optional: risk flags (quantile-based, robust)
    df["is_low_engagement"] = (df["engagement_score"] < df["engagement_score"].quantile(0.25)).astype(int)
    df["is_high_inactivity"] = (pd.to_numeric(last_login_days, errors="coerce").fillna(0) > pd.to_numeric(last_login_days, errors="coerce").fillna(0).quantile(0.75)).astype(int)
    df["is_price_sensitive"] = (df["value_score"] < df["value_score"].quantile(0.25)).astype(int)

    # If we identified a spend column, keep a note (optional)
    if spend_col:
        df["spend_source_col"] = spend_col
    else:
        df["spend_source_col"] = "unknown"

    return df

def add_rfm_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    RFM analysis using Netflix-like proxies:
      R (Recency): last_login_days  (lower is better)
      F (Frequency): watch_hours    (higher is better)
      M (Monetary): monthly_spend   (higher is better)

    Produces:
      r_score, f_score, m_score (1..5)
      rfm_score (3..15)
      rfm_segment (e.g., '555')
      rfm_label (Champions, Loyal, At-Risk, etc.)
    """
    df = df.copy()

    # Safety checks (must exist after Step 3 customer features)
    required = ["last_login_days", "watch_hours", "monthly_spend"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for RFM: {missing}")

    # Make sure numeric
    df["last_login_days"] = pd.to_numeric(df["last_login_days"], errors="coerce").fillna(df["last_login_days"].median())
    df["watch_hours"] = pd.to_numeric(df["watch_hours"], errors="coerce").fillna(df["watch_hours"].median())
    df["monthly_spend"] = pd.to_numeric(df["monthly_spend"], errors="coerce").fillna(df["monthly_spend"].median())

    # --- R: lower recency = better (invert scoring)
    # qcut can fail when many duplicate values; rank fixes that
    df["r_score"] = pd.qcut(
        df["last_login_days"].rank(method="first"),
        5,
        labels=[5, 4, 3, 2, 1]
    ).astype(int)

    # --- F: higher watch_hours = better
    df["f_score"] = pd.qcut(
        df["watch_hours"].rank(method="first"),
        5,
        labels=[1, 2, 3, 4, 5]
    ).astype(int)

    # --- M: higher spend = better
    df["m_score"] = pd.qcut(
        df["monthly_spend"].rank(method="first"),
        5,
        labels=[1, 2, 3, 4, 5]
    ).astype(int)

    # Combined
    df["rfm_score"] = df["r_score"] + df["f_score"] + df["m_score"]
    df["rfm_segment"] = (
        df["r_score"].astype(str) +
        df["f_score"].astype(str) +
        df["m_score"].astype(str)
    )

    # Human readable business labels (simple + effective)
    def label_row(r, f, m):
        # Champions: best in everything
        if r >= 4 and f >= 4 and m >= 4:
            return "Champions"
        # Loyal: frequent + decent spend
        if f >= 4 and m >= 3:
            return "Loyal"
        # Potential Loyalists: recently active but not yet high frequency/spend
        if r >= 4 and f >= 2:
            return "Potential Loyalist"
        # Big Spenders: high spend but not necessarily frequent
        if m >= 4 and f <= 3:
            return "Big Spender"
        # At Risk: not recent but used to be good
        if r <= 2 and (f >= 3 or m >= 3):
            return "At Risk"
        # Hibernating: low recency + low frequency
        if r <= 2 and f <= 2:
            return "Hibernating"
        # New/Low value
        if r >= 4 and f <= 2 and m <= 2:
            return "New / Low Value"
        return "Regular"

    df["rfm_label"] = df.apply(lambda x: label_row(x["r_score"], x["f_score"], x["m_score"]), axis=1)

    return df



# --------------------------
# 2) Genre Intelligence from Titles
# --------------------------
def build_genre_feature_table(titles: pd.DataFrame) -> pd.DataFrame:
    """
    Build aggregated stats per genre using titles dataset.
    Requires titles_clean.csv to have:
      - genres (stringified list or list)
      - type
      - release_year
      - rating_score
      - duration_minutes
      - seasons
    """
    t = titles.copy()

    # If genres got saved as a string like "['dramas','thrillers']"
    # convert it back to list safely.
    if "genres" in t.columns and t["genres"].dtype == "object":
        # attempt parse if looks like list-string
        sample = t["genres"].iloc[0]
        if isinstance(sample, str) and sample.startswith("[") and sample.endswith("]"):
            t["genres"] = t["genres"].apply(lambda x: eval(x) if isinstance(x, str) else x)

    # explode genres
    base_cols = [c for c in ["type", "release_year", "rating_score", "duration_minutes", "seasons", "genres"] if c in t.columns]
    g = t[base_cols].explode("genres").rename(columns={"genres": "genre"})

    g["genre"] = g["genre"].fillna("unknown").astype(str).str.strip().str.lower()
    if "type" in g.columns:
        g["type_clean"] = g["type"].astype(str).str.strip().str.lower()
    else:
        g["type_clean"] = "unknown"

    # Basic aggregations
    agg = g.groupby("genre").agg(
        genre_title_count=("genre", "size"),
        genre_avg_release_year=("release_year", "mean") if "release_year" in g.columns else ("genre", "size"),
        genre_avg_rating_score=("rating_score", "mean") if "rating_score" in g.columns else ("genre", "size"),
        genre_avg_duration_minutes=("duration_minutes", "mean") if "duration_minutes" in g.columns else ("genre", "size"),
        genre_avg_seasons=("seasons", "mean") if "seasons" in g.columns else ("genre", "size"),
    ).reset_index()

    # Movie ratio per genre
    g["is_movie"] = (g["type_clean"] == "movie").astype(int)
    movie_ratio = g.groupby("genre")["is_movie"].mean().reset_index().rename(columns={"is_movie": "genre_movie_ratio"})
    agg = agg.merge(movie_ratio, on="genre", how="left")

    # Fill missing numeric values with median
    numeric_cols = [c for c in agg.columns if c != "genre"]
    for c in numeric_cols:
        agg[c] = pd.to_numeric(agg[c], errors="coerce")
        agg[c] = agg[c].fillna(agg[c].median())

    return agg


# --------------------------
# 3) Merge Genre Features into Customers
# --------------------------
def merge_content_features(customers: pd.DataFrame, genre_table: pd.DataFrame) -> pd.DataFrame:
    """
    Attach genre intelligence stats to each customer based on their favorite genre.
    Common churn columns that may represent favorite genre:
      - favorite_genre
      - preferred_genre
      - genre
    """
    df = customers.copy()

    # Detect genre column in customer data
    genre_col_candidates = ["favorite_genre", "preferred_genre", "genre"]
    genre_col = next((c for c in genre_col_candidates if c in df.columns), None)

    if genre_col is None:
        # If no genre column exists, just attach global median as defaults
        for c in genre_table.columns:
            if c != "genre":
                df[c] = genre_table[c].median()
        df["matched_genre"] = "unknown"
        return df

    df["matched_genre"] = df[genre_col].astype(str).str.strip().str.lower().fillna("unknown")

    merged = df.merge(
        genre_table,
        left_on="matched_genre",
        right_on="genre",
        how="left"
    )

    # For unmatched genres, fill with medians
    for c in genre_table.columns:
        if c == "genre":
            continue
        merged[c] = pd.to_numeric(merged[c], errors="coerce")
        merged[c] = merged[c].fillna(genre_table[c].median())

    merged = merged.drop(columns=["genre"], errors="ignore")
    return merged


# --------------------------
# 4) Build Final Dataset
# --------------------------
def build_final_dataset() -> pd.DataFrame:
    churn = pd.read_csv(CHURN_CLEAN_PATH)
    titles = pd.read_csv(TITLES_CLEAN_PATH)

    churn_feat = engineer_customer_features(churn)
    churn_feat = add_rfm_features(churn_feat)
    genre_table = build_genre_feature_table(titles)

    final_df = merge_content_features(churn_feat, genre_table)

    # Final cleanup: drop duplicates
    final_df = final_df.drop_duplicates()

    return final_df


def main():
    _ensure_dir(OUT_PATH)

    final_df = build_final_dataset()
    final_df.to_csv(OUT_PATH, index=False)

    print("✅ Final ML-ready dataset saved to:", OUT_PATH)
    print("Final shape:", final_df.shape)

    # Quick check of key columns
    preview_cols = [c for c in ["churned", "engagement_score", "value_score", "binge_index", "matched_genre", "genre_title_count"] if c in final_df.columns]
    print("Preview cols:", preview_cols)
    print(final_df[preview_cols].head(5))





if __name__ == "__main__":
    main()
