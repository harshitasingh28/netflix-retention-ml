# src/modeling/segmentation.py

from __future__ import annotations
import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import joblib


DATA_PATH = "data/processed/netflix_final_ml_ready.csv"
OUT_DATA_PATH = "data/processed/netflix_segmented.csv"

MODEL_DIR = "models/"
SCALER_PATH = os.path.join(MODEL_DIR, "segmentation_scaler.pkl")
PCA_PATH = os.path.join(MODEL_DIR, "segmentation_pca.pkl")
KMEANS_PATH = os.path.join(MODEL_DIR, "segmentation_kmeans.pkl")


def ensure_dirs():
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)


def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def select_segmentation_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Only numeric, behavior-driven features.
    These define customer 'type', not churn outcome.
    """
    features = [
        "tenure_months_final",
        "monthly_spend",
        "engagement_score",
        "value_score",
        "binge_index",
        "watch_intensity",
        "price_per_profile",
        "inactivity_score",
        "genre_title_count",
        "genre_movie_ratio",
        "genre_avg_rating_score",
    ]

    existing = [f for f in features if f in df.columns]
    return df[existing]


def build_segmentation(df: pd.DataFrame, n_clusters: int = 4):
    X = select_segmentation_features(df)

    # 1️⃣ Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2️⃣ PCA (retain ~80–90% variance)
    pca = PCA(n_components=0.85, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    # 3️⃣ KMeans
    kmeans = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=20,
        random_state=42
    )
    clusters = kmeans.fit_predict(X_pca)

    return clusters, scaler, pca, kmeans


def label_segments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optional human-readable naming later.
    For now we assign numeric segment IDs.
    """
    df = df.copy()
    df["segment"] = df["segment"].astype(int)
    return df


def create_segment_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create human-readable labels for segments using cluster statistics.
    """
    df = df.copy()

    stats = df.groupby("segment").agg(
        avg_spend=("monthly_spend", "mean"),
        avg_eng=("engagement_score", "mean"),
        avg_inact=("inactivity_score", "mean"),
        churn_rate=("churned", "mean")
    ).reset_index()

    labels = {}
    for _, r in stats.iterrows():
        s = int(r["segment"])

        if r["avg_spend"] > stats["avg_spend"].median() and r["avg_eng"] > stats["avg_eng"].median() and r["avg_inact"] < stats["avg_inact"].median():
            labels[s] = "Loyal High-Value"
        elif r["avg_spend"] > stats["avg_spend"].median() and r["avg_inact"] > stats["avg_inact"].median():
            labels[s] = "High-Value At-Risk"
        elif r["avg_eng"] > stats["avg_eng"].median():
            labels[s] = "Engaged Low-Spend"
        else:
            labels[s] = "Low Engagement / Budget"

    df["segment_label"] = df["segment"].map(labels)
    return df



def main():
    ensure_dirs()

    df = load_data()
    clusters, scaler, pca, kmeans = build_segmentation(df, n_clusters=4)

    df["segment"] = clusters
    df = create_segment_labels(df)

    # Save outputs
    df.to_csv(OUT_DATA_PATH, index=False)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(pca, PCA_PATH)
    joblib.dump(kmeans, KMEANS_PATH)

    print("✅ Segmentation completed")
    print("Segment counts:")
    print(df["segment"].value_counts().sort_index())

    print("\nPCA components:", pca.n_components_)
    print("Explained variance:", round(pca.explained_variance_ratio_.sum(), 3))


if __name__ == "__main__":
    main()
