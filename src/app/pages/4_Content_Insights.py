# src/app/pages/4_Content_Insights.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os


TITLES_PATH = "data/interim/titles_clean.csv"
CUSTOMER_PATH = "data/processed/netflix_segmented.csv"

st.set_page_config(page_title="Content Insights", layout="wide")
st.title("🎥 Content Insights")
st.caption("Explore Netflix content metadata and its relationship with customer behavior.")


@st.cache_data
def load_titles():
    return pd.read_csv(TITLES_PATH)


@st.cache_data
def load_customers():
    return pd.read_csv(CUSTOMER_PATH)


# --------------------------
# Safety checks
# --------------------------
if not os.path.exists(TITLES_PATH):
    st.error(f"Missing file: {TITLES_PATH}. Run Step 2 first.")
    st.stop()

titles = load_titles()

# If genres saved as "['a','b']" string, attempt to parse
if "genres" in titles.columns:
    sample = titles["genres"].iloc[0]
    if isinstance(sample, str) and sample.startswith("[") and sample.endswith("]"):
        titles["genres"] = titles["genres"].apply(lambda x: eval(x) if isinstance(x, str) else x)
else:
    st.error("Column 'genres' not found in titles dataset. Re-run Step 2 cleaning.")
    st.stop()

# --------------------------
# Build genre table
# --------------------------
base_cols = [c for c in ["type", "release_year", "rating_score", "duration_minutes", "seasons", "genres"] if c in titles.columns]
g = titles[base_cols].explode("genres").rename(columns={"genres": "genre"})
g["genre"] = g["genre"].fillna("unknown").astype(str).str.strip().str.lower()
g["type_clean"] = g["type"].astype(str).str.strip().str.lower()

# Aggregations
genre_stats = g.groupby("genre").agg(
    title_count=("genre", "size"),
    avg_release_year=("release_year", "mean"),
    avg_rating_score=("rating_score", "mean"),
    avg_duration_minutes=("duration_minutes", "mean"),
    avg_seasons=("seasons", "mean"),
    movie_ratio=("type_clean", lambda s: (s == "movie").mean())
).reset_index()

genre_stats = genre_stats.sort_values("title_count", ascending=False)

# Sidebar control
st.sidebar.header("Controls")
top_n = st.sidebar.slider("Top N genres", min_value=5, max_value=50, value=15, step=5)
top = genre_stats.head(top_n)

st.divider()

# --------------------------
# Plots
# --------------------------
c1, c2 = st.columns(2)

fig1 = px.bar(
    top,
    x="genre",
    y="title_count",
    title="Top Genres by Number of Titles",
    labels={"genre": "Genre", "title_count": "Titles"}
)
c1.plotly_chart(fig1, use_container_width=True)

fig2 = px.bar(
    top,
    x="genre",
    y="movie_ratio",
    title="Movie Ratio by Genre (1 = mostly movies, 0 = mostly TV)",
    labels={"genre": "Genre", "movie_ratio": "Movie Ratio"}
)
c2.plotly_chart(fig2, use_container_width=True)

c3, c4 = st.columns(2)

fig3 = px.bar(
    top,
    x="genre",
    y="avg_rating_score",
    title="Average Rating Maturity Score by Genre",
    labels={"genre": "Genre", "avg_rating_score": "Rating Score"}
)
c3.plotly_chart(fig3, use_container_width=True)

fig4 = px.bar(
    top,
    x="genre",
    y="avg_release_year",
    title="Average Release Year by Genre",
    labels={"genre": "Genre", "avg_release_year": "Avg Release Year"}
)
c4.plotly_chart(fig4, use_container_width=True)

st.divider()

st.subheader("📋 Genre Summary Table")
st.dataframe(top, use_container_width=True)

# --------------------------
# Optional: Customer churn by matched genre
# --------------------------
if os.path.exists(CUSTOMER_PATH):
    customers = load_customers()

    if "matched_genre" in customers.columns and "churned" in customers.columns:
        st.divider()
        st.subheader("👥 Customer Churn Rate by Matched Genre")

        churn_by_genre = (
            customers.groupby("matched_genre")["churned"]
            .mean()
            .reset_index()
            .rename(columns={"matched_genre": "genre", "churned": "churn_rate"})
        )
        churn_by_genre["churn_rate_percent"] = churn_by_genre["churn_rate"] * 100
        churn_by_genre = churn_by_genre.sort_values("churn_rate_percent", ascending=False).head(top_n)

        fig5 = px.bar(
            churn_by_genre,
            x="genre",
            y="churn_rate_percent",
            title="Churn Rate by Genre (Top Genres)",
            labels={"genre": "Genre", "churn_rate_percent": "Churn Rate (%)"}
        )
        st.plotly_chart(fig5, use_container_width=True)

        st.info("This is correlation, not causation — but it can hint at which audience segments churn more.")
