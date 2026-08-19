# src/app/pages/1_Segmentation_Insights.py

import streamlit as st
import pandas as pd
import plotly.express as px


DATA_PATH = "data/processed/netflix_segmented.csv"

st.set_page_config(page_title="Segmentation Insights", layout="wide")
st.title("🧩 Segmentation Insights")
st.caption("Understand customer clusters and how churn varies across segments.")


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


df = load_data()

# --------------------------
# Sidebar filters
# --------------------------
st.sidebar.header("Filters")

# Optional filters if these columns exist
filters = {}

if "region" in df.columns:
    regions = ["all"] + sorted(df["region"].astype(str).unique().tolist())
    filters["region"] = st.sidebar.selectbox("Region", regions)

if "subscription_type" in df.columns:
    plans = ["all"] + sorted(df["subscription_type"].astype(str).unique().tolist())
    filters["subscription_type"] = st.sidebar.selectbox("Subscription Type", plans)

if "device" in df.columns:
    devices = ["all"] + sorted(df["device"].astype(str).unique().tolist())
    filters["device"] = st.sidebar.selectbox("Device", devices)

# Apply filters
filtered = df.copy()
for k, v in filters.items():
    if v != "all":
        filtered = filtered[filtered[k] == v]

st.write(f"Showing **{len(filtered):,}** customers after filters.")

st.divider()

# --------------------------
# KPI row
# --------------------------
c1, c2, c3 = st.columns(3)
c1.metric("Total Customers", f"{len(filtered):,}")
c2.metric("Churn Rate", f"{filtered['churned'].mean() * 100:.2f}%")
c3.metric("Segments", f"{filtered['segment'].nunique()}")

st.divider()

# --------------------------
# Segment counts plot
# --------------------------
seg_col = "segment_label" if "segment_label" in filtered.columns else "segment"

seg_counts = (
    filtered[seg_col]
    .value_counts()
    .reset_index()
)
seg_counts.columns = ["segment", "count"]

fig_counts = px.bar(
    seg_counts,
    x="segment",
    y="count",
    title="Customers per Segment",
    labels={"segment": "Segment ID", "count": "Customers"}
)
st.plotly_chart(fig_counts, use_container_width=True)

# --------------------------
# Churn rate per segment plot
# --------------------------
seg_churn = (
    filtered.groupby(seg_col)["churned"]
    .mean()
    .reset_index()
    .rename(columns={seg_col: "segment", "churned": "churn_rate"})
)
seg_churn["churn_rate_percent"] = seg_churn["churn_rate"] * 100

fig_churn = px.bar(
    seg_churn.sort_values("segment"),
    x="segment",
    y="churn_rate_percent",
    title="Churn Rate per Segment",
    labels={"segment": "Segment ID", "churn_rate_percent": "Churn Rate (%)"}
)
st.plotly_chart(fig_churn, use_container_width=True)

st.divider()

# --------------------------
# Segment profiling table
# --------------------------
profile_features = [
    "monthly_spend",
    "tenure_months_final",
    "engagement_score",
    "inactivity_score",
    "value_score",
    "binge_index",
    "price_per_profile",
    "watch_intensity",
    "r_score",
    "f_score",
    "m_score",
    "rfm_score",
]

existing_features = [c for c in profile_features if c in filtered.columns]

seg_profile = (
    filtered.groupby(seg_col)[existing_features + ["churned"]]
    .mean()
    .reset_index()
    .rename(columns={seg_col: "segment", "churned": "avg_churn_rate"})
)

seg_profile["avg_churn_rate"] = seg_profile["avg_churn_rate"] * 100
seg_profile = seg_profile.sort_values("segment")

st.subheader("📊 Segment Profiles (Averages)")
st.dataframe(seg_profile, use_container_width=True)

st.info(
    "Tip: A good segment is **distinct** in spend/engagement/inactivity and shows different churn behavior."
)
