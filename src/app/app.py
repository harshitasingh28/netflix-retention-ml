# src/app/app.py

import streamlit as st
import pandas as pd
import joblib
import os


st.set_page_config(page_title="Netflix Retention Intelligence", layout="wide")

DATA_SEGMENTED = "data/processed/netflix_segmented.csv"
DATA_RETENTION = "data/processed/netflix_retention_recommendations.csv"
MODEL_PATH = "models/churn_model.pkl"

st.title("🎬 Netflix Customer Segmentation & Retention Intelligence")
st.caption("Segmentation • Churn Prediction • Retention Actions • Business Insights")

# Load datasets
@st.cache_data
def load_segmented():
    return pd.read_csv(DATA_SEGMENTED)

@st.cache_data
def load_retention():
    return pd.read_csv(DATA_RETENTION)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

# Quick checks
if not os.path.exists(DATA_SEGMENTED):
    st.error(f"Missing file: {DATA_SEGMENTED}. Run Step 4 first.")
    st.stop()

if not os.path.exists(DATA_RETENTION):
    st.error(f"Missing file: {DATA_RETENTION}. Run Step 7 first.")
    st.stop()

df_seg = load_segmented()
df_ret = load_retention()

# KPIs
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Customers", f"{len(df_seg):,}")
c2.metric("Churn Rate", f"{df_seg['churned'].mean()*100:.2f}%")
c3.metric("Segments", f"{df_seg['segment'].nunique()}")
c4.metric("High Risk Customers", f"{(df_ret['risk_bucket']=='high').sum():,}")

st.divider()

st.subheader("📌 What do you want to explore?")
st.markdown(
"""
Use the left sidebar pages:
- **Segmentation Insights** → cluster breakdown + churn rates  
- **Churn Prediction** → predict churn probability for selected customer  
- **Retention Actions** → top priority customers to save + recommended actions  
- **Content Insights** → genre intelligence features  
"""
)
