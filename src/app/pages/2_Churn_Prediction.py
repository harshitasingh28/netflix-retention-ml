# src/app/pages/2_Churn_Prediction.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os


DATA_PATH = "data/processed/netflix_segmented.csv"
MODEL_PATH = "models/churn_model.pkl"

st.set_page_config(page_title="Churn Prediction", layout="wide")
st.title("📉 Churn Prediction")
st.caption("Select a customer record and predict churn probability using the trained model.")


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def risk_bucket(p: float) -> str:
    if p >= 0.75:
        return "High"
    if p >= 0.40:
        return "Medium"
    return "Low"


# --------------------------
# Safety checks
# --------------------------
if not os.path.exists(DATA_PATH):
    st.error(f"Missing file: {DATA_PATH}. Run Step 4 first.")
    st.stop()

if not os.path.exists(MODEL_PATH):
    st.error(f"Missing model: {MODEL_PATH}. Run Step 5 first.")
    st.stop()

df = load_data()
model = load_model()

# --------------------------
# User selection
# --------------------------
st.sidebar.header("Choose Customer")

max_index = len(df) - 1
idx = st.sidebar.number_input(
    "Customer Row Index",
    min_value=0,
    max_value=int(max_index),
    value=0,
    step=1
)

row = df.iloc[int(idx)].copy()

# Separate X (features) and y (true churn if available)
true_churn = None
if "churned" in row.index:
    true_churn = int(row["churned"])

X_row = row.drop(labels=["churned"], errors="ignore")
X_input = pd.DataFrame([X_row])

# --------------------------
# Prediction
# --------------------------
proba = float(model.predict_proba(X_input)[:, 1][0])
bucket = risk_bucket(proba)

# --------------------------
# Display
# --------------------------
c1, c2, c3 = st.columns(3)
c1.metric("Churn Probability", f"{proba:.3f}")
c2.metric("Risk Bucket", bucket)
if true_churn is not None:
    c3.metric("Actual Churn Label", "Churned" if true_churn == 1 else "Not Churned")
else:
    c3.metric("Actual Churn Label", "N/A")

st.divider()

# Show important fields (only if present)
important_cols = [
    "segment", "rfm_label", "rfm_segment",
    "r_score", "f_score", "m_score", "rfm_score",
    "gender", "region", "device", "payment_method",
    "subscription_type", "subscription_tier",
    "monthly_spend", "monthly_fee",
    "tenure_months_final",
    "engagement_score", "inactivity_score", "value_score", "binge_index"
]
existing = [c for c in important_cols if c in df.columns]

st.subheader("🧾 Customer Snapshot")
st.dataframe(pd.DataFrame([row[existing]]), use_container_width=True)

st.divider()

# Show full row (optional)
with st.expander("Show full customer record"):
    st.dataframe(pd.DataFrame([row]), use_container_width=True)

# --------------------------
# Simple "reason hints" (proxy explanation)
# --------------------------
st.subheader("🧠 Why might this customer churn? (simple signals)")
signals = []

# These are interpretable heuristics, not SHAP (we add SHAP per-customer later if you want)
if "last_login_days" in row.index and row["last_login_days"] > df["last_login_days"].quantile(0.75):
    signals.append("High inactivity: last_login_days is in top 25% (not logging in often).")

if "engagement_score" in row.index and row["engagement_score"] < df["engagement_score"].quantile(0.25):
    signals.append("Low engagement: engagement_score is in bottom 25%.")

if "value_score" in row.index and row["value_score"] < df["value_score"].quantile(0.25):
    signals.append("Low perceived value: watch_hours per spend is in bottom 25%.")

if "monthly_spend" in row.index and "watch_hours" in row.index:
    if row["monthly_spend"] > df["monthly_spend"].quantile(0.75) and row["watch_hours"] < df["watch_hours"].quantile(0.25):
        signals.append("High spend but low usage (possible dissatisfaction / price sensitivity).")

if not signals:
    signals.append("No strong heuristic red flags detected. This customer looks relatively healthy.")

for s in signals:
    st.write("•", s)
