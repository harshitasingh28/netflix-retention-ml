# src/app/pages/3_Retention_Actions.py

import streamlit as st
import pandas as pd
import os

DATA_PATH = "data/processed/netflix_retention_recommendations.csv"

st.set_page_config(page_title="Retention Actions", layout="wide")
st.title("🛟 Retention Actions")
st.caption("Prioritized list of customers to save with recommended actions based on churn risk + value.")

# ✅ This line should appear no matter what (proves page rendered)
st.success("✅ Retention Actions page loaded")

# Sidebar reload (no rerun loop unless clicked)
if st.sidebar.button("🔄 Reload Data"):
    st.cache_data.clear()
    st.rerun()


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


try:
    # --------------------------
    # Safety check
    # --------------------------
    if not os.path.exists(DATA_PATH):
        st.error(f"Missing file: {DATA_PATH}")
        st.info("Run: `python -m src.modeling.retention_engine`")
        st.stop()

    df = load_data(DATA_PATH)

    # --------------------------
    # Ensure required columns exist (fallbacks)
    # --------------------------
    if "risk_bucket" not in df.columns:
        st.warning("Column 'risk_bucket' missing. File might be outdated.")
        df["risk_bucket"] = "unknown"

    if "churn_probability" not in df.columns:
        df["churn_probability"] = 0.0

    if "clv_proxy" not in df.columns:
        df["clv_proxy"] = 0.0

    if "retention_priority" not in df.columns:
        df["retention_priority"] = df["churn_probability"]

    if "recommended_action" not in df.columns:
        df["recommended_action"] = "N/A"

    # --------------------------
    # Sidebar filters
    # --------------------------
    st.sidebar.header("Filters")

    risk_options = ["all"] + sorted(df["risk_bucket"].astype(str).unique().tolist())
    risk_sel = st.sidebar.selectbox("Risk Bucket", risk_options, index=0)

    if "segment_label" in df.columns:
        seg_options = ["all"] + sorted(df["segment_label"].astype(str).unique().tolist())
        seg_sel = st.sidebar.selectbox("Segment Label", seg_options, index=0)
    else:
        seg_sel = "all"

    if "rfm_label" in df.columns:
        rfm_options = ["all"] + sorted(df["rfm_label"].astype(str).unique().tolist())
        rfm_sel = st.sidebar.selectbox("RFM Label", rfm_options, index=0)
    else:
        rfm_sel = "all"

    if "region" in df.columns:
        reg_options = ["all"] + sorted(df["region"].astype(str).unique().tolist())
        reg_sel = st.sidebar.selectbox("Region", reg_options, index=0)
    else:
        reg_sel = "all"

    # Apply filters
    filtered = df.copy()

    if risk_sel != "all":
        filtered = filtered[filtered["risk_bucket"] == risk_sel]

    if seg_sel != "all" and "segment_label" in filtered.columns:
        filtered = filtered[filtered["segment_label"] == seg_sel]

    if rfm_sel != "all" and "rfm_label" in filtered.columns:
        filtered = filtered[filtered["rfm_label"] == rfm_sel]

    if reg_sel != "all" and "region" in filtered.columns:
        filtered = filtered[filtered["region"] == reg_sel]

    st.write(f"Showing **{len(filtered):,}** customers after filters.")
    st.divider()

    # --------------------------
    # KPIs
    # --------------------------
    c1, c2, c3 = st.columns(3)
    c1.metric("High Risk", f"{(filtered['risk_bucket']=='high').sum():,}")
    c2.metric("Avg Churn Probability", f"{filtered['churn_probability'].mean():.3f}")
    c3.metric("Avg CLV Proxy", f"{filtered['clv_proxy'].mean():.2f}")

    st.divider()

    # --------------------------
    # Table view (top N)
    # --------------------------
    st.subheader("📌 Top Retention Priorities")
    top_n = st.slider("Show top N customers", min_value=20, max_value=500, value=100, step=20)

    cols_to_show = [
        "retention_priority",
        "churn_probability",
        "risk_bucket",
        "rfm_label",
        "rfm_score",
        "segment",
        "segment_label",
        "clv_proxy",
        "recommended_action",
    ]
    cols_to_show = [c for c in cols_to_show if c in filtered.columns]

    view = filtered.sort_values("retention_priority", ascending=False).head(top_n)
    st.dataframe(view[cols_to_show], use_container_width=True)

    st.divider()

    # --------------------------
    # Download
    # --------------------------
    csv_bytes = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download filtered retention recommendations (CSV)",
        data=csv_bytes,
        file_name="retention_recommendations_filtered.csv",
        mime="text/csv"
    )

except Exception as e:
    # ✅ This prevents black page — shows the real error
    st.error("❌ Retention Actions page crashed with an exception:")
    st.exception(e)
    st.stop()
