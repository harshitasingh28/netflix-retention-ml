🎬 Netflix Customer Segmentation & Retention Intelligence System

An end-to-end, production-style data science project that analyzes customer behavior, predicts churn risk, and recommends actionable retention strategies using machine learning, RFM analysis, explainability, and an interactive dashboard.

This project is inspired by real-world subscription businesses like Netflix and focuses on turning ML outputs into business decisions, not just predictions.

📌 Problem Statement

Subscription-based platforms face significant revenue loss due to customer churn.
The key challenges are:

Identifying different types of customers

Predicting who is likely to churn

Understanding why they churn

Deciding who to retain first and how

This project solves all four problems in a single, unified system.

🧠 Solution Overview

The system combines classical business analytics with modern machine learning:

Customer Segmentation (Who are our users?)

RFM Analysis (How valuable and engaged are they?)

Churn Prediction (Who will leave?)

Explainability (SHAP) (Why will they leave?)

Retention Strategy Engine (What should we do?)

Interactive Dashboard (How do teams use this?)

🗂️ Datasets Used

Netflix Customer Churn Dataset

Customer demographics, usage behavior, subscription details, churn labels

Netflix Movies & TV Shows Dataset

Content metadata: genres, type (movie/TV), maturity ratings, release year

These datasets are combined to create content-aware customer features, similar to how real streaming platforms operate.


🏗️ Project Architecture

netflix-retention-ml/
│
├── data/
│   ├── raw/                # Original datasets
│   ├── interim/            # Cleaned content data
│   └── processed/          # ML-ready & retention outputs
│
├── src/
│   ├── features/           # Feature engineering + RFM
│   ├── modeling/
│   │   ├── segmentation.py
│   │   ├── churn_model.py
│   │   ├── explainability.py
│   │   └── retention_engine.py
│   └── app/                # Streamlit dashboard
│
├── models/                 # Trained ML models
├── reports/                # Metrics & SHAP outputs
├── requirements.txt
└── README.md

🔍 Feature Engineering Highlights
🔹 Behavioral Features

Engagement score

Watch intensity

Binge index

Inactivity score

Value score (usage vs spend)

🔹 Content-Aware Features

Matched primary genre

Genre diversity

Movie vs TV ratio

Average rating maturity

Average release year

🔹 RFM Analysis (Explicit)

R (Recency): last login activity

F (Frequency): watch hours

M (Monetary): monthly spend

Generated:

r_score, f_score, m_score

rfm_score (3–15)

rfm_label (Champions, Loyal, At Risk, Hibernating, etc.)

🧩 Customer Segmentation

Algorithm: KMeans

Inputs: engagement, spend, inactivity, content features

Output:

Numeric cluster IDs (for modeling)

Human-readable segment labels:

Loyal High-Value

High-Value At-Risk

Engaged Low-Spend

Low Engagement / Budget

🤖 Churn Prediction

Models trained:

Logistic Regression

Random Forest (selected as best)

Evaluation Metrics:

ROC-AUC

PR-AUC

Precision / Recall / F1

📈 Achieved very strong performance, with careful feature handling and no manual label leakage.

🔎 Explainability (SHAP)

To avoid black-box predictions, SHAP is used to:

Identify global churn drivers

Understand feature impact on churn

Validate business logic behind predictions

Top churn drivers typically include:

High inactivity

Low engagement

Poor value perception

Certain customer segments

🛟 Retention Strategy Engine

This is where the project becomes business-ready.

For each customer, the system computes:

Churn probability

Risk bucket (Low / Medium / High)

CLV proxy (value estimation)

Retention priority score

Recommended action

Example Actions

Personalized content recommendations

Re-engagement campaigns

Discounts or plan adjustments

Upsell / cross-sell opportunities

The output is a ranked list of customers to retain, not just predictions.

📊 Streamlit Dashboard

The interactive dashboard allows non-technical users to explore insights.

Pages

Segmentation Insights

Segment sizes

Churn rate by segment

RFM + behavioral profiles

Churn Prediction

Predict churn for any customer

View key features & RFM label

Retention Actions

Top priority customers

Filters by risk, segment, RFM

Downloadable action list

Content Insights

Genre distribution

Movie vs TV trends

Genre-level churn patterns

🧠 Key Learnings

Combining RFM + ML segmentation gives both interpretability and power

High model accuracy is useless without explainability and actions

Retention is a decision problem, not just a prediction task

Dashboards matter as much as models in real organizations

⚠️ Limitations & Future Work

Dataset does not contain real campaign/treatment logs
→ Uplift modeling intentionally not applied

Future improvements:

Time-based churn prediction

Cluster validation (Silhouette, DB index)

Per-customer SHAP explanations in dashboard

Retention uplift modeling with A/B data

🚀 How to Run Locally
pip install -r requirements.txt
streamlit run src/app/app.py

👤 Author

Mayank Kumar
Machine Learning & Data Science Enthusiast

⭐ Final Note

This project is designed to reflect how ML is actually used in subscription businesses — from raw data to actionable retention decisions.

If you like this project, feel free to ⭐ the repo or reach out!
