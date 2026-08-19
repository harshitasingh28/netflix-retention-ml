# src/modeling/churn_model.py

from __future__ import annotations
import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)

import joblib


DATA_PATH = "data/processed/netflix_segmented.csv"
MODEL_DIR = "models/"
REPORT_DIR = "reports/"
MODEL_OUT = os.path.join(MODEL_DIR, "churn_model.pkl")
METRICS_OUT = os.path.join(REPORT_DIR, "churn_metrics.csv")


def ensure_dirs():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)


def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def split_features_target(df: pd.DataFrame):
    if "churned" not in df.columns:
        raise ValueError("Target column 'churned' not found in dataset.")

    y = df["churned"].astype(int)
    X = df.drop(columns=["churned"], errors="ignore")

    return X, y


def build_preprocessor(X: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    """
    Detect numeric/categorical columns automatically.
    """
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    # Good practice: remove leakage-ish identifiers if present
    for leak_col in ["customer_id", "id", "user_id"]:
        if leak_col in numeric_cols:
            numeric_cols.remove(leak_col)
        if leak_col in categorical_cols:
            categorical_cols.remove(leak_col)

    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols)
        ],
        remainder="drop"
    )

    return preprocessor, numeric_cols, categorical_cols


def evaluate_model(model, X_test, y_test) -> dict:
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    metrics = {
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba),
        "f1": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
    }

    cm = confusion_matrix(y_test, y_pred)
    metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"] = cm.ravel()

    return metrics


def train_and_select_best(X_train, X_test, y_train, y_test, preprocessor):
    """
    Train 2 models and pick the best by PR-AUC (often best for churn imbalance).
    """
    candidates = {}

    # 1) Logistic Regression (strong baseline)
    logreg = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced"))
    ])
    logreg.fit(X_train, y_train)
    candidates["logreg"] = (logreg, evaluate_model(logreg, X_test, y_test))

    # 2) Random Forest (nonlinear, robust)
    rf = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=400,
            random_state=42,
            class_weight="balanced_subsample",
            max_depth=None,
            min_samples_split=5,
            min_samples_leaf=2,
            n_jobs=-1
        ))
    ])
    rf.fit(X_train, y_train)
    candidates["random_forest"] = (rf, evaluate_model(rf, X_test, y_test))

    # pick best by PR-AUC
    best_name = max(candidates.keys(), key=lambda k: candidates[k][1]["pr_auc"])
    best_model, best_metrics = candidates[best_name]

    return best_name, best_model, best_metrics, candidates


def main():
    ensure_dirs()

    df = load_data()
    X, y = split_features_target(df)

    # Split (stratify keeps churn ratio same)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    preprocessor, num_cols, cat_cols = build_preprocessor(X_train)

    best_name, best_model, best_metrics, all_models = train_and_select_best(
        X_train, X_test, y_train, y_test, preprocessor
    )

    # Save best model
    joblib.dump(best_model, MODEL_OUT)

    # Save metrics report
    rows = []
    for name, (model, metrics) in all_models.items():
        row = {"model": name}
        row.update(metrics)
        rows.append(row)
    report = pd.DataFrame(rows).sort_values(by="pr_auc", ascending=False)
    report.to_csv(METRICS_OUT, index=False)

    print("✅ Churn modeling complete")
    print("Saved best model to:", MODEL_OUT)
    print("Saved metrics report to:", METRICS_OUT)

    print("\nNumeric columns used:", num_cols)
    print("Categorical columns used:", cat_cols)

    print("\nBest model:", best_name)
    print("Best metrics:", {k: round(v, 4) if isinstance(v, float) else v for k, v in best_metrics.items()})


if __name__ == "__main__":
    main()
