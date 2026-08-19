# src/modeling/explainability.py

from __future__ import annotations
import os
import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt


DATA_PATH = "data/processed/netflix_segmented.csv"
MODEL_PATH = "models/churn_model.pkl"
OUT_DIR = "reports/shap/"

MAX_BACKGROUND = 300      # for performance
MAX_EXPLAIN = 500         # rows to explain


def ensure_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)


def load_data_and_model():
    df = pd.read_csv(DATA_PATH)
    model = joblib.load(MODEL_PATH)
    return df, model


def split_features_target(df: pd.DataFrame):
    y = df["churned"].astype(int)
    X = df.drop(columns=["churned"], errors="ignore")
    return X, y


def get_transformed_data(pipeline, X: pd.DataFrame):
    """
    Apply preprocessing part of pipeline and return:
    - transformed X
    - feature names after preprocessing
    """
    preprocessor = pipeline.named_steps["preprocess"]
    X_transformed = preprocessor.transform(X)

    # Numeric + one-hot feature names
    num_features = preprocessor.transformers_[0][2]
    cat_features = (
        preprocessor.transformers_[1][1]
        .named_steps["onehot"]
        .get_feature_names_out(preprocessor.transformers_[1][2])
    )

    feature_names = list(num_features) + list(cat_features)
    return X_transformed, feature_names


def compute_shap(pipeline, X: pd.DataFrame):
    """
    Compute SHAP values for the RandomForest model.
    Fixes:
      - sparse -> dense conversion
      - shap_values shape differences across SHAP versions
    """
    rf_model = pipeline.named_steps["model"]

    # sample rows to explain
    X_sample = X.sample(min(len(X), MAX_EXPLAIN), random_state=42)

    # transform using preprocess step
    X_transformed, feature_names = get_transformed_data(pipeline, X_sample)

    # Convert sparse matrix to dense array for SHAP plots
    try:
        X_dense = X_transformed.toarray()
    except Exception:
        X_dense = np.array(X_transformed)

    explainer = shap.TreeExplainer(rf_model)
    sv = explainer.shap_values(X_dense)

    # Handle different SHAP output formats safely
    # Case A: list (binary classification returns [class0, class1])
    if isinstance(sv, list):
        shap_values = sv[1] if len(sv) > 1 else sv[0]

    # Case B: 3D array (n_samples, n_features, n_classes)
    elif isinstance(sv, np.ndarray) and sv.ndim == 3:
        shap_values = sv[:, :, 1]  # churn class = 1

    # Case C: already 2D array
    else:
        shap_values = sv

    # Safety check
    if shap_values.shape[1] != X_dense.shape[1]:
        raise ValueError(
            f"SHAP shape mismatch: shap_values={shap_values.shape}, X={X_dense.shape}"
        )

    return shap_values, X_dense, feature_names



def save_global_plots(shap_values, X_dense, feature_names):
    # Beeswarm summary
    shap.summary_plot(
        shap_values,
        X_dense,
        feature_names=feature_names,
        show=False
    )
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "shap_summary.png"), dpi=300)
    plt.close()

    # Bar importance
    shap.summary_plot(
        shap_values,
        X_dense,
        feature_names=feature_names,
        plot_type="bar",
        show=False
    )
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "shap_feature_importance.png"), dpi=300)
    plt.close()



def save_top_drivers(shap_values, feature_names, top_n: int = 15):
    mean_abs = np.abs(shap_values).mean(axis=0)
    importance = (
        pd.DataFrame({
            "feature": feature_names,
            "mean_abs_shap": mean_abs
        })
        .sort_values("mean_abs_shap", ascending=False)
        .head(top_n)
    )

    importance.to_csv(
        os.path.join(OUT_DIR, "top_churn_drivers.csv"),
        index=False
    )


def main():
    ensure_dirs()

    df, pipeline = load_data_and_model()
    X, _ = split_features_target(df)

    shap_values, X_dense, feature_names = compute_shap(pipeline, X)

    save_global_plots(shap_values, X_dense, feature_names)
    save_top_drivers(shap_values, feature_names)

    print("✅ SHAP explainability completed")
    print("Saved to:", OUT_DIR)
    print("Files:")
    print(" - shap_summary.png")
    print(" - shap_feature_importance.png")
    print(" - top_churn_drivers.csv")


if __name__ == "__main__":
    main()
