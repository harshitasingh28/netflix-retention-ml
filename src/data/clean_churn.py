# src/data/clean_churn.py

from __future__ import annotations
import os
import pandas as pd
import numpy as np


RAW_PATH = "data/raw/netflix_customer_churn.csv"
OUT_PATH = "data/interim/churn_clean.csv"


def _standardize_colnames(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^\w]+", "_", regex=True)  # spaces/symbols -> _
        .str.replace(r"__+", "_", regex=True)
        .str.strip("_")
    )
    return df


def _clean_strings(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    obj_cols = df.select_dtypes(include=["object"]).columns
    for c in obj_cols:
        df[c] = df[c].astype(str).str.strip()
        # normalize common "missing" tokens
        df[c] = df[c].replace({"nan": np.nan, "none": np.nan, "null": np.nan, "": np.nan})
    return df


def _coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    - numeric: fill with median
    - categorical: fill with "unknown"
    """
    df = df.copy()

    # numeric fill
    num_cols = df.select_dtypes(include=[np.number]).columns
    for c in num_cols:
        med = df[c].median()
        df[c] = df[c].fillna(med)

    # categorical fill
    cat_cols = df.select_dtypes(include=["object"]).columns
    for c in cat_cols:
        df[c] = df[c].fillna("unknown")
        df[c] = df[c].astype(str).str.strip().str.lower()

    return df


def _cap_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cap obvious numeric outliers (safe for real-world noisy data).
    We don't drop rows aggressively to avoid losing data.
    """
    df = df.copy()

    # Common churn dataset columns (only apply if they exist)
    if "age" in df.columns:
        df["age"] = df["age"].clip(lower=10, upper=100)

    if "tenure" in df.columns:
        df["tenure"] = df["tenure"].clip(lower=0)

    if "tenure_months" in df.columns:
        df["tenure_months"] = df["tenure_months"].clip(lower=0)

    if "monthly_charges" in df.columns:
        df["monthly_charges"] = df["monthly_charges"].clip(lower=0)

    if "monthly_fee" in df.columns:
        df["monthly_fee"] = df["monthly_fee"].clip(lower=0)

    if "total_charges" in df.columns:
        df["total_charges"] = df["total_charges"].clip(lower=0)

    if "watch_hours" in df.columns:
        df["watch_hours"] = df["watch_hours"].clip(lower=0)

    if "avg_watch_time_per_day" in df.columns:
        df["avg_watch_time_per_day"] = df["avg_watch_time_per_day"].clip(lower=0, upper=24)

    if "last_login_days" in df.columns:
        df["last_login_days"] = df["last_login_days"].clip(lower=0)

    if "number_of_profiles" in df.columns:
        df["number_of_profiles"] = df["number_of_profiles"].clip(lower=1, upper=10)

    return df


def _normalize_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize churn target to a single column named 'churned' as 0/1 if possible.
    Supports common variations like: churn, churned, exited, left, is_churned, etc.
    """
    df = df.copy()

    # If churned already exists, just coerce to 0/1
    if "churned" in df.columns:
        df["churned"] = pd.to_numeric(df["churned"], errors="coerce")
        df["churned"] = df["churned"].fillna(0).astype(int).clip(0, 1)
        return df

    # Try to find an existing churn-like column
    candidates = [c for c in df.columns if c in {"churn", "exited", "left", "is_churned", "is_churn", "target"}]
    if candidates:
        src = candidates[0]
        s = df[src]

        # handle yes/no strings
        if s.dtype == "object":
            s2 = s.astype(str).str.strip().str.lower()
            s2 = s2.replace({"yes": 1, "y": 1, "true": 1, "churn": 1, "1": 1,
                             "no": 0, "n": 0, "false": 0, "0": 0})
            df["churned"] = pd.to_numeric(s2, errors="coerce").fillna(0).astype(int).clip(0, 1)
        else:
            df["churned"] = pd.to_numeric(s, errors="coerce").fillna(0).astype(int).clip(0, 1)

    return df


def clean_churn_data(raw_path: str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(raw_path)
    df = _standardize_colnames(df)
    df = _clean_strings(df)

    # Coerce common numeric fields if present
    numeric_guess = [
        "age", "tenure", "tenure_months",
        "monthly_fee", "monthly_charges",
        "total_charges",
        "watch_hours", "avg_watch_time_per_day",
        "last_login_days",
        "number_of_profiles"
    ]
    df = _coerce_numeric(df, numeric_guess)

    df = _normalize_target(df)
    df = _cap_outliers(df)
    df = _fill_missing(df)

    # Drop exact duplicates
    df = df.drop_duplicates()

    return df


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    df_clean = clean_churn_data(RAW_PATH)
    df_clean.to_csv(OUT_PATH, index=False)

    print("✅ Cleaned churn saved to:", OUT_PATH)
    print("Shape:", df_clean.shape)
    print("Columns:", list(df_clean.columns)[:25], "..." if len(df_clean.columns) > 25 else "")

    # quick sanity checks
    if "churned" in df_clean.columns:
        print("Churned value counts:\n", df_clean["churned"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
