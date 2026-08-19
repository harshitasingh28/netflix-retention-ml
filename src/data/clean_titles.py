# src/data/clean_titles.py

from __future__ import annotations
import os
import re
import pandas as pd
import numpy as np


RAW_PATH = "data/raw/netflix_titles.csv"
OUT_PATH = "data/interim/titles_clean.csv"


# Maturity / rating mapping (simple numeric scale)
RATING_SCORE = {
    # kids/general
    "g": 1, "tv-y": 1, "tv-g": 1,
    # older kids
    "pg": 2, "tv-y7": 2, "tv-y7-fv": 2,
    # teen
    "pg-13": 3, "tv-pg": 3,
    # mature
    "tv-14": 4, "r": 4,
    # adult
    "tv-ma": 5, "nc-17": 5,
    # unknown-ish fallback
    "nr": 3, "ur": 3
}


def _standardize_colnames(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^\w]+", "_", regex=True)
        .str.replace(r"__+", "_", regex=True)
        .str.strip("_")
    )
    return df


def _clean_strings(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    obj_cols = df.select_dtypes(include=["object"]).columns
    for c in obj_cols:
        df[c] = df[c].astype(str).str.strip()
        df[c] = df[c].replace({"nan": np.nan, "none": np.nan, "null": np.nan, "": np.nan})
    return df


def _parse_date_added(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "date_added" in df.columns:
        df["date_added"] = pd.to_datetime(df["date_added"], errors="coerce")
    return df


def _parse_release_year(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "release_year" in df.columns:
        df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce")
    return df


def _clean_rating(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "rating" in df.columns:
        df["rating_clean"] = df["rating"].fillna("nr").astype(str).str.strip().str.lower()
        df["rating_score"] = df["rating_clean"].map(RATING_SCORE).fillna(3).astype(float)
    else:
        df["rating_clean"] = "nr"
        df["rating_score"] = 3.0
    return df


def _split_genres(df: pd.DataFrame) -> pd.DataFrame:
    """
    Netflix titles dataset stores genres in 'listed_in' like:
    "Dramas, International Movies, Thrillers"
    We convert that into:
    genres = ["dramas","international movies","thrillers"]
    and also keep genre_count.
    """
    df = df.copy()

    if "listed_in" in df.columns:
        df["genres"] = df["listed_in"].fillna("unknown").astype(str).apply(
            lambda x: [g.strip().lower() for g in x.split(",") if g.strip()]
        )
    else:
        df["genres"] = [[] for _ in range(len(df))]

    df["genre_count"] = df["genres"].apply(len)
    return df


def _parse_duration(df: pd.DataFrame) -> pd.DataFrame:
    """
    duration examples:
    - "90 min"
    - "1 Season"
    - "2 Seasons"
    We create:
      duration_minutes (float) for movies
      seasons (float) for TV shows
    """
    df = df.copy()

    if "duration" not in df.columns or "type" not in df.columns:
        df["duration_minutes"] = np.nan
        df["seasons"] = np.nan
        return df

    def parse_row(row):
        t = str(row["type"]).strip().lower()
        d = str(row["duration"]).strip().lower() if pd.notna(row["duration"]) else ""

        if t == "movie":
            m = re.search(r"(\d+)\s*min", d)
            return (float(m.group(1)) if m else np.nan, np.nan)

        # tv show
        m = re.search(r"(\d+)\s*season", d)
        return (np.nan, float(m.group(1)) if m else np.nan)

    parsed = df.apply(parse_row, axis=1)
    df["duration_minutes"] = [p[0] for p in parsed]
    df["seasons"] = [p[1] for p in parsed]

    return df


def _fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # common important text columns
    for c in ["type", "country", "director", "cast"]:
        if c in df.columns:
            df[c] = df[c].fillna("unknown").astype(str).str.strip().str.lower()

    # numeric fills (medians)
    for c in ["release_year", "duration_minutes", "seasons", "rating_score", "genre_count"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df[c] = df[c].fillna(df[c].median())

    return df


def _dedupe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicates conservatively.
    Netflix titles can have similar titles across different years/types.
    """
    df = df.copy()
    if all(c in df.columns for c in ["title", "type", "release_year"]):
        df = df.drop_duplicates(subset=["title", "type", "release_year"])
    else:
        df = df.drop_duplicates()
    return df


def clean_titles_data(raw_path: str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(raw_path)

    df = _standardize_colnames(df)
    df = _clean_strings(df)
    df = _parse_date_added(df)
    df = _parse_release_year(df)
    df = _clean_rating(df)
    df = _split_genres(df)
    df = _parse_duration(df)
    df = _fill_missing(df)
    df = _dedupe(df)

    return df


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    df_clean = clean_titles_data(RAW_PATH)
    df_clean.to_csv(OUT_PATH, index=False)

    print("✅ Cleaned titles saved to:", OUT_PATH)
    print("Shape:", df_clean.shape)

    # quick checks
    if "type" in df_clean.columns:
        print("Type counts:\n", df_clean["type"].value_counts())

    if "genre_count" in df_clean.columns:
        print("Avg genre_count:", df_clean["genre_count"].mean())


if __name__ == "__main__":
    main()
