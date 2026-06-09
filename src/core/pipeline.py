"""
Data pipeline with Polars.

Loads raw CSV, validates structure, cleans data, and returns arrays
ready for DoubleML estimation.
"""

from pathlib import Path
from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass
class PreparedData:
    """Clean arrays ready for DoubleML."""

    X: np.ndarray          # confounders (n, p)
    D: np.ndarray          # treatment   (n,)
    Y: np.ndarray          # outcome     (n,)
    feature_names: list[str]
    n_obs: int
    n_features: int


def load_and_validate(
    path: str | Path,
    treatment_col: str,
    outcome_col: str,
) -> pl.DataFrame:
    """Load CSV with Polars and validate that required columns exist."""
    df = pl.scan_csv(path).collect()

    missing = {treatment_col, outcome_col} - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in data: {missing}")

    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Columns: {df.columns}")
    return df


def clean_data(df: pl.DataFrame) -> pl.DataFrame:
    """Handle missing values and encode categoricals."""

    # Fill numeric nulls with column median
    numeric_cols = [c for c in df.columns if df[c].dtype in (pl.Float64, pl.Int64, pl.Float32, pl.Int32)]
    for col in numeric_cols:
        median_val = df[col].median()
        df = df.with_columns(pl.col(col).fill_null(median_val))

    # One-hot encode string columns
    string_cols = [c for c in df.columns if df[c].dtype == pl.Utf8]
    if string_cols:
        df = df.to_dummies(columns=string_cols, drop_first=True)

    return df


def prepare_for_doubleml(
    df: pl.DataFrame,
    treatment_col: str,
    outcome_col: str,
) -> PreparedData:
    """Split into X, D, Y arrays for DoubleML."""

    feature_cols = [c for c in df.columns if c not in (treatment_col, outcome_col)]

    X = df.select(feature_cols).to_numpy()
    D = df[treatment_col].to_numpy().flatten()
    Y = df[outcome_col].to_numpy().flatten()

    return PreparedData(
        X=X,
        D=D,
        Y=Y,
        feature_names=feature_cols,
        n_obs=X.shape[0],
        n_features=X.shape[1],
    )

def log_summary(df: pl.DataFrame) -> None:
    """Print summary statistics for all numerical columns."""
    numeric_cols = [c for c in df.columns if df[c].dtype in (pl.Float64, pl.Int64, pl.Float32, pl.Int32)]
    summary = df.select(
        pl.col(numeric_cols).mean().name.prefix("mean_"),
        pl.col(numeric_cols).std().name.prefix("std_"),
        pl.col(numeric_cols).min().name.prefix("min_"), 
        pl.col(numeric_cols).max().name.prefix("max_"),
    )
    print("\n--- Summary Statistics for Numeric Vars ---")
    print(summary)


# from schemas.py
from src.core.schemas import DatasetSummary, ColumnStats

def summarize_dataset(df: pl.DataFrame) -> DatasetSummary:
    numeric_cols = [c for c in df.columns if df[c].dtype in (pl.Float64, pl.Int64, pl.Float32, pl.Int32)]
    stats = []
    for col in numeric_cols:
        stats.append(ColumnStats(
            name=col,
            mean=df[col].mean(),
            std=df[col].std(),
            min=df[col].min(),
            max=df[col].max(),
        ))
    return DatasetSummary(
        n_rows=df.shape[0],
        n_columns=df.shape[1],
        column_stats=stats,
    )




def run_pipeline(
    path: str | Path,
    treatment_col: str = "treatment",
    outcome_col: str = "outcome",
) -> PreparedData:
    """Full pipeline: load → clean → prepare."""
    df = load_and_validate(path, treatment_col, outcome_col)
    log_summary(df) # print summary stats for numerics
    print(f"Nulls before cleaning: {df.null_count()}")
    df = clean_data(df)
    print(f"Nulls before cleaning: {df.null_count()}")
    data = prepare_for_doubleml(df, treatment_col, outcome_col)
    print(f"Prepared: {data.n_obs} observations, {data.n_features} features")
    return data


# ---------------------------------------------------------------------------
# Quick test: run this file directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    data = run_pipeline("data/sample_data.csv")
    print(f"\nX shape: {data.X.shape}")
    print(f"D shape: {data.D.shape}")
    print(f"Y shape: {data.Y.shape}")
    print(f"Features: {data.feature_names}")
