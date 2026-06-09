"""Tests for Phase 1: Polars data pipeline."""

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src.core.pipeline import load_and_validate, clean_data, prepare_for_doubleml, run_pipeline

SAMPLE_DATA = Path("data/sample_data.csv")


def test_load_and_validate():
    df = load_and_validate(SAMPLE_DATA, "treatment", "outcome")
    assert isinstance(df, pl.DataFrame)
    assert df.shape[0] == 30
    assert "treatment" in df.columns
    assert "outcome" in df.columns


def test_load_missing_column():
    with pytest.raises(ValueError, match="Missing columns"):
        load_and_validate(SAMPLE_DATA, "nonexistent", "outcome")


def test_clean_data_encodes_categoricals():
    df = load_and_validate(SAMPLE_DATA, "treatment", "outcome")
    cleaned = clean_data(df)
    # "region" was a string column — should be one-hot encoded now
    assert "region" not in cleaned.columns
    # Should have region_* dummy columns
    region_cols = [c for c in cleaned.columns if c.startswith("region_")]
    assert len(region_cols) > 0


def test_prepare_for_doubleml_shapes():
    df = load_and_validate(SAMPLE_DATA, "treatment", "outcome")
    df = clean_data(df)
    data = prepare_for_doubleml(df, "treatment", "outcome")

    assert data.X.shape[0] == 30
    assert data.D.shape == (30,)
    assert data.Y.shape == (30,)
    assert data.n_obs == 30
    assert data.n_features == data.X.shape[1]


def test_full_pipeline():
    data = run_pipeline(SAMPLE_DATA)
    assert data.n_obs == 30
    assert isinstance(data.X, np.ndarray)
    assert isinstance(data.D, np.ndarray)
    assert isinstance(data.Y, np.ndarray)
