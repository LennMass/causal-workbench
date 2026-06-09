"""Tests for Phase 3: FastAPI endpoints.

Uses httpx + FastAPI TestClient for async testing.
Install: pip install httpx
"""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)

SAMPLE_FILE = "data/sample_data.csv"


def test_health():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_list_estimators():
    response = client.get("/estimators")
    data = response.json()
    assert "plr" in data["estimators"]
    assert "sklearn" in data["learners"]


def test_analyze():
    with open(SAMPLE_FILE, "rb") as f:
        response = client.post(
            "/analyze",
            files={"file": ("sample.csv", f, "text/csv")},
            data={
                "treatment_col": "treatment",
                "outcome_col": "outcome",
                "estimator": "plr",
                "learner": "sklearn",
            },
        )
    assert response.status_code == 200
    result = response.json()
    assert "coefficient" in result
    assert "p_value" in result
    assert "ci_lower" in result
    assert "ci_upper" in result


def test_analyze_invalid_column():
    with open(SAMPLE_FILE, "rb") as f:
        response = client.post(
            "/analyze",
            files={"file": ("sample.csv", f, "text/csv")},
            params={"treatment_col": "nonexistent"},
        )
    assert response.status_code == 422

def test_dataset_info():
    with open(SAMPLE_FILE, "rb") as f:
        response = client.post(
            "/info",
            files={"file": ("sample.csv", f, "text/csv")},
            data={"treatment_col": "treatment", "outcome_col": "outcome"},
        )
    assert response.status_code == 200
    data = response.json()
    assert data["n_rows"] == 30
    assert data["n_treated"] + data["n_control"] == 30
