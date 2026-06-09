"""Tests for Phase 2: Pydantic schemas."""

import pytest
from pydantic import ValidationError

from src.core.schemas import AnalysisRequest, CausalResult, EstimatorType, LearnerType


def test_analysis_request_defaults():
    req = AnalysisRequest()
    assert req.treatment_col == "treatment"
    assert req.outcome_col == "outcome"
    assert req.estimator == EstimatorType.PLR
    assert req.learner == LearnerType.SKLEARN
    assert req.confidence_level == 0.95


def test_analysis_request_custom():
    req = AnalysisRequest(
        treatment_col="D",
        outcome_col="Y",
        estimator="irm",
        learner="tabpfn",
        confidence_level=0.90,
    )
    assert req.estimator == EstimatorType.IRM
    assert req.learner == LearnerType.TABPFN


def test_analysis_request_invalid_confidence():
    with pytest.raises(ValidationError):
        AnalysisRequest(confidence_level=1.5)

    with pytest.raises(ValidationError):
        AnalysisRequest(confidence_level=0.1)


def test_causal_result_serialization():
    result = CausalResult(
        estimator="PLR",
        learner="sklearn",
        coefficient=25.3,
        std_error=2.1,
        ci_lower=21.2,
        ci_upper=29.4,
        p_value=0.001,
        confidence_level=0.95,
        significant=True,
    )
    d = result.model_dump()
    assert d["coefficient"] == 25.3
    assert d["significant"] is True

    # Round-trip
    restored = CausalResult.model_validate(d)
    assert restored.coefficient == result.coefficient
