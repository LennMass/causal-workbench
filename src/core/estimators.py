"""
Phase 1 — Causal estimators using DoubleML.

Wraps DoubleML's PLR and IRM models with pluggable ML backends:
  - sklearn (Random Forest) — fast baseline
  - TabPFN — zero-shot, great for small data

LEARN:
  - DoubleML data backend (DoubleMLData)
  - Partially Linear Regression (PLR) vs Interactive Regression (IRM)
  - Swapping nuisance learners and comparing results
"""

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

import doubleml as dml

from src.core.pipeline import PreparedData


@dataclass
class CausalEstimate:
    """Raw result from a DoubleML estimation."""

    estimator: str
    learner: str
    coefficient: float
    std_error: float
    ci_lower: float
    ci_upper: float
    p_value: float
    confidence_level: float


def _get_sklearn_learners() -> tuple:
    """Default sklearn learners for nuisance parameters."""
    ml_l = RandomForestRegressor(n_estimators=100, random_state=42)
    ml_m = RandomForestClassifier(n_estimators=100, random_state=42)
    return ml_l, ml_m


def _get_tabpfn_learners() -> tuple:
    """TabPFN learners — works best with n < 1000, p < 100."""
    # TODO: Phase 1 exercise — import and configure TabPFN here.
    #
    # from tabpfn import TabPFNRegressor, TabPFNClassifier
    # ml_l = TabPFNRegressor()
    # ml_m = TabPFNClassifier()
    # return ml_l, ml_m
    #
    # For now, fall back to sklearn:
    print("⚠ TabPFN not configured yet — using sklearn as fallback")
    return _get_sklearn_learners()


def get_learners(name: str = "sklearn") -> tuple:
    """Factory for nuisance learner pairs."""
    factories = {
        "sklearn": _get_sklearn_learners,
        "tabpfn": _get_tabpfn_learners,
        # Phase 4 will add: "pytorch": _get_pytorch_learners
    }
    if name not in factories:
        raise ValueError(f"Unknown learner '{name}'. Choose from: {list(factories.keys())}")
    return factories[name]()


def estimate_plr(
    data: PreparedData,
    learner: str = "sklearn",
    confidence_level: float = 0.95,
) -> CausalEstimate:
    """Partially Linear Regression: Y = D*theta + g(X) + U."""

    dml_data = dml.DoubleMLData.from_arrays(x=data.X, y=data.Y, d=data.D)
    ml_l, ml_m = get_learners(learner)

    model = dml.DoubleMLPLR(dml_data, ml_l=ml_l, ml_m=ml_m)
    model.fit()

    ci = model.confint(level=confidence_level)

    return CausalEstimate(
        estimator="PLR",
        learner=learner,
        coefficient=model.coef[0],
        std_error=model.se[0],
        ci_lower=ci.iloc[0, 0],
        ci_upper=ci.iloc[0, 1],
        p_value=model.pval[0],
        confidence_level=confidence_level,
    )


def estimate_irm(
    data: PreparedData,
    learner: str = "sklearn",
    confidence_level: float = 0.95,
) -> CausalEstimate:
    """Interactive Regression Model — for heterogeneous effects."""

    dml_data = dml.DoubleMLData.from_arrays(x=data.X, y=data.Y, d=data.D)
    ml_l, ml_m = get_learners(learner)
    # IRM needs ml_g (outcome model) and ml_m (propensity)
    ml_g = ml_l  # reuse the regressor for g

    model = dml.DoubleMLIRM(dml_data, ml_g=ml_g, ml_m=ml_m)
    model.fit()

    ci = model.confint(level=confidence_level)

    return CausalEstimate(
        estimator="IRM",
        learner=learner,
        coefficient=model.coef[0],
        std_error=model.se[0],
        ci_lower=ci.iloc[0, 0],
        ci_upper=ci.iloc[0, 1],
        p_value=model.pval[0],
        confidence_level=confidence_level,
    )


def run_estimation(
    data: PreparedData,
    estimator: str = "plr",
    learner: str = "sklearn",
    confidence_level: float = 0.95,
) -> CausalEstimate:
    """Dispatch to the right estimator."""
    estimators = {
        "plr": estimate_plr,
        "irm": estimate_irm,
    }
    if estimator not in estimators:
        raise ValueError(f"Unknown estimator '{estimator}'. Choose from: {list(estimators.keys())}")

    return estimators[estimator](data, learner=learner, confidence_level=confidence_level)


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.core.pipeline import run_pipeline

    data = run_pipeline("data/sample_data.csv")
    result = run_estimation(data, estimator="plr", learner="sklearn")

    print(f"\n{'='*50}")
    print(f"Estimator:  {result.estimator}")
    print(f"Learner:    {result.learner}")
    print(f"ATE:        {result.coefficient:.4f}")
    print(f"Std Error:  {result.std_error:.4f}")
    print(f"95% CI:     [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")
    print(f"p-value:    {result.p_value:.6f}")
