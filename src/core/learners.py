"""
XGBoost nuisance learners for DoubleML.

XGBoost gradient boosting as alternative nuisance learner.
sklearn-compatible out of the box — no wrapper needed.
"""

from xgboost import XGBRegressor, XGBClassifier


def get_xgboost_regressor():
    """Sklearn-compatible regressor for outcome model (ml_l)."""
    return XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        verbosity=0,
    )


def get_xgboost_classifier():
    """Sklearn-compatible classifier for propensity model (ml_m)."""
    return XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        verbosity=0,
        eval_metric="logloss",
    )