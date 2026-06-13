"""
Celery background tasks.

Move heavy computation off the request thread so the API responds instantly
with a job ID, and the user polls for results.


- Celery app configuration with Redis as broker
- Defining tasks with @app.task
- Task states: PENDING, STARTED, SUCCESS, FAILURE
- Retrieving results with AsyncResult
- How to run: celery -A src.api.tasks worker --loglevel=info

"""

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
torch.set_default_device("cpu")


from celery import Celery

celery_app = Celery(
    "causal_workbench",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/1",
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
)


@celery_app.task(bind=True)
def run_analysis_task(self, csv_path: str, config: dict) -> dict:
    """Background task for causal analysis."""
    from src.core.pipeline import run_pipeline
    from src.core.estimators import run_estimation

    self.update_state(state="RUNNING")

    text_cols = config.get("text_cols")
    if isinstance(text_cols, str):
        text_cols = [c.strip() for c in text_cols.split(",")]

    data = run_pipeline(
        csv_path,
        treatment_col=config["treatment_col"],
        outcome_col=config["outcome_col"],
        text_cols=text_cols,
        n_components=config.get("n_components", 10),
    )

    result = run_estimation(
        data,
        estimator=config["estimator"],
        learner=config["learner"],
        confidence_level=config.get("confidence_level", 0.95),
    )

    # Clean up the temp file
    import os
    try:
        os.unlink(csv_path)
    except OSError:
        pass

    return {
        "estimator": result.estimator,
        "learner": result.learner,
        "coefficient": result.coefficient,
        "std_error": result.std_error,
        "ci_lower": result.ci_lower,
        "ci_upper": result.ci_upper,
        "p_value": result.p_value,
        "confidence_level": result.confidence_level,
    }