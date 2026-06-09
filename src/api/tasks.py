"""
Phase 6 — Celery background tasks.

Move heavy computation off the request thread so the API responds instantly
with a job ID, and the user polls for results.

LEARN:
  - Celery app configuration with Redis as broker
  - Defining tasks with @app.task
  - Task states: PENDING, STARTED, SUCCESS, FAILURE
  - Retrieving results with AsyncResult
  - How to run: celery -A src.api.tasks worker --loglevel=info

SETUP:
  1. Install Redis: brew install redis (Mac) / apt install redis (Linux) / docker run -d redis
  2. Start Redis: redis-server
  3. Start Celery worker: celery -A src.api.tasks worker --loglevel=info
  4. Start FastAPI: uvicorn src.api.main:app --reload

TODO (your exercises):
  1. Uncomment and implement the task below
  2. Add POST /analyze/async and GET /results/{job_id} to main.py
  3. Test: POST to /analyze/async, get job_id, poll /results/{job_id}
  4. Try submitting multiple jobs and watching them process in parallel
"""

# from celery import Celery
#
# celery_app = Celery(
#     "causal_workbench",
#     broker="redis://localhost:6379/0",
#     backend="redis://localhost:6379/1",
# )
#
# celery_app.conf.update(
#     task_serializer="json",
#     result_serializer="json",
#     accept_content=["json"],
#     task_track_started=True,
# )
#
#
# @celery_app.task(bind=True)
# def run_analysis_task(self, csv_path: str, config: dict) -> dict:
#     """Background task for causal analysis."""
#     from src.core.pipeline import run_pipeline
#     from src.core.estimators import run_estimation
#
#     self.update_state(state="RUNNING")
#
#     data = run_pipeline(
#         csv_path,
#         treatment_col=config["treatment_col"],
#         outcome_col=config["outcome_col"],
#     )
#     result = run_estimation(
#         data,
#         estimator=config["estimator"],
#         learner=config["learner"],
#         confidence_level=config.get("confidence_level", 0.95),
#     )
#
#     return {
#         "estimator": result.estimator,
#         "learner": result.learner,
#         "coefficient": result.coefficient,
#         "std_error": result.std_error,
#         "ci_lower": result.ci_lower,
#         "ci_upper": result.ci_upper,
#         "p_value": result.p_value,
#         "confidence_level": result.confidence_level,
#     }
