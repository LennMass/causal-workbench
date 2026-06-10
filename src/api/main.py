"""
FastAPI REST API.

Serves the causal pipeline over HTTP. Upload a CSV, configure the analysis,
get results. Auto-generated interactive docs at /docs.

Run with: uvicorn src.api.main:app --reload

ENDPOINTS:
  GET  /                → health check
  GET  /estimators      → list available estimators and learners
  POST /analyze         → upload CSV + config, get causal result
  POST /compare         → run all learners, compare results
"""

import io
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.core.schemas import (
    AnalysisRequest,
    CausalResult,
    ComparisonResult,
    DatasetInfo,
    EstimatorType,
    LearnerType,
)
from src.core.pipeline import run_pipeline, clean_data, prepare_for_doubleml
from src.core.estimators import run_estimation
from src.core.text_features import embed_text_column


app = FastAPI(
    title="Causal Inference Workbench",
    description="Upload observational data, estimate causal treatment effects.",
    version="0.1.0",
)

# Allow requests from any origin (useful for local dev / frontends)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helper: save upload to temp file and run pipeline; file size validation
# ---------------------------------------------------------------------------

async def _save_upload(file: UploadFile) -> Path:
    """Save uploaded file to a temp path and return it."""
    content = await file.read()
    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
    tmp.write(content)
    tmp.flush()
    return Path(tmp.name)

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

async def validate_file_size(file: UploadFile) -> None:
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File exceeds 10MB limit")
    await file.seek(0)  # reset to start so the file can be read again


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def health():
    return {"status": "ok", "service": "causal-workbench"}


@app.get("/estimators")
def list_estimators():
    """List available estimators and learners."""
    return {
        "estimators": [e.value for e in EstimatorType],
        "learners": [l.value for l in LearnerType],
    }

@app.post("/info", response_model=DatasetInfo)
async def dataset_info(
    file: UploadFile = File(...),
    treatment_col: str = "treatment",
    outcome_col: str = "outcome",
):
    """Upload a CSV and get a summary of the dataset."""

    await validate_file_size(file)
    path = await _save_upload(file)

    try:
        import polars as pl
        df = pl.read_csv(path)

        missing = {treatment_col, outcome_col} - set(df.columns)
        if missing:
            raise HTTPException(status_code=422, detail=f"Missing columns: {missing}")

        return DatasetInfo(
            n_rows=df.shape[0],
            n_columns=df.shape[1],
            columns=df.columns,
            treatment_col=treatment_col,
            outcome_col=outcome_col,
            n_treated=int(df[treatment_col].sum()),
            n_control=int(df.shape[0] - df[treatment_col].sum()),
        )
    finally:
        path.unlink(missing_ok=True)

@app.post("/analyze", response_model=CausalResult)
async def analyze(
    file: UploadFile = File(...),
    treatment_col: str = "treatment",
    outcome_col: str = "outcome",
    text_cols: str | None = None,  # comma-separated: "col1,col2"
    estimator: EstimatorType = EstimatorType.PLR,
    learner: LearnerType = LearnerType.SKLEARN,
    confidence_level: float = 0.95,
):
    path = await _save_upload(file)

    try:
        import polars as pl
        df = pl.read_csv(path)

        # Embed text columns if provided
        if text_cols:
            for col in text_cols.split(","):
                col = col.strip()
                if col in df.columns:
                    df = embed_text_column(df, col)

        df = clean_data(df)
        data = prepare_for_doubleml(df, treatment_col, outcome_col)
        result = run_estimation(data, estimator=estimator.value, learner=learner.value, confidence_level=confidence_level)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))
    finally:
        path.unlink(missing_ok=True)

    return CausalResult(
        estimator=result.estimator,
        learner=result.learner,
        coefficient=result.coefficient,
        std_error=result.std_error,
        ci_lower=result.ci_lower,
        ci_upper=result.ci_upper,
        p_value=result.p_value,
        confidence_level=result.confidence_level,
        significant=result.p_value < (1 - confidence_level),
    )


@app.post("/compare", response_model=ComparisonResult)
async def compare_learners(
    file: UploadFile = File(...),
    treatment_col: str = "treatment",
    outcome_col: str = "outcome",
    text_cols: str | None = None,
    estimator: EstimatorType = EstimatorType.PLR,
    confidence_level: float = 0.95,
):
    """Run the same estimator with all available learners and compare."""

    path = await _save_upload(file)

    try:
        text_col_list = [c.strip() for c in text_cols.split(",")] if text_cols else None
        data = run_pipeline(path, treatment_col, outcome_col, text_cols=text_col_list)

        results = []
        for learner_type in LearnerType:
            try:
                est = run_estimation(
                    data,
                    estimator=estimator.value,
                    learner=learner_type.value,
                    confidence_level=confidence_level,
                )
                results.append(CausalResult(
                    estimator=est.estimator,
                    learner=est.learner,
                    coefficient=est.coefficient,
                    std_error=est.std_error,
                    ci_lower=est.ci_lower,
                    ci_upper=est.ci_upper,
                    p_value=est.p_value,
                    confidence_level=est.confidence_level,
                    significant=est.p_value < (1 - confidence_level),
                ))
            except Exception as e:
                print(f"Learner {learner_type.value} failed: {e}")

        import polars as pl
        df = pl.read_csv(path)
        info = DatasetInfo(
            n_rows=df.shape[0],
            n_columns=df.shape[1],
            columns=df.columns,
            treatment_col=treatment_col,
            outcome_col=outcome_col,
            n_treated=int(df[treatment_col].sum()),
            n_control=int(df.shape[0] - df[treatment_col].sum()),
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))
    finally:
        path.unlink(missing_ok=True)

    return ComparisonResult(results=results, dataset_info=info)


# ---------------------------------------------------------------------------
# Phase 6: will add these async endpoints
# ---------------------------------------------------------------------------
# @app.post("/analyze/async")    → returns job_id
# @app.get("/results/{job_id}")  → returns status + result

# ---------------------------------------------------------------------------
# Phase 7: will add this explanation endpoint
# ---------------------------------------------------------------------------
# @app.post("/explain")          → returns CausalResult + LLM interpretation
