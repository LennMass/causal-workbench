"""
Pydantic models for data validation and serialization.

These schemas define the contract for everything flowing through the system:
requests, responses, and internal data structures.

- Pydantic BaseModel with type annotations
- Field validators and constraints
- Enum types for controlled vocabularies
- model_dump() / model_validate() for serialization
- How FastAPI uses these automatically
"""

from enum import Enum

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums for controlled choices
# ---------------------------------------------------------------------------

class EstimatorType(str, Enum):
    PLR = "plr"
    IRM = "irm"


class LearnerType(str, Enum):
    SKLEARN = "sklearn"
    TABPFN = "tabpfn"
    #PYTORCH = "pytorch" # currently not implemented


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class AnalysisRequest(BaseModel):
    """Configuration for a causal analysis run."""

    treatment_col: str = Field(
        default="treatment",
        description="Name of the binary treatment column",
    )
    outcome_col: str = Field(
        default="outcome",
        description="Name of the continuous outcome column",
    )
    description: str | None = Field(
        default=None,
        description="Optional description of the analysis"
    )
    estimator: EstimatorType = Field(
        default=EstimatorType.PLR,
        description="DoubleML estimator to use",
    )
    learner: LearnerType = Field(
        default=LearnerType.SKLEARN,
        description="ML backend for nuisance parameters",
    )
    confidence_level: float = Field(
        default=0.95,
        ge=0.5,
        le=0.99,
        description="Confidence level for the interval (0.5 to 0.99)",
    )


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class CausalResult(BaseModel):
    """Result of a causal estimation."""

    estimator: str
    learner: str
    coefficient: float = Field(description="Estimated average treatment effect")
    std_error: float = Field(description="Standard error of the estimate")
    ci_lower: float = Field(description="Lower bound of confidence interval")
    ci_upper: float = Field(description="Upper bound of confidence interval")
    p_value: float = Field(description="p-value for H0: coefficient = 0")
    confidence_level: float
    significant: bool = Field(description="Whether p < (1 - confidence_level)")


class DatasetInfo(BaseModel):
    """Summary of a loaded dataset."""

    n_rows: int
    n_columns: int
    columns: list[str]
    treatment_col: str
    outcome_col: str
    n_treated: int
    n_control: int


class ComparisonResult(BaseModel):
    """Side-by-side results from multiple learners."""

    results: list[CausalResult]
    dataset_info: DatasetInfo



# ---------------------------------------------------------------------------
# Phase 6 additions (async jobs)
# ---------------------------------------------------------------------------

# TODO: Add these in Phase 6
#
# class JobStatus(str, Enum):
#     PENDING = "pending"
#     RUNNING = "running"
#     COMPLETED = "completed"
#     FAILED = "failed"
#
# class JobResponse(BaseModel):
#     job_id: str
#     status: JobStatus
#     result: CausalResult | None = None
#     error: str | None = None


# ---------------------------------------------------------------------------
# Phase 7 additions (LLM agent)
# ---------------------------------------------------------------------------

# TODO: Add these in Phase 7
#
# class CausalInterpretation(BaseModel):
#     summary: str = Field(description="Plain-language summary")
#     significance_assessment: str
#     threats_to_validity: list[str]
#     suggested_robustness_checks: list[str]
