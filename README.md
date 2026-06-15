# Causal Inference Workbench: Automated ATE estimation with integrated LLM explainer

A self-hosted API for automated average treatment effect (ATE) estimation. We use DoubleML at the core for effect estimation and an integrated LLM explainer to interpret, summarize and assess results while pointing out possible threats to estimation validity. 




---

## Prerequisites

- Python 3.10+ (3.11 or 3.12 recommended)
- VS Code (or any other code editor) with the **Python** extension installed
- Git
- [Redis](https://redis.io/) — via Docker or local install
- TabPFN requires a one-time license acceptance to download model weights for local inference: [Login](https://ux.priorlabs.ai/account), [License]( https://ux.priorlabs.ai/account/licenses).
- LLM API Key for automatic LLM results explainer (i.e. via [Claude Platform](https://platform.claude.com/dashboard))


---

## VS Code Setup — Step by Step

### 1. Clone and open the project

```bash
git clone https://github.com/LennMass/causal-workbench causal-workbench
cd causal-workbench
code .
```

### 2. Create a virtual environment

Open the VS Code **terminal** (`Ctrl+`` ` or `Cmd+`` ` on Mac) and run:

```bash
python -m venv .venv
```

VS Code will usually detect this automatically and ask
**"We noticed a new environment…"** — click **Yes** to select it.

If it doesn't, press `Ctrl+Shift+P` → type `Python: Select Interpreter` →
choose the `.venv` one.

### 3. Activate the environment

The VS Code terminal should auto-activate. If not:

```bash
# Linux / Mac
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -e ".[all]"
```

If you are interested just in parts of this project, replace `all` with whichever sub-phase you're working on (see `pyproject.toml` fur further information.).

### 5. Set your LLM API key and run the API of this project

```bash
export ANTHROPIC_API_KEY="sk-ant-api03-dein-key" # for Claude API
uvicorn src.api.main:app --reload
```

Then open http://localhost:8000/docs in your browser for the interactive Swagger UI.

### 6. Run tests

```bash
pytest tests/ -v
```

---

## Main dependencies

Main dependencies installed with `pip install -e ".[all]"`: 

- Polars, DoubleML, TabPFN, scikit-learn
- Pydantic 
- PyTorch, skorch 
- FastAPI, Uvicorn
- sentence-transformers
- Celery, Redis
- pydantic-ai, anthropic

---

## Project Structure

```
causal-workbench/
├── src/
│   ├── __init__.py
│   ├── core/                  # Phases 1-2: data + causal logic
│   │   ├── __init__.py
│   │   ├── pipeline.py        # Phase 1: Polars data wrangling
│   │   ├── schemas.py         # Phase 2: Pydantic models
│   │   ├── estimators.py      # Phase 1: DoubleML + TabPFN runners
│   │   ├── learners.py        # Phase 4: PyTorch nuisance learners
│   │   └── text_features.py   # Phase 5: Transformer embeddings
│   ├── api/                   # Phase 3+: FastAPI app
│   │   ├── __init__.py
│   │   ├── main.py            # FastAPI app + routes
│   │   └── tasks.py           # Phase 6: Celery tasks
│   └── agents/                # Phase 7: LLM agent
│       ├── __init__.py
│       └── explainer.py       # Pydantic AI agent
├── data/
│   └── sample_data.csv        # Example dataset to get started
├── tests/
│   ├── __init__.py
│   ├── test_pipeline.py       # Phase 1 tests
│   ├── test_schemas.py        # Phase 2 tests
│   └── test_api.py            # Phase 3 tests
├── docs/
│   ├── phase1_polars.md
│   ├── phase2_pydantic.md
│   ├── phase3_fastapi.md
│   ├── phase4_pytorch.md
│   ├── phase5_transformers.md
│   ├── phase6_celery.md
│   └── phase7_pydantic_ai.md
├── pyproject.toml
├── .gitignore
└── README.md
```

---

## Overview

### Polars (data wrangling)
Load CSV, validate columns, handle missing values, encode
categoricals, return clean arrays for DoubleML. See `docs/phase1_polars.md`.

### Pydantic (schemas & validation)
`AnalysisRequest`, `CausalResult`, and `DatasetInfo` as strict typed
models. See `docs/phase2_pydantic.md`.

### FastAPI (API layer)
REST endpoints to upload data, configure analysis, get
results. Auto-generated docs at `/docs`. See `docs/phase3_fastapi.md`.

### Transformers (text confounders)
Embed text columns with sentence-transformers so they can enter the causal
model as numeric confounders. See `docs/phase5_transformers.md`.

### Celery (async background jobs)
Move heavy computation to background workers. Return job IDs, poll for
results. See `docs/phase6_celery.md`.

### Pydantic AI (LLM explanation agent)
An agent that interprets your causal results in plain language with
structured, validated output. See `docs/phase7_pydantic_ai.md`.

---

## Sample Workflow

```bash

# Activate venv and install dependencies
source .venv/bin/activate
pip install -e ".[all]"

# Export your API key and Start the API
export ANTHROPIC_API_KEY="sk-ant-api03-dein-key"
uvicorn src.api.main:app --reload

# From another terminal: Upload data and run analysis
curl -X POST http://localhost:8000/explain \
  -F "file=@data/sample_data.csv" \
  -F 'config={"treatment_col":"treatment","outcome_col":"outcome","estimator":"plr","learner":"tabpfn"}'
```

---

## Example Output

### Causal Estimation (`POST /analyze`)

Upload `sample_data.csv` (5,000 observations, job training → monthly income, true ATE = 350):

```json
{
  "estimator": "PLR",
  "learner": "sklearn",
  "coefficient": 349.87,
  "std_error": 44.12,
  "ci_lower": 263.39,
  "ci_upper": 436.35,
  "p_value": 0.000001,
  "confidence_level": 0.95,
  "significant": true
}
```

### Learner Comparison (`POST /compare`)

| Learner | ATE | Std Error | 95% CI | p-value |
|---------|-----|-----------|--------|---------|
| sklearn | 356.01 | 8.55 | [339.24, 372.77] | 0.000 |
| TabPFN | 353.78 | 6.39 | [341.246, 366.32] | 0.000 |

Both learners recover the true treatment effect (~350), with TabPFN
showing the tightest confidence interval.

### LLM Interpretation (`POST /explain`)

The Pydantic AI agent produces structured, validated output:

```json
{"result":
  {
    "estimator":"PLR",
    "learner":"tabpfn",
    "coefficient":353.78401649137635,
    "std_error":6.397033345551332,
    "ci_lower":341.24606152619396,
    "ci_upper":366.32197145655874,
    "p_value":0.0,
    "confidence_level":0.95,
    "significant":true
  },
"interpretation":
  {
    "summary":
    "Using a Partially Linear Regression (PLR) model with a TabPFN machine learning learner to control for 18 confounders across 5,000 observations, the analysis estimates that receiving the treatment causes an average increase of approximately 353.78 units in the outcome. This effect is estimated with high precision, and we can be very confident it is not due to random chance. However, this causal interpretation is only valid if all important confounders have been measured and included in the model.",

    "significance_assessment":
    "The result is statistically highly significant — the p-value is essentially zero, and the 95% confidence interval (341.25 to 366.32) is narrow and entirely positive, meaning there is virtually no chance this effect is a statistical fluke given the model assumptions. Practically, whether a ~354-unit increase in the outcome is meaningful depends entirely on the real-world scale of the outcome variable (e.g., dollars, test scores, health units). Statistical significance does NOT automatically imply practical importance — but here, the narrow confidence interval also suggests the estimate is precise enough to make business or policy decisions with reasonable confidence, pending subject-matter judgment on what a 354-unit change means in context.",

    "effect_size_context":
    "Without knowing the units or natural scale of the outcome variable, it is difficult to benchmark this effect in isolation. To put it in context, consider: if the outcome is measured in dollars (e.g., revenue), this would be a $353.78 average lift per treated unit. If the outcome has a known mean or standard deviation, computing the effect as a percentage change or in standard deviation units (Cohen's d) would clarify whether this is a small, medium, or large effect. The standard error of ~6.40 is less than 2% of the point estimate (353.78), indicating a very tight and well-identified estimate — roughly a ±1.8% margin of error around the ATE.",

    "threats_to_validity":
    [
    "Unobserved confounders (key assumption violation): The PLR estimator assumes all variables that jointly influence both treatment assignment and the outcome have been measured and included as the 18 features. If any important confounder is missing (e.g., user motivation, socioeconomic status, hidden selection criteria), the ATE estimate will be biased — potentially severely.","Selection bias in treatment assignment: If units self-selected into treatment (rather than being randomly assigned), and the reasons for selection are not fully captured in the 18 features, the estimate may reflect selection effects rather than a true causal effect.",
    "Model misspecification in nuisance estimation: Although TabPFN is a powerful ML learner, it may not perfectly model the relationship between confounders and treatment/outcome, especially if there are complex interactions or out-of-distribution patterns in the 18 features. Errors in nuisance estimation can leak bias into the ATE.",
    "Overlap/common support violation: If certain covariate profiles exist only in the treated or untreated group (lack of overlap), the PLR estimator may extrapolate into regions without empirical support, leading to unreliable estimates.","SUTVA (Stable Unit Treatment Value Assumption): The estimate assumes no interference between units (i.e., one unit's treatment does not affect another's outcome) and that there is only one version of the treatment. Violations — common in social or networked settings — would invalidate the causal interpretation.",
    "External validity: The ATE of ~353.78 applies to this specific sample of 5,000 observations. Whether this effect generalizes to different populations, time periods, or contexts is not guaranteed."
    ],

    "suggested_robustness_checks":
    [
    "Try alternative ML learners for nuisance parameters (e.g., XGBoost, LightGBM, Random Forest, Lasso) and compare ATE estimates — large discrepancies would signal sensitivity to nuisance model choice.",
    "Run a placebo/falsification test: randomly permute the treatment variable and re-estimate the ATE. It should be close to zero; a non-zero placebo effect suggests model or data issues.","Conduct a sensitivity analysis for unobserved confounding (e.g., Rosenbaum bounds or the E-value framework) to quantify how strong an unobserved confounder would need to be to explain away the estimated effect of 353.78.","Estimate the Average Treatment Effect on the Treated (ATT) and compare it to the ATE — a large difference may indicate heterogeneity or overlap problems between treated and control groups.",
    "Check overlap/propensity score distributions: plot the estimated propensity scores for treated vs. untreated units to verify common support. Trim or reweight observations in low-overlap regions and re-estimate.","Test for Heterogeneous Treatment Effects (HTE) using a CATE estimator (e.g., Causal Forest or X-Learner) to check whether the ATE masks important subgroup variation across the 18 features.",
    "Cross-fit with different random splits (the PLR uses cross-fitting to avoid overfitting) and verify that ATE estimates are stable across folds and different random seeds.","If panel or time-series data is available, consider a Difference-in-Differences (DiD) or event study design as an alternative identification strategy to cross-validate this result."
    ]
  }
}
```



---

## License

MIT
