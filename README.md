# Causal Inference Workbench

A self-hosted API for causal treatment effect estimation with DoubleML and TabPFN at the core.



---

## Prerequisites

- Python 3.10+ (3.11 or 3.12 recommended)
- VS Code (or any other code editor) with the **Python** extension installed
- Git
- (Phase 6 only) Redis — via Docker or local install
- TabPFN requires a one-time license acceptance to download model weights for local inference: [Login](https://ux.priorlabs.ai/account), [License]( https://ux.priorlabs.ai/account/licenses).


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

### 4. Install dependencies for the current phase

```bash
pip install -e ".[phase1]"
```

Replace `phase1` with whichever phase you're working on (see extras below).

### 5. Run the API (Phase 3+)

```bash
uvicorn src.api.main:app --reload
```

Then open http://localhost:8000/docs in your browser for the interactive Swagger UI.

### 6. Run tests

```bash
pytest tests/ -v
```

---

## Install Extras by Phase

Each phase adds new dependencies. Install only what you need:

| Command | What it adds |
|---|---|
| `pip install -e ".[phase1]"` | Polars, DoubleML, TabPFN, scikit-learn |
| `pip install -e ".[phase2]"` | + Pydantic (already included, but explicit) |
| `pip install -e ".[phase3]"` | + FastAPI, Uvicorn |
| `pip install -e ".[phase4]"` | + PyTorch, skorch |
| `pip install -e ".[phase5]"` | + sentence-transformers |
| `pip install -e ".[phase6]"` | + Celery, Redis |
| `pip install -e ".[phase7]"` | + pydantic-ai, anthropic |
| `pip install -e ".[all]"` | Everything |

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
│   └── sample_data.csv        # Toy dataset to get started
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

## Phase Overview

### Phase 1 — Polars (data wrangling)
Load CSV, validate columns, handle missing values, encode
categoricals, return clean arrays for DoubleML. See `docs/phase1_polars.md`.

### Phase 2 — Pydantic (schemas & validation)
Define `AnalysisRequest`, `CausalResult`, and `DatasetInfo` as strict typed
models. See `docs/phase2_pydantic.md`.

### Phase 3 — FastAPI (API layer)
Wrap the pipeline in REST endpoints. Upload data, configure analysis, get
results. Auto-generated docs at `/docs`. See `docs/phase3_fastapi.md`.

### Phase 4 — PyTorch (custom nuisance learner)
Build a small neural net that plugs into DoubleML as `ml_l` / `ml_m`.
Compare against TabPFN and sklearn. See `docs/phase4_pytorch.md`.

### Phase 5 — Transformers (text confounders)
Embed text columns with sentence-transformers so they can enter the causal
model as numeric confounders. See `docs/phase5_transformers.md`.

### Phase 6 — Celery (async background jobs)
Move heavy computation to background workers. Return job IDs, poll for
results. See `docs/phase6_celery.md`.

### Phase 7 — Pydantic AI (LLM explanation agent)
An agent that interprets your causal results in plain language with
structured, validated output. See `docs/phase7_pydantic_ai.md`.

---

## Sample Workflow (after Phase 3)

```bash
# Start the API
uvicorn src.api.main:app --reload

# Upload data and run analysis (from another terminal)
curl -X POST http://localhost:8000/analyze \
  -F "file=@data/sample_data.csv" \
  -F 'config={"treatment_col":"treatment","outcome_col":"outcome","estimator":"plr","learner":"tabpfn"}'
```

---

## License

MIT
