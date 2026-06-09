# Phase 3 — FastAPI API Layer

## Goal
Serve your causal pipeline as a REST API with auto-generated docs.

## What to learn
- `FastAPI()` app, `@app.get()` / `@app.post()` decorators
- `UploadFile` for file uploads
- Pydantic models as `response_model` — auto-validation and docs
- Swagger UI at `/docs`, ReDoc at `/redoc`
- Error handling with `HTTPException`

## How to run
```bash
uvicorn src.api.main:app --reload
```
Then open http://localhost:8000/docs

## Files to work on
- `src/api/main.py` — already implemented

## Exercises
1. Start the server and try `/docs` in your browser
2. Upload `data/sample_data.csv` via the Swagger UI
3. Try the `/compare` endpoint
4. Add a `GET /info` endpoint that accepts a file upload and returns `DatasetInfo`
5. Add query parameter validation (e.g. reject files > 10MB)
6. Run `pytest tests/test_api.py -v`

## VS Code debugging
Press F5 and select "FastAPI: Run Server" from the launch config.
Set breakpoints in `main.py` — they'll hit when you make requests.
