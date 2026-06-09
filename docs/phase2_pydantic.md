# Phase 2 — Pydantic Schemas

## Goal
Define strict typed data contracts for every piece of data flowing through the system.

## What to learn
- `BaseModel` with type annotations
- `Field()` for defaults, constraints (`ge`, `le`), and descriptions
- `str` Enums for controlled vocabularies
- `.model_dump()` → dict, `.model_validate()` → model from dict
- Automatic validation: wrong types raise `ValidationError`

## Files to work on
- `src/core/schemas.py` — already implemented

## Exercises
1. Run `pytest tests/test_schemas.py -v`
2. Try creating an `AnalysisRequest(confidence_level=2.0)` in a Python shell — see the validation error
3. Add a new field `description: str | None = None` to `AnalysisRequest`
4. Add a `DatasetSummary` model with field-level statistics (mean, std per column)
5. Try `model_dump_json()` for direct JSON serialization

## Why this matters
In Phase 3, FastAPI will use these exact models as request/response types.
You write the schema once, and get validation + docs + serialization for free.
