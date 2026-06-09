# Phase 5 — Text Confounders with Transformers

## Goal
Embed free-text columns into numeric vectors so they can act as confounders
in the causal model.

## What to learn
- `SentenceTransformer` model loading and encoding
- How text features change causal estimates (omitted variable bias)
- Integration with the Polars pipeline

## Files to work on
- `src/core/text_features.py` — uncomment and implement
- `src/core/pipeline.py` — add optional text embedding step

## Exercises
1. Create a dataset with a text column (e.g. product descriptions)
2. Implement `embed_text_column()` in `text_features.py`
3. Run estimation with and without text features — compare ATE
4. Try different models: `all-MiniLM-L6-v2` (fast) vs `all-mpnet-base-v2` (better)
5. Add a `text_cols: list[str] | None` parameter to the API

## Model sizes (all run on CPU)
| Model | Dimensions | Speed |
|---|---|---|
| all-MiniLM-L6-v2 | 384 | ~14k sentences/sec |
| all-mpnet-base-v2 | 768 | ~2.8k sentences/sec |
