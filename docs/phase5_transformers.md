# Text Confounders with Transformers

Embed free-text columns into numeric vectors so they can act as confounders
in the causal model.

- `SentenceTransformer` model loading and encoding
- Text features may change causal estimates (omitted variable bias)
- Integration with the Polars pipeline
- Model: `all-MiniLM-L6-v2` (fast)

## Files
- `src/core/text_features.py` 
- `src/core/pipeline.py` 



## Resources
- [SentenceTransformer Models](https://sbert.net/)