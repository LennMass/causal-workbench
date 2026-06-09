# Phase 1 — Polars Data Pipeline

Polars for all data wrangling. Load, clean, and prepare
observational data for DoubleML.

## Files
- `src/core/pipeline.py`

Install `pip install pytest` to run and test via `pytest tests/test_pipeline.py -v`.

## Key Polars vs Pandas differences
| Pandas | Polars |
|---|---|
| `df['col']` | `df['col']` or `df.select('col')` |
| `df.fillna(0)` | `df.with_columns(pl.col('x').fill_null(0))` |
| `df.groupby('x').mean()` | `df.group_by('x').agg(pl.col('y').mean())` |
| `pd.get_dummies(df)` | `df.to_dummies()` |
| Mutates in place | Always returns new DataFrame |

## Resources
- [Polars User Guide](https://docs.pola.rs/)
- [Polars vs Pandas cheat sheet](https://docs.pola.rs/user-guide/migration/pandas/)
