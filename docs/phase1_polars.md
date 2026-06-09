# Phase 1 — Polars Data Pipeline

## Goal
Replace pandas with Polars for all data wrangling. Load, clean, and prepare
observational data for DoubleML.

## What to learn
- `pl.read_csv()` vs `pl.scan_csv()` (eager vs lazy evaluation)
- Expression API: `pl.col()`, `.with_columns()`, `.filter()`, `.select()`
- No index, no inplace mutations — everything returns a new DataFrame
- Null handling: `.fill_null()`, `.drop_nulls()`, `.is_null()`
- One-hot encoding: `.to_dummies()`
- Converting to numpy: `.to_numpy()`

## Files to work on
- `src/core/pipeline.py` — already implemented as a starting point

## Exercises
1. Run `python -m src.core.pipeline` and inspect the output
2. Try the lazy API: replace `pl.read_csv()` with `pl.scan_csv().collect()`
3. Add a step that logs summary statistics (mean, std, min, max per column)
4. Handle a dataset with missing values — create one by editing sample_data.csv
5. Run `pytest tests/test_pipeline.py -v`

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
