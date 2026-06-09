# Phase 4 — PyTorch Nuisance Learners

## Goal
Build a neural net that plugs into DoubleML as an alternative nuisance learner.

## What to learn
- `nn.Module`: define layers in `__init__`, computation in `forward()`
- Training loop: forward → loss → backward → optimizer step
- `skorch`: wraps PyTorch models in sklearn's fit/predict API
- Comparing neural net vs tree-based vs TabPFN nuisance models

## Files to work on
- `src/core/learners.py` — uncomment and implement
- `src/core/estimators.py` — register `"pytorch"` in `get_learners()`

## Exercises
1. Implement `NuisanceNet` in `learners.py`
2. Wrap it with `skorch.NeuralNetRegressor`
3. Add `"pytorch": _get_pytorch_learners` to `estimators.py`
4. Hit `/compare` — now you'll see sklearn vs tabpfn vs pytorch side by side
5. Experiment: does network depth/width change the ATE estimate?

## Tips
- Start small: 2 hidden layers, 64 units each
- Use `BatchNorm1d` between layers for stability
- Set `verbose=0` in skorch to avoid noisy training output in the API
- TabPFN will likely win on small data — that's expected and instructive
