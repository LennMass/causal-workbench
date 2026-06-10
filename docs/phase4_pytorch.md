# PyTorch Nuisance Learners

Builds a neural net that plugs into DoubleML as an alternative nuisance learner.

- `nn.Module`: define layers in `__init__`, computation in `forward()`
- Training loop: forward → loss → backward → optimizer step
- `skorch`: wraps PyTorch models in sklearn's fit/predict API
- Comparing neural net vs tree-based vs TabPFN nuisance models

## Files 
- `src/core/learners.py` 
- `src/core/estimators.py` 


## Resources
- [PyTorch](https://pytorch.org/)
- [TabPFN](https://github.com/PriorLabs/tabpfn)
