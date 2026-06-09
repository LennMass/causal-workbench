"""
Phase 4 — PyTorch nuisance learners for DoubleML.

Build a small neural network that conforms to sklearn's fit/predict API
so it can be plugged into DoubleML as ml_l (outcome model) or ml_m
(propensity model).

LEARN:
  - torch.nn.Module for defining networks
  - Training loops: forward pass, loss, backward, optimizer step
  - skorch as a sklearn-compatible wrapper for PyTorch
  - Comparing NN nuisance learners vs RandomForest vs TabPFN

TODO (your exercises):
  1. Implement NuisanceNet below
  2. Wrap it with skorch (NeuralNetRegressor / NeuralNetClassifier)
  3. Register "pytorch" in estimators.py get_learners()
  4. Run /compare and see how results change
"""

# import torch
# import torch.nn as nn
# from skorch import NeuralNetRegressor, NeuralNetClassifier


# class NuisanceNet(nn.Module):
#     """Small feedforward net for nuisance estimation."""
#
#     def __init__(self, n_features: int, hidden_size: int = 64):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(n_features, hidden_size),
#             nn.ReLU(),
#             nn.BatchNorm1d(hidden_size),
#             nn.Linear(hidden_size, hidden_size),
#             nn.ReLU(),
#             nn.Linear(hidden_size, 1),
#         )
#
#     def forward(self, x):
#         return self.net(x)


# def get_pytorch_regressor(n_features: int):
#     """Sklearn-compatible regressor using skorch."""
#     return NeuralNetRegressor(
#         module=NuisanceNet,
#         module__n_features=n_features,
#         max_epochs=50,
#         lr=0.001,
#         batch_size=32,
#         verbose=0,
#     )


# def get_pytorch_classifier(n_features: int):
#     """Sklearn-compatible classifier using skorch."""
#     ...  # Similar, but use NeuralNetClassifier + sigmoid output
