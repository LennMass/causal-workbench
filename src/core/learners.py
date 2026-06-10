"""
PyTorch nuisance learners for DoubleML.

Builds a small neural network that conforms to sklearn's fit/predict API
so it can be plugged into DoubleML as ml_l (outcome model) or ml_m
(propensity model).

- torch.nn.Module for defining networks
- Training loops: forward pass, loss, backward, optimizer step
- skorch as a sklearn-compatible wrapper for PyTorch
- Comparing NN nuisance learners vs RandomForest vs TabPFN


"""

import torch
import torch.nn as nn
from skorch import NeuralNetRegressor, NeuralNetClassifier


class RegressionNet(nn.Module):
    """Net for outcome model (ml_l) — outputs a single value."""

    def __init__(self, n_features: int, hidden_size: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        return self.net(x.float()).squeeze(-1)


class ClassificationNet(nn.Module):
    """Net for propensity model (ml_m) — outputs 2-class log probabilities."""

    def __init__(self, n_features: int, hidden_size: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2),
            nn.LogSoftmax(dim=-1),
        )

    def forward(self, x):
        return self.net(x.float())


def get_pytorch_regressor(n_features: int):
    return NeuralNetRegressor(
        module=RegressionNet,
        module__n_features=n_features,
        module__hidden_size=64,
        max_epochs=50,
        lr=0.001,
        batch_size=32,
        verbose=0,
        train_split=None,
    )


def get_pytorch_classifier(n_features: int):
    return NeuralNetClassifier(
        module=ClassificationNet,
        module__n_features=n_features,
        module__hidden_size=64,
        max_epochs=50,
        lr=0.001,
        batch_size=32,
        verbose=0,
        train_split=None,
    )