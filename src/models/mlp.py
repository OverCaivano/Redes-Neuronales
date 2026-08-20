"""Perceptron multicapa base para regresion."""

import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=32, output_dim=1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features):
        return self.network(features)


RedNeuronalBase = MLP
