import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear


def weight_initializer(weight):
    fan_avg = 0.5 * (weight.shape[-1] + weight.shape[-2])
    std = math.sqrt(1.0 / fan_avg / 10.0)
    a = -2.0 * std
    b = 2.0 * std
    torch.nn.init.trunc_normal_(weight, 0.0, std, a, b)
    return weight


class Linear(torch.nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias=False):
        super().__init__(in_features, out_features, bias=bias)  # Set bias=False

    def reset_parameters(self) -> None:
        weight_initializer(self.weight)


class SimpleResidualLayer(nn.Module):
    def __init__(self, width, dropout_rate=0.0):
        super().__init__()
        # First layer
        self.fc1 = Linear(width, width)
        self.norm1 = nn.LayerNorm(width)

        # Second layer
        self.fc2 = Linear(width, width)
        self.norm2 = nn.LayerNorm(width)

        # Activation and dropout
        self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        residual = x

        # First layer
        out = self.norm1(x)
        # out = self.relu(out)
        out = self.fc1(out)
        # out = self.dropout(out)

        # Second layer
        # out = self.norm2(out)
        out = self.relu(out)
        # out = self.norm2(out)
        out = self.fc2(out)

        # Residual connection
        out = out + residual
        out = self.relu(out)

        return out


class SimpleMetadataEncoder(nn.Module):
    def __init__(self, feature_dim, depth=8, dropout=0.0, output_dims=None):
        super().__init__()
        layers = []
        hidden_dim = feature_dim * 2

        # Input layer
        layers.append(Linear(feature_dim, hidden_dim))
        # layers.append(nn.LayerNorm(hidden_dim))  #
        layers.append(nn.ReLU())

        # Residual blocks
        for _ in range(depth):
            layers.append(SimpleResidualLayer(hidden_dim, dropout_rate=dropout))

        # Output layer if needed
        if output_dims is not None:
            # layers.append(nn.LayerNorm(hidden_dim))
            # layers.append(nn.ReLU())
            layers.append(Linear(hidden_dim, output_dims))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # Process through the model
        x = self.model(x)
        return x


class BaseResidualLayer(nn.Module):
    def __init__(self, width, dropout_rate=0.0):
        super().__init__()
        # First layer
        self.fc1 = Linear(width, width)
        self.norm1 = nn.LayerNorm(width)

        # Second layer
        self.fc2 = Linear(width, width)
        self.norm2 = nn.LayerNorm(width)

        # Activation and dropout
        self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        residual = x

        # First layer
        out = self.norm1(x)
        out = self.relu(out)
        out = self.fc1(out)
        # out = self.dropout(out)

        # Second layer
        out = self.norm2(out)
        out = self.relu(out)
        # out = self.norm2(out)
        out = self.fc2(out)

        # Residual connection
        out = out + residual
        out = self.relu(out)

        return out


class BaseMetadataEncoder(nn.Module):
    def __init__(self, feature_dim=2, depth=10, dropout=0.0, output_dims=64):
        super().__init__()
        layers = []
        hidden_dim = feature_dim * 2

        # Input layer
        layers.append(Linear(feature_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))  #
        layers.append(nn.ReLU())

        # Residual blocks
        for _ in range(depth):
            layers.append(BaseResidualLayer(hidden_dim, dropout_rate=dropout))

        # Output layer if needed
        if output_dims is not None:
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(Linear(hidden_dim, output_dims))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # Check input for NaN values
        if torch.isnan(x).any():
            print("WARNING: NaN values in BaseMetadataEncoder input!")
            print("NaN count:", torch.isnan(x).sum().item())
            print(
                "Stats - min:",
                x.min().item(),
                "max:",
                x.max().item(),
                "mean:",
                x.mean().item(),
            )

        # Process through the model
        x = self.model(x)

        # Check output for NaN values
        if torch.isnan(x).any():
            print("WARNING: NaN values in BaseMetadataEncoder output!")
            print("NaN count:", torch.isnan(x).sum().item())
            print(
                "Stats - min:",
                x.min().item(),
                "max:",
                x.max().item(),
                "mean:",
                x.mean().item(),
            )

        return x
