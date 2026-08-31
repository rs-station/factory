import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear


class ShoeboxEncoder(nn.Module):
    def __init__(
        self,
        input_shape=(3, 21, 21),  # (D, H, W)
        in_channels=1,
        out_dim=64,
        conv1_out_channels=16,
        conv1_kernel=(1, 3, 3),
        conv1_padding=(0, 1, 1),
        norm1_num_groups=4,
        pool_kernel=(1, 2, 2),
        pool_stride=(1, 2, 2),
        conv2_out_channels=32,
        conv2_kernel=(3, 3, 3),
        conv2_padding=(0, 0, 0),
        norm2_num_groups=4,
    ):
        super().__init__()

        self.conv1 = nn.Conv3d(
            in_channels=in_channels,
            out_channels=conv1_out_channels,
            kernel_size=conv1_kernel,
            padding=conv1_padding,
        )
        self.norm1 = nn.GroupNorm(norm1_num_groups, conv1_out_channels)
        self.pool = nn.MaxPool3d(
            kernel_size=pool_kernel, stride=pool_stride, ceil_mode=True
        )
        self.conv2 = nn.Conv3d(
            in_channels=conv1_out_channels,
            out_channels=conv2_out_channels,
            kernel_size=conv2_kernel,
            padding=conv2_padding,
        )
        self.norm2 = nn.GroupNorm(norm2_num_groups, conv2_out_channels)

        # Dynamically calculate flattened size
        self.flattened_size = self._infer_flattened_size(
            input_shape=input_shape, in_channels=in_channels
        )
        self.fc = nn.Linear(self.flattened_size, out_dim)

    def _infer_flattened_size(self, input_shape, in_channels):
        # input_shape: (H, W, D)
        with torch.no_grad():
            dummy = torch.zeros(
                1, in_channels, input_shape[0], input_shape[1], input_shape[2]
            )  # (B, C, D, H, W)
            x = self.pool(F.relu(self.norm1(self.conv1(dummy))))
            x = F.relu(self.norm2(self.conv2(x)))
            return x.numel()

    def forward(self, x, mask=None):
        x = F.relu(self.norm1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.norm2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        return F.relu(self.fc(x))


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


class ResidualLayer(nn.Module):
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


class MLPMetadataEncoder(nn.Module):
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
            layers.append(ResidualLayer(hidden_dim, dropout_rate=dropout))

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
