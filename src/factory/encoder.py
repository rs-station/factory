# adapted from Luis Aldama @https://github.com/Hekstra-Lab/integrator.git
import torch


class CNN_3d_(torch.nn.Module):
    def __init__(self, Z=3, H=21, W=21, conv_channels=64, use_norm=True):
        super().__init__()
        self.Z = Z
        self.H = H
        self.W = W

        # Simple counts pathway only
        self.features = torch.nn.Sequential(
            # Initial 3D convolution
            torch.nn.Conv3d(
                in_channels=1,  # Just counts
                out_channels=conv_channels,
                kernel_size=3,
                padding=1,
            ),
            # Optional normalization
            torch.nn.BatchNorm3d(conv_channels) if use_norm else torch.nn.Identity(),
            torch.nn.ReLU(inplace=True),
            # Second conv block
            torch.nn.Conv3d(
                in_channels=conv_channels,
                out_channels=conv_channels,
                kernel_size=3,
                padding=1,
            ),
            torch.nn.BatchNorm3d(conv_channels) if use_norm else torch.nn.Identity(),
            torch.nn.ReLU(inplace=True),
        )

        # Global pooling
        self.pool = torch.nn.AdaptiveAvgPool3d((1, 1, 1))

    def reshape_input(self, x, mask=None):
        # Extract counts and reshape to 3D volume
        counts = x[..., -1]  # Shape: [batch_size, Z*H*W]
        counts = counts.view(-1, self.Z, self.H, self.W)
        counts = counts.unsqueeze(1)  # Add channel dim: [batch_size, 1, Z, H, W]

        if mask is not None:
            mask = mask.view(-1, 1, self.Z, self.H, self.W)
            counts = counts * mask

        return counts

    def forward(self, x, mask=None):
        # Reshape input
        x = self.reshape_input(x, mask)

        # Process through CNN
        x = self.features(x)

        # Global pooling and flatten
        x = self.pool(x)
        x = torch.flatten(x, 1)

        return x


class CNN_3d(torch.nn.Module):
    def __init__(self, out_dim=64):
        """
        Args:
            out_dim: Output dimension of the encoded representation.
        """
        # super(ShoeboxEncoder, self).__init__()
        # The input shape is assumed to be (B, 1, 3, 21, 21).
        # Convolution layer #1: Use (1, 3, 3) so that the depth (z) dimension stays unchanged.
        self.conv1 = nn.Conv3d(
            in_channels=1,
            out_channels=16,
            kernel_size=(1, 3, 3),
            stride=1,
            padding=(0, 1, 1),
        )
        self.norm1 = nn.GroupNorm(num_groups=4, num_channels=16)

        # Pooling applied only across height and width.
        self.pool = nn.MaxPool3d(
            kernel_size=(1, 2, 2), stride=(1, 2, 2), ceil_mode=True
        )

        # Convolution layer #2: Use a kernel that spans the entire depth (3) to collapse it.
        self.conv2 = nn.Conv3d(
            in_channels=16, out_channels=32, kernel_size=(3, 3, 3), stride=1, padding=0
        )
        self.norm2 = nn.GroupNorm(num_groups=4, num_channels=32)

        # After conv1, input shape is: (B, 16, 3, 21, 21);
        # after pooling: (B, 16, 3, approx. ceil(21/2)=11, ceil(21/2)=11);
        # after conv2: depth: 3-3+1=1; spatial dims: 11-3+1=9 (assuming exact arithmetic).
        flattened_size = 32 * 1 * 9 * 9  # = 2592
        self.fc = nn.Linear(flattened_size, out_dim)

    def forward(self, x, mask=None):
        # assuming input  is shape
        x = x[:, :, -1].reshape(x.shape[0], 1, 3, 21, 21)

        x = F.relu(self.norm1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.norm2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        rep = F.relu(self.fc(x))
        return rep


class MLP(torch.nn.Module):
    def __init__(self, width, depth, dropout=None, output_dims=None):
        super().__init__()
        layers = [ResidualLayer(width, dropout=dropout) for _ in range(depth)]
        if output_dims is not None:
            layers.append(torch.nn.Linear(width, output_dims))
        self.main = torch.nn.Sequential(*layers)

    def forward(self, data):
        # Check if the input has 2 or 3 dimensions
        if len(data.shape) == 3:
            batch_size, num_pixels, features = data.shape
            data = data.view(
                -1, features
            )  # Flatten to [batch_size * num_pixels, features]
        elif len(data.shape) == 2:
            batch_size, features = data.shape
            num_pixels = None  # No pixels in this case

        out = self.main(data)

        if num_pixels is not None:
            out = out.view(batch_size, num_pixels, -1)  # Reshape back if needed
        return out


class ResidualLayer(torch.nn.Module):
    def __init__(self, width, dropout=None):
        super().__init__()

        self.fc1 = torch.nn.Linear(width, width)
        self.fc2 = torch.nn.Linear(width, width)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout) if dropout else None

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.relu(out)
        if self.dropout:
            out = self.dropout(out)
        out = self.fc2(out)
        out += residual
        out = self.relu(out)
        return out


class MLPMetadataEncoder(torch.nn.Module):
    def __init__(
        self, depth=10, dmodel=64, feature_dim=2, output_dims=64, dropout=None
    ):
        super().__init__()
        self.linear = torch.nn.Linear(feature_dim, dmodel)
        # self.relu = torch.nn.ReLU(inplace=True)
        self.relu = torch.nn.ReLU()
        # self.batch_norm = nn.BatchNorm1d(dmodel)
        self.layer_norm = torch.nn.LayerNorm(dmodel)
        self.mlp_1 = MLP(dmodel, depth, dropout=dropout, output_dims=output_dims)

    def forward(self, shoebox_data):
        # shoebox_data shape: [batch_size, num_pixels, feature_dim]
        batch_size, features = shoebox_data.shape

        # Initial linear transformation
        out = self.linear(shoebox_data)
        out = self.relu(out)
        out = self.layer_norm(out)
        out = self.layer_norm(out)
        out = self.mlp_1(out)
        return out


def main():
    # Test the model
    model = CNN_3d()
    x = torch.randn(2, 3 * 21 * 21, 7)
    x = torch.clamp(x, 0, 1)

    out = model(x)

    print(out.shape)


if __name__ == "__main__":
    main()
