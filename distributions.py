import torch
import torch.nn.functional as F
import networks
from networks import Linear
import math


class Distribution(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, representation):
        pass

class LRMVN_Distribution(torch.nn.Module):
    def __init__(self, batch_size, hidden_dim=64, number_of_mc_samples=100):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.number_of_mc_samples = number_of_mc_samples
        self.batch_size = batch_size

        self.d = 3
        self.h = 21
        self.w = 21	

        z_coords = torch.arange(self.d).float() - (self.d - 1) / 2
        y_coords = torch.arange(self.h).float() - (self.h - 1) / 2
        x_coords = torch.arange(self.w).float() - (self.w - 1) / 2

        z_coords = z_coords.view(self.d, 1, 1).expand(self.d, self.h, self.w)
        y_coords = y_coords.view(1, self.h, 1).expand(self.d, self.h, self.w)
        x_coords = x_coords.view(1, 1, self.w).expand(self.d, self.h, self.w)

        # Stack coordinates
        pixel_positions = torch.stack([x_coords, y_coords, z_coords], dim=-1)
        pixel_positions = pixel_positions.view(-1, 3)

        # Register buffer
        self.register_buffer("pixel_positions", pixel_positions)

        self.mvn_mean_mean_layer = Linear(
            in_features=self.hidden_dim,
            out_features=3,
        )
        self.mvn_mean_std_layer = Linear(
            in_features=self.hidden_dim,
            out_features=3,
        )
        self.mvn_cov_factor_mean_layer = Linear(
            in_features=self.hidden_dim,
            out_features=3,
        )
        self.mvn_cov_factor_std_layer = Linear(
            in_features=self.hidden_dim,
            out_features=3,
        )
        self.mvn_cov_diagonal_scale_layer = Linear(
            in_features=self.hidden_dim,
            out_features=3,
        )

    def forward(self, shoebox_representation, image_representation):# shoebox, dials, masks, metadata, counts):

        mvn_mean_mean = self.mvn_mean_mean_layer(shoebox_representation).unsqueeze(-1)
        mvn_mean_std = F.softplus(self.mvn_mean_std_layer(shoebox_representation)).unsqueeze(-1)

        mvn_cov_factor_mean = self.mvn_cov_factor_mean_layer(shoebox_representation + image_representation).unsqueeze(-1)
        mvn_cov_factor_std = F.softplus(self.mvn_cov_factor_std_layer(shoebox_representation + image_representation)).unsqueeze(-1)

        # for the inverse gammas
        mvn_cov_scale = F.softplus(self.mvn_cov_diagonal_scale_layer(shoebox_representation)).unsqueeze(-1)

        mvn_mean = torch.distributions.Normal(
            loc=mvn_mean_mean,
            scale=mvn_mean_std,
        )

        mvn_cov_factor = torch.distributions.Normal(
            loc=mvn_cov_factor_mean,
            scale=mvn_cov_factor_std,
        )

        mvn_cov_scale = torch.distributions.half_normal.HalfNormal(
            scale=mvn_cov_scale,
        )
        mvn_mean_samples = mvn_mean.rsample([self.number_of_mc_samples]).squeeze(-1).permute(1, 0, 2)
        mvn_cov_factor_samples = mvn_cov_factor.rsample([self.number_of_mc_samples]).permute(1, 0, 2, 3)
        diag_samples = (
            mvn_cov_scale.rsample([self.number_of_mc_samples]).squeeze(-1).permute(1, 0, 2) + 1e-6
        )
        profile_mvn_distribution = (
            torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal(
                loc=mvn_mean_samples.view(self.batch_size, self.number_of_mc_samples, 1, 3),
                cov_factor=mvn_cov_factor_samples.view(self.batch_size, self.number_of_mc_samples, 1, 3, 1),
                cov_diag=diag_samples.view(self.batch_size, self.number_of_mc_samples, 1, 3),
            )
        )

        return profile_mvn_distribution