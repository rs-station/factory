import torch
import torch.nn.functional as F
import networks
from networks import Linear
import math

from networks import Linear, Constraint
from torch.distributions import HalfNormal

class DirichletProfile(torch.nn.Module):
    """
    Dirichlet profile model
    """

    def __init__(self, dmodel=None, input_shape=(3, 21, 21)):
        super().__init__()
        self.num_components = input_shape[0] * input_shape[1] * input_shape[2]
        if dmodel is not None:
            self.alpha_layer = Linear(dmodel, self.num_components)
        self.dmodel = dmodel
        self.eps = 1e-6

    def forward(self, alphas):
        if self.dmodel is not None:
            alphas = self.alpha_layer(alphas)
        alphas = F.softplus(alphas) + self.eps
        q_p = torch.distributions.Dirichlet(alphas)

        return q_p

class Distribution(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, representation):
        pass


class HalfNormalDistribution(torch.nn.Module):
    def __init__(
        self,
        dmodel,
        constraint=Constraint(),
        out_features=1,
    ):
        super().__init__()
        self.fc = Linear(
            in_features=dmodel,
            out_features=out_features,
        )
        self.constraint = constraint
        self.min_value = 1e-3
        self.max_value = 100.0

    def distribution(self, params):
        scale = self.constraint(params)
        # scale = torch.clamp(scale, min=self.min_value, max=self.max_value)
        return torch.distributions.half_normal.HalfNormal(scale)

    def forward(self, representation):
        params = self.fc(representation)
        norm = self.distribution(params)
        return norm

class LRMVN_Distribution(torch.nn.Module):

    mvn_mean_distribution: torch.distributions.Normal
    mvn_cov_factor_distribution: torch.distributions.Normal
    mvn_cov_scale_distribution: torch.distributions.HalfNormal
    profile_mvn_distribution: torch.distributions.LowRankMultivariateNormal
    prior_mvn_mean: torch.distributions.Normal
    prior_mvn_cov_factor: torch.distributions.Normal
    prior_mvn_cov_scale: torch.distributions.HalfNormal

    def __init__(self, hidden_dim=64, number_of_mc_samples=100):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.number_of_mc_samples = number_of_mc_samples

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

        # Initialize prior distributions
        self.prior_mvn_mean = torch.distributions.Normal(
            loc=torch.zeros(3),
            scale=torch.ones(3)
        )
        self.prior_mvn_cov_factor = torch.distributions.Normal(
            loc=torch.zeros(3),
            scale=torch.ones(3)
        )
        self.prior_mvn_cov_scale = torch.distributions.HalfNormal(
            scale=torch.ones(3)
        )

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
        

    def compute_kl_divergence(self, predicted_distribution, target_distribution):
            try: 
                return torch.distributions.kl.kl_divergence(predicted_distribution, target_distribution)
            except NotImplementedError:
                print("KL divergence not implemented for this distribution: use sampling method.")
                samples = predicted_distribution.rsample([100])
                log_q = predicted_distribution.log_prob(samples)
                log_p = target_distribution.log_prob(samples)
                return (log_q - log_p).mean(dim=0)
        
    def kl_divergence(self, *weights):
        """Compute the KL divergence between the predicted and prior distributions.

        Args:
            *weights: Weights for each component of the KL divergence (mean, cov_factor, cov_scale)

        Returns:
            torch.Tensor: The weighted sum of KL divergences
        """
        _kl_divergence = self.compute_kl_divergence(
            self.mvn_mean_distribution, self.prior_mvn_mean
        ).sum() * weights[0]    
        _kl_divergence += self.compute_kl_divergence(
            self.mvn_cov_factor_distribution, self.prior_mvn_cov_factor
        ).sum() * weights[1]
        _kl_divergence += self.compute_kl_divergence(
            self.mvn_cov_scale_distribution, self.prior_mvn_cov_scale
        ).sum() * weights[2]
        return _kl_divergence

    def compute_profile(self, *representations: torch.Tensor) -> torch.Tensor:
        """Compute the profile from the LRMVN distribution.

        Args:
            *representations: Variable number of representations. The first representation must be the shoebox_representation.
                            Additional representations are optional and will be combined with the first one.

        Returns:
            torch.Tensor: The computed profile normalized to sum to 1.
        """
        batch_size = representations[0].shape[0]
        profile_mvn_distribution = self.lrmvn_distribution_for_shoebox_profile(*representations)
        log_probs = profile_mvn_distribution.log_prob(
            self.lrmvn_distribution_for_shoebox_profile.pixel_positions.expand(batch_size, 1, 1323, 3)
        )

        log_probs_stable = log_probs - log_probs.max(dim=-1, keepdim=True)[0]

        profile = torch.exp(log_probs_stable)

        avg_profile = profile.mean(dim=1)
        avg_profile = avg_profile / (avg_profile.sum(dim=-1, keepdim=True) + 1e-10)

        return profile

    def forward(self, *representations: torch.Tensor) -> torch.distributions.LowRankMultivariateNormal:
        """Forward pass of the LRMVN distribution.

        Args:
            *representations: Variable number of representations. The first representation must be the shoebox_representation.
                            Additional representations are optional and will be combined with the first one.

        Returns:
            torch.distributions.LowRankMultivariateNormal: The profile MVN distribution.
        """
        self.batch_size = representations[0].shape[0]

        # Combine all representations if more than one is provided
        combined_representation = sum(representations) if len(representations) > 1 else representations[0]

        mvn_mean_mean = self.mvn_mean_mean_layer(representations[0]).unsqueeze(-1)
        mvn_mean_std = F.softplus(self.mvn_mean_std_layer(representations[0])).unsqueeze(-1)

        mvn_cov_factor_mean = self.mvn_cov_factor_mean_layer(combined_representation).unsqueeze(-1)
        mvn_cov_factor_std = F.softplus(self.mvn_cov_factor_std_layer(combined_representation)).unsqueeze(-1)

        # for the inverse gammas
        mvn_cov_scale_parameter = F.softplus(self.mvn_cov_diagonal_scale_layer(representations[0])).unsqueeze(-1)

        self.mvn_mean_distribution = torch.distributions.Normal(
            loc=mvn_mean_mean,
            scale=mvn_mean_std,
        )

        self.mvn_cov_factor_distribution = torch.distributions.Normal(
            loc=mvn_cov_factor_mean,
            scale=mvn_cov_factor_std,
        )

        self.mvn_cov_scale_distribution = torch.distributions.half_normal.HalfNormal(
            scale=mvn_cov_scale_parameter,
        )
        mvn_mean_samples = self.mvn_mean_distribution.rsample([self.number_of_mc_samples]).squeeze(-1).permute(1, 0, 2)
        mvn_cov_factor_samples = self.mvn_cov_factor_distribution.rsample([self.number_of_mc_samples]).permute(1, 0, 2, 3)
        diag_samples = (
            self.mvn_cov_scale_distribution.rsample([self.number_of_mc_samples]).squeeze(-1).permute(1, 0, 2) + 1e-6
        )
        self.profile_mvn_distribution = (
            torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal(
                loc=mvn_mean_samples.view(self.batch_size, self.number_of_mc_samples, 1, 3),
                cov_factor=mvn_cov_factor_samples.view(self.batch_size, self.number_of_mc_samples, 1, 3, 1),
                cov_diag=diag_samples.view(self.batch_size, self.number_of_mc_samples, 1, 3),
            )
        )

        return self.profile_mvn_distribution