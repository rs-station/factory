import torch
import torch.nn.functional as F
import networks
from networks import Linear
import math

from networks import Linear, Constraint
from torch.distributions import HalfNormal, Exponential

from abc import ABC, abstractmethod


class ProfileDistribution(ABC, torch.nn.Module):
    """Base class for profile distributions that can compute KL divergence and generate profiles."""

    def __init__(self):
        super().__init__()
    
    def compute_kl_divergence(self, predicted_distribution, target_distribution):
        try: 
            return torch.distributions.kl.kl_divergence(predicted_distribution, target_distribution)
        except NotImplementedError:
            print("KL divergence not implemented for this distribution: use sampling method.")
            samples = predicted_distribution.rsample([100])
            log_q = predicted_distribution.log_prob(samples)
            log_p = target_distribution.log_prob(samples)
            return (log_q - log_p).mean(dim=0)

    @abstractmethod
    def _update_prior_distributions(self):
        """Update prior distributions to use current device."""
        pass
    
    @abstractmethod
    def kl_divergence_weighted(self, weight):
        """Compute KL divergence between predicted and prior distributions.
        
        Args:
            weight: Weight factor for the KL divergence
        """
        pass
        
    @abstractmethod
    def compute_profile(self, *representations: torch.Tensor) -> torch.Tensor:
        """Compute the profile from the distribution.
        
        Args:
            *representations: Variable number of representations to combine
            
        Returns:
            torch.Tensor: The computed profile normalized to sum to 1
        """
        pass
        
    @abstractmethod
    def forward(self, *representations):
        """Forward pass of the distribution.
        
        Args:
            *representations: Variable number of representations to combine
        """
        pass


def compute_kl_divergence(predicted_distribution, target_distribution):
            try: 
                # Create proper copies of the distributions
                # pred_copy = type(predicted_distribution)(
                #     loc=predicted_distribution.loc.clone(),
                #     scale=predicted_distribution.scale.clone()
                # )
                # target_copy = type(target_distribution)(
                #     loc=target_distribution.loc.clone(),
                #     scale=target_distribution.scale.clone()
                # )
                return torch.distributions.kl.kl_divergence(predicted_distribution, target_distribution)
            except NotImplementedError:
                print("KL divergence not implemented for this distribution: use sampling method.")
                samples = predicted_distribution.rsample([100])
                log_q = predicted_distribution.log_prob(samples)
                log_p = target_distribution.log_prob(samples)
                return (log_q - log_p).mean(dim=0)

class DirichletProfile(ProfileDistribution):
    """
    Dirichlet profile model
    """

    def __init__(self, concentration_vector, dmodel, input_shape=(3, 21, 21)):
        super().__init__()
        self.num_components = input_shape[0] * input_shape[1] * input_shape[2]
        if dmodel is not None:
            self.alpha_layer = Linear(dmodel, self.num_components)
            # Make profile parameters non-trainable
            # for param in self.alpha_layer.parameters():
            #     param.requires_grad = False
        self.dmodel = dmodel
        self.eps = 1e-6
        concentration_vector[concentration_vector>torch.quantile(concentration_vector, 0.99)] *= 40
        concentration_vector /= concentration_vector.max()
        self.register_buffer("concentration", concentration_vector) #.reshape((21,21,3)))
        self.q_p = None
    
    def _update_prior_distributions(self):
        """Update prior distributions to use current device."""
        self.dirichlet_prior = torch.distributions.Dirichlet(self.concentration)
    
    def kl_divergence_weighted(self, weight):
        self._update_prior_distributions()

        print("shape inot kl profile", self.q_p.sample().shape, self.dirichlet_prior.sample().shape)
        return self.compute_kl_divergence(self.q_p, self.dirichlet_prior)* weight
    
    def compute_profile(self, *representations: torch.Tensor) -> torch.Tensor:
        dirichlet_profile = self.forward(*representations)
        return dirichlet_profile
        # samples = dirichlet_profile.rsample()
        # avg_profile = samples.mean(dim=0)
        # avg_profile = avg_profile / (avg_profile.sum() + self.eps)
        # return avg_profile

    def forward(self, *representations):
        alphas = sum(representations) if len(representations) > 1 else representations[0]
        print("alpha shape", alphas.shape)
        
        # Check for NaN values in input representations
        if torch.isnan(alphas).any():
            print("WARNING: NaN values detected in input representations!")
            print("NaN count:", torch.isnan(alphas).sum().item())
            print("Input stats - min:", alphas.min().item(), "max:", alphas.max().item(), "mean:", alphas.mean().item())
        
        if self.dmodel is not None:
            alphas = self.alpha_layer(alphas)
            # Check for NaN values after linear layer
            if torch.isnan(alphas).any():
                print("WARNING: NaN values detected after alpha_layer!")
                print("NaN count:", torch.isnan(alphas).sum().item())
                print("Alpha layer output stats - min:", alphas.min().item(), "max:", alphas.max().item(), "mean:", alphas.mean().item())
        
        alphas = F.softplus(alphas) + self.eps
        print("profile alphas shape", alphas.shape)
        
        # Check for NaN values after softplus
        if torch.isnan(alphas).any():
            print("WARNING: NaN values detected after softplus!")
            print("NaN count:", torch.isnan(alphas).sum().item())
            print("Softplus output stats - min:", alphas.min().item(), "max:", alphas.max().item(), "mean:", alphas.mean().item())
            
            # Replace NaN values with a safe default
        
        # Ensure all values are positive and finite
        
        self.q_p = torch.distributions.Dirichlet(alphas)
        print("profile q_p shape", self.q_p.rsample().shape)
        return self.q_p

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

class ExponentialDistribution(torch.nn.Module):
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
        return torch.distributions.Exponential(scale)

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

    def __init__(self, dmodel=64, number_of_mc_samples=100):
        super().__init__()
        self.hidden_dim = dmodel
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
        self.register_buffer('prior_mvn_mean_loc', torch.tensor(0.0))
        self.register_buffer('prior_mvn_mean_scale', torch.tensor(5.0))
        self.register_buffer('prior_mvn_cov_factor_loc', torch.tensor(0.0))
        self.register_buffer('prior_mvn_cov_factor_scale', torch.tensor(0.01))
        self.register_buffer('prior_mvn_cov_scale_scale', torch.tensor(0.01))

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

    def _update_prior_distributions(self):
        """Update prior distributions to use current device."""
        self.prior_mvn_mean = torch.distributions.Normal(
            loc=self.prior_mvn_mean_loc,
            scale=self.prior_mvn_mean_scale
        )
        self.prior_mvn_cov_factor = torch.distributions.Normal(
            loc=self.prior_mvn_cov_factor_loc,
            scale=self.prior_mvn_cov_factor_scale
        )
        self.prior_mvn_cov_scale = torch.distributions.HalfNormal(
            scale=self.prior_mvn_cov_scale_scale
        )
        
    def kl_divergence_weighted(self, weights):
        """Compute the KL divergence between the predicted and prior distributions.

        Args:
            weights: A tuple or list of weights for each component of the KL divergence (mean, cov_factor, cov_scale)

        Returns:
            torch.Tensor: The weighted sum of KL divergences
        """
        _kl_divergence = compute_kl_divergence(
            self.mvn_mean_distribution, self.prior_mvn_mean
        ).sum() * weights[0]    
        _kl_divergence += compute_kl_divergence(
            self.mvn_cov_factor_distribution, self.prior_mvn_cov_factor
        ).sum() * weights[1]
        _kl_divergence += compute_kl_divergence(
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
        profile_mvn_distribution = self.forward(*representations)
        log_probs = profile_mvn_distribution.log_prob(
            self.pixel_positions.expand(batch_size, 1, 1323, 3)
        )

        log_probs_stable = log_probs - log_probs.max(dim=-1, keepdim=True)[0]

        profile = torch.exp(log_probs_stable)

        avg_profile = profile.mean(dim=1)
        avg_profile = avg_profile / (avg_profile.sum(dim=-1, keepdim=True) + 1e-10)

        return profile_mvn_distribution #profile

    def forward(self, *representations: torch.Tensor) -> torch.distributions.LowRankMultivariateNormal:
        """Forward pass of the LRMVN distribution."""
        # Update prior distributions to ensure they're on the correct device
        self._update_prior_distributions()
        
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

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        # Get device from the first parameter
        device = next(self.parameters()).device
        
        # Move all registered buffers
        for name, buffer in self.named_buffers():
            setattr(self, name, buffer.to(device))
            
        # Update prior distributions
        self._update_prior_distributions()
        
        return self