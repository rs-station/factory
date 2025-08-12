import torch
from typing import Optional
from abismal_torch.surrogate_posterior import FoldedNormalPosterior
from abismal_torch.symmetry import ReciprocalASUCollection
import rs_distributions.modules as rsm


class FrequencyTrackingPosterior(FoldedNormalPosterior):

    def __init__(
        self,
        rac: ReciprocalASUCollection,
        loc: torch.Tensor,
        scale: torch.Tensor,
        epsilon: Optional[float] = 1e-12,
        **kwargs
    ):
        """
        A surrogate posterior that tracks how many times each HKL has been observed.
        
        Args:
            rac (ReciprocalASUCollection): ReciprocalASUCollection.
            loc (torch.Tensor): Unconstrained location parameter of the distribution.
            scale (torch.Tensor): Unconstrained scale parameter of the distribution.
            epsilon (float, optional): Epsilon value for numerical stability. Defaults to 1e-12.
        """
        super().__init__(rac, loc, scale, epsilon, **kwargs)
        
        self.register_buffer(
            "observation_count", torch.zeros(self.rac.rac_size, dtype=torch.long)
        )
    
    # def to(self, *args, **kwargs):
    #     super().to(*args, **kwargs)
    #     device = next(self.parameters()).device
    #     if hasattr(self, 'observation_count'):
    #         self.observation_count = self.observation_count.to(device)
    #     if hasattr(self, 'observed'):
    #         self.observed = self.observed.to(device)
    #     return self
    
    def update_observed(self, rasu_id: torch.Tensor, H: torch.Tensor) -> None:
        """
        Update both the observed buffer and the observation count.
        
        Args:
            rasu_id (torch.Tensor): A tensor of shape (n_refln,) that contains the
                rasu ID of each reflection.
            H (torch.Tensor): A tensor of shape (n_refln, 3).
        """
        
        h, k, l = H.T

        
        observed_idx = self.rac.reflection_id_grid[rasu_id, h, k, l]

        
        self.observed[observed_idx] = True
        self.observation_count[observed_idx] += 1
    
    def reliable_observations_mask(self, min_observations: int = 5) -> torch.Tensor:
        """
        Returns a boolean tensor indicating which HKLs have been observed enough times
        to be considered reliable.
        """
        return self.observation_count >= min_observations

    def get_distribution(self, rasu_ids: torch.Tensor, H: torch.Tensor):

        h, k, l = H.T
        idx = self.rac.reflection_id_grid[rasu_ids, h, k, l]
        gathered_loc = self.distribution.loc[idx]
        gathered_scale = self.distribution.scale[idx]
        return rsm.FoldedNormal(gathered_loc, gathered_scale)




from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import rs_distributions.distributions as rsd
from abismal_torch.surrogate_posterior.base import PosteriorBase
from abismal_torch.symmetry import ReciprocalASUCollection

class SparseFoldedNormalPosterior(PosteriorBase):
    def __init__(
        self,
        rac: ReciprocalASUCollection,
        loc: torch.Tensor,
        scale: torch.Tensor,
        epsilon: Optional[float] = 1e-6,
        **kwargs
    ):
        # Call parent with a dummy distribution; we'll override it below
        super().__init__(rac, distribution=None, epsilon=epsilon, **kwargs)

        # Number of unique ASU parameters
        num_params = rac.rac_size  # or however many unique positions you have

        # Embeddings whose gradients will be sparse
        self.loc_embed   = nn.Embedding(num_params, 1, sparse=True)
        self.scale_embed = nn.Embedding(num_params, 1, sparse=True)

        # Initialize them to your starting guesses
        with torch.no_grad():
            # loc_init & scale_init should each be shape [num_params]
            self.loc_embed.weight[:, 0]   = loc
            self.scale_embed.weight[:, 0] = scale

        self.epsilon = epsilon

        self.register_buffer(
            "observation_count", torch.zeros(self.rac.rac_size, dtype=torch.long)
        )


    def get_distribution(self, hkl_indices):
        # Map each reflection to its ASU‐index
        indices = hkl_indices

        # Gather embeddings and squeeze out the singleton dim
        loc_raw   = self.loc_embed(indices).squeeze(-1)
        scale_raw = self.scale_embed(indices).squeeze(-1)

        # Enforce positivity (you could also use transform_to())
        scale = F.softplus(scale_raw) + self.epsilon

        # Functional FoldedNormal has no trainable params—it just wraps a torch.distributions
        return rsd.FoldedNormal(loc=loc_raw, scale=scale)
    
    def update_observed(self, rasu_id: torch.Tensor, H: torch.Tensor) -> None:
        """
        Update both the observed buffer and the observation count.
        
        Args:
            rasu_id (torch.Tensor): A tensor of shape (n_refln,) that contains the
                rasu ID of each reflection.
            H (torch.Tensor): A tensor of shape (n_refln, 3).
        """
        
        h, k, l = H.T

        
        observed_idx = self.rac.reflection_id_grid[rasu_id, h, k, l]

        
        self.observed[observed_idx] = True
        self.observation_count[observed_idx] += 1
    
    def reliable_observations_mask(self, min_observations: int = 5) -> torch.Tensor:
        """
        Returns a boolean tensor indicating which HKLs have been observed enough times
        to be considered reliable.
        """
        return self.observation_count >= min_observations
