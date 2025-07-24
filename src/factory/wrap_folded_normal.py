import torch
from typing import Optional
from abismal_torch.surrogate_posterior import FoldedNormalPosterior
from abismal_torch.symmetry import ReciprocalASUCollection

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