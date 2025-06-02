import torch
import model
import dataclasses
from typing import Optional

@dataclasses.dataclass
class LossOutput():
    loss: Optional[torch.Tensor] = None
    kl_structure_factors: Optional[torch.Tensor] = None
    kl_background: Optional[torch.Tensor] = None
    kl_profile: Optional[torch.Tensor] = None
    log_likelihood: Optional[torch.Tensor] = None

class LossFunction(torch.nn.Module):
    def __init__(self, settings: 'model.Settings', loss_settings: 'model.LossSettings'):
        super().__init__()
        self.settings = settings
        self.LossSettings = loss_settings
        pass

    def compute_kl_divergence(self, predicted_distribution, target_distribution):
        try: 
            return torch.distributions.kl.kl_divergence(predicted_distribution, target_distribution)
        except NotImplementedError:
            print("KL divergence not implemented for this distribution: use sampling method.")
            samples = predicted_distribution.rsample([100])
            log_q = predicted_distribution.log_prob(samples)
            log_p = target_distribution.log_prob(samples)
            return (log_q - log_p).mean(dim=0)
        
    def get_prior(self, distribution, device):
        return distribution #.to(device)
    
    def forward(self, counts, mask, ordered_miller_indices, photon_rate, background_distribution, surrogate_posterior, 
                profile_distribution) -> LossOutput:
        device = photon_rate.device
        batch_size = counts.shape[0]

        loss_output = LossOutput()

        prior_background = self.get_prior(self.settings.background_prior_distribution, device=device)
        prior_intensity = self.get_prior(self.settings.intensity_prior_distibution, device=device)

        # Compute KL divergence for structure factors
        _kl_divergence_folded_normal_all_reflections = self.compute_kl_divergence(
            surrogate_posterior, prior_intensity.distribution()
        )

        rasu_ids = surrogate_posterior.rac.rasu_ids[0]

        _kl_divergence_folded_normal = surrogate_posterior.rac.gather(
            source=_kl_divergence_folded_normal_all_reflections.T, rasu_id=rasu_ids, H=ordered_miller_indices.to(torch.int)
        )

        loss_output.kl_structure_factors = _kl_divergence_folded_normal.sum(-1) * self.LossSettings.prior_intensity_weight

        # Compute KL divergence for background
        loss_output.kl_background = self.compute_kl_divergence(
            background_distribution, prior_background
        ).sum() * self.LossSettings.prior_background_weight

        # Compute KL divergence for profile
        loss_output.kl_profile = (
            profile_distribution.kl_divergence(
                self.LossSettings.profile_loss_weight,
            )
        )

        rate = photon_rate.clamp(min=self.LossSettings.eps)
        log_likelihood = torch.distributions.Poisson(rate=rate+self.LossSettings.eps).log_prob(
            counts.unsqueeze(1))
        log_likelihood_mean = torch.mean(log_likelihood, dim=1) * mask # .unsqueeze(-1)
        negative_log_likelihood_per_batch = (-log_likelihood_mean).sum(dim=1)

        loss_per_batch = negative_log_likelihood_per_batch + _kl_divergence_profile
        loss_output.log_likelihood = negative_log_likelihood_per_batch.sum()
        loss_output.loss = loss_per_batch.mean()
        return loss_output


