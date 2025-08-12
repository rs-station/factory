import torch
import model
import dataclasses
from typing import Optional
import settings
from tensordict.nn.distributions import Delta

from wrap_folded_normal import SparseFoldedNormalPosterior

@dataclasses.dataclass
class LossOutput():
    loss: Optional[torch.Tensor] = None
    kl_structure_factors: Optional[torch.Tensor] = None
    kl_background: Optional[torch.Tensor] = None
    kl_profile: Optional[torch.Tensor] = None
    kl_scale: Optional[torch.Tensor] = None
    log_likelihood: Optional[torch.Tensor] = None

class LossFunction(torch.nn.Module):
    def __init__(self):#, model_settings: settings.ModelSettings, loss_settings:settings.LossSettings):
        super().__init__()
        # self.model_settings = model_settings
        # self.loss_settings = loss_settings

    def compute_kl_divergence(self, predicted_distribution, target_distribution):
        try: 
            return torch.distributions.kl.kl_divergence(predicted_distribution, target_distribution)
        except NotImplementedError:
            print("KL divergence not implemented for this distribution: use sampling method.")
            samples = predicted_distribution.rsample([50])
            log_q = predicted_distribution.log_prob(samples)
            log_p = target_distribution.log_prob(samples)
            return (log_q - log_p).mean(dim=0)

    def compute_kl_divergence_verbose(self, predicted_distribution, target_distribution):
        
        samples = predicted_distribution.rsample([50])
        log_q = predicted_distribution.log_prob(samples)
        print(" pred", log_q)
        log_p = target_distribution.log_prob(samples)
        print(" target", log_p)
        return (log_q - log_p).mean(dim=0)

    def compute_kl_divergence_surrogate_parameters(self, predicted_distribution, target_distribution, ordered_miller_indices):

        samples = predicted_distribution.rsample([50])
        log_q = predicted_distribution.log_prob(samples)
        print("scale: pred", log_q)
        log_p = target_distribution.log_prob(samples)

        rasu_ids = surrogate_posterior.rac.rasu_ids[0]
        print("prior_intensity.rsample([20]).permute(1,0)", prior_intensity.rsample([20]).permute(1,0))
        prior_intensity_samples = surrogate_posterior.rac.gather(
            source=prior_intensity.rsample([20]).permute(1,0).T, rasu_id=rasu_ids, H=ordered_miller_indices.to(torch.int)
        )

        

        log_p = target_distribution.log_prob(samples)
        print("scale: target", log_p)
        return (log_q - log_p).mean(dim=0)
        
    def get_prior(self, distribution, device):
        # Move parameters to device and recreate the distribution
        # if isinstance(distribution, torch.distributions.HalfCauchy):
        #     return torch.distributions.HalfCauchy(torch.tensor(20.0, device=device))
        # elif isinstance(distribution, torch.distributions.HalfNormal):
        #     return torch.distributions.HalfNormal(distribution.scale.to(device))
        # Add more cases as needed
        return distribution
    
    def forward(self, counts, mask, ordered_miller_indices, photon_rate, background_distribution, scale_distribution, surrogate_posterior, 
                profile_distribution, model_settings, loss_settings) -> LossOutput:
        print("in loss")
        try:
            device = photon_rate.device
            batch_size = counts.shape[0]

            loss_output = LossOutput()

            # print("cuda=", next(self.model_settings.intensity_prior_distibution.parameters()).device)
            prior_background = self.get_prior(model_settings.background_prior_distribution, device=device)
            print("got priors bg")
            prior_intensity = model_settings.build_intensity_prior_distribution(model_settings.rac.to(device))
            
            
            # self.get_prior(model_settings.intensity_prior_distibution, device=device)
            print("got priors sf")
            prior_scale = self.get_prior(model_settings.scale_prior_distibution, device=device)
            print("got priors")
            # Compute KL divergence for structure factors

            if model_settings.use_surrogate_parameters:

                rasu_ids = torch.tensor([0 for _ in range(len(ordered_miller_indices))], device=device)

                _kl_divergence_folded_normal = self.compute_kl_divergence(
                        surrogate_posterior, prior_intensity.distribution(rasu_ids, ordered_miller_indices))
                # else:
                #     # Handle case where no valid parameters were found
                #     raise ValueError("No valid surrogate parameters found")

            elif isinstance(surrogate_posterior, SparseFoldedNormalPosterior):
                _kl_divergence_folded_normal = self.compute_kl_divergence(
                    surrogate_posterior.get_distribution(ordered_miller_indices), prior_intensity.distribution())

            else:
                rasu_ids = torch.tensor([0 for _ in range(len(ordered_miller_indices))], device=device)

                _kl_divergence_folded_normal = self.compute_kl_divergence_verbose(
                    surrogate_posterior, prior_intensity.distribution(rasu_ids, ordered_miller_indices))
                print("structure factor distr", surrogate_posterior.rsample([1]).shape)#, prior_intensity.rsample([1]).shape)

                # rasu_ids = surrogate_posterior.rac.rasu_ids[0]

                # _kl_divergence_folded_normal = surrogate_posterior.rac.gather(
                #     source=_kl_divergence_folded_normal_all_reflections.T, rasu_id=rasu_ids, H=ordered_miller_indices.to(torch.int)
                # )

            kl_structure_factors = _kl_divergence_folded_normal * loss_settings.prior_structure_factors_weight

            # print("bkg", background_distribution.sample().shape, prior_background.sample())
            # print(background_distribution.scale.device, prior_background.concentration.device)
            kl_background = self.compute_kl_divergence(
                background_distribution, prior_background
            ) * loss_settings.prior_background_weight

            print("shape background kl", kl_background.shape)

            kl_profile = (
                profile_distribution.kl_divergence_weighted(
                    loss_settings.prior_profile_weight,
                )
            )
            print("shape profile kl", kl_profile.shape)

            print("scale_distribution type:", type(scale_distribution))
            if isinstance(scale_distribution, Delta):
                kl_scale = torch.tensor(0.0, device=scale_distribution.mean.device, dtype=scale_distribution.mean.dtype)
            else:
                kl_scale = self.compute_kl_divergence_verbose(
                    scale_distribution, prior_scale
                ) * loss_settings.prior_scale_weight

            print("kl scale", kl_scale.shape, kl_scale)

            rate = photon_rate #.clamp(min=loss_settings.eps)
            log_likelihood = torch.distributions.Poisson(rate=rate+loss_settings.eps).log_prob(
                counts.unsqueeze(1)) #n_batches, mc_samples, pixels
            print("dim loglikelihood", log_likelihood.shape)
            log_likelihood_mean = torch.mean(log_likelihood, dim=1) * mask # .unsqueeze(-1)
            print("log_likelihood_mean: n_batches, pixels", log_likelihood_mean.shape)
            print("mask", mask.shape)
            negative_log_likelihood_per_batch = (-log_likelihood_mean).sum(dim=1)# batch_size

            loss_per_batch = negative_log_likelihood_per_batch + kl_structure_factors + kl_background + kl_profile + kl_scale
            
            loss_output.loss = loss_per_batch.mean()

            loss_output.kl_structure_factors = kl_structure_factors.mean()
            loss_output.kl_background = kl_background.mean()
            loss_output.kl_profile = kl_profile.mean()
            loss_output.log_likelihood = negative_log_likelihood_per_batch.mean()
            loss_output.kl_scale = kl_scale.mean()

            print("loss_output", loss_output)
            return loss_output
        except Exception as e:
            print(f"loss computation failed with {e}")


