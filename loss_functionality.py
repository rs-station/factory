import torch
import model

class LossFunction(torch.nn.Module):
    def __init__(self, settings: model.Settings, loss_settings: model.LossSettings):
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
    
    def forward(self, counts, mask, photon_rate, background_distribution, intensity_distribution, 
                mvn_mean_distribution, mvn_cov_factor_distribution, mvn_cov_scale_distibution):
        device = photon_rate.device

        prior_background = self.get_prior(self.settings.background_prior_distribution, device=device)
        prior_intensity = self.get_prior(self.settings.build_intensity_prior_distribution, device=device)
        prior_mvn_mean = self.get_prior(self.settings.mvn_mean_prior_distribution, device=device)
        prior_mvn_cov_factor = self.get_prior(self.settings.mvn_cov_factor_prior_distribution, device=device)
        prior_mvn_cov_scale = self.get_prior(self.settings.mvn_cov_scale_prior_distribution, device=device)

        kl_divergence = torch.zeros(self.settings.batch_size, device=device)
        kl_divergence += self.compute_kl_divergence(
            background_distribution, prior_background
        ).sum(-1) * self.LossSettings.prior_background_weight

        # kl_divergence += self.compute_kl_divergence(
        #     intensity_distribution, prior_intensity
        # ) * self.LossSettings.prior_intensity_weight

        kl_divergence += self.compute_kl_divergence(
            mvn_mean_distribution, prior_mvn_mean
        ).sum() * self.LossSettings.mvn_mean_weight
        kl_divergence += self.compute_kl_divergence(
            mvn_cov_factor_distribution, prior_mvn_cov_factor
        ).sum() * self.LossSettings.mvn_cov_factor_weight
        kl_divergence += self.compute_kl_divergence(
            mvn_cov_scale_distibution, prior_mvn_cov_scale
        ).sum() * self.LossSettings.mvn_cov_scale_weight

        rate = photon_rate.clamp(min=self.LossSettings.eps)
        log_likelihood = torch.distributions.Poisson(rate=rate+self.LossSettings.eps).log_prob(
            counts.unsqueeze(1))
        log_likelihood_mean = torch.mean(log_likelihood, dim=1) * mask# .unsqueeze(-1)
        negative_log_likelihood_per_batch = (-log_likelihood_mean).sum(dim=1)

        loss_per_batch = negative_log_likelihood_per_batch + kl_divergence
        loss = loss_per_batch.mean()
        return loss


