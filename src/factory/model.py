import sys, os
import cProfile
# only needed for my local run
repo_root = os.path.abspath(os.path.join(__file__, os.pardir))
inner_pkg = os.path.join(repo_root, "abismal_torch")  

sys.path.insert(0, inner_pkg)
sys.path.insert(0, repo_root)

import data_loader
import torch
import numpy as np
import lightning as L
import dataclasses
# import encoder
import metadata_encoder
import shoebox_encoder
import torch.nn.functional as F
from networks import *
import distributions
import loss_functionality
from callbacks import LossLogging, Plotting, CorrelationPlotting, ScalePlotting
import get_protein_data
import reciprocalspaceship as rs
from abismal_torch.prior import WilsonPrior
from abismal_torch.symmetry.reciprocal_asu import ReciprocalASU, ReciprocalASUGraph
from rasu import *
from abismal_torch.likelihood import NormalLikelihood
from abismal_torch.surrogate_posterior import FoldedNormalPosterior
from wrap_folded_normal import FrequencyTrackingPosterior
from abismal_torch.merging import VariationalMergingModel
from abismal_torch.scaling import ImageScaler

from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
import wandb
# from pytorch_lightning.callbacks import ModelCheckpoint

from torchvision.transforms import ToPILImage
from lightning.pytorch.utilities import grad_norm
import matplotlib.pyplot as plt

from settings import ModelSettings, LossSettings
import settings




import wandb

torch.set_float32_matmul_precision("medium")


# import integrator
# from integrator import data_loaders
# from integrator.model.encoders import cnn_3d, fc_encoder



class Model(L.LightningModule):

    # shoebox_encoder: shoebox_encoder.ShoeboxEncoder #encoder.CNN_3d
    # metadata_encoder: metadata_encoder.MetadataEncoder #encoder.MLPMetadataEncoder
    # scale_function: MLPScale
    profile_distribution: distributions.LRMVN_Distribution
    background_distribution: distributions.HalfNormalDistribution
    surrogate_posterior: FoldedNormalPosterior

    def __init__(self, model_settings: ModelSettings, loss_settings: LossSettings):
        super().__init__()
        self.model_settings = model_settings
        self.shoebox_encoder = model_settings.shoebox_encoder #encoder.CNN_3d()
        self.metadata_encoder = model_settings.metadata_encoder
        self.scale_function = self.model_settings.scale_function
        self.distribution_for_shoebox_profile = self.model_settings.build_shoebox_profile_distribution(torch.load(os.path.join(self.model_settings.data_directory, "concentration.pt"), weights_only=True), dmodel=self.model_settings.dmodel)
        # self.distribution_for_shoebox_profile = self.model_settings.build_shoebox_profile_distribution()
        self.background_distribution = self.model_settings.build_background_distribution(dmodel=self.model_settings.dmodel)
        self.surrogate_posterior = self.compute_intensity_distribution()
        self.optimizer = self.model_settings.optimizer(self.parameters(), lr=self.model_settings.learning_rate)
        self.loss_settings = loss_settings
        self.loss_function = loss_functionality.LossFunction()
        self.loss = torch.Tensor(0)

        self.lr = self.model_settings.learning_rate 
        # self._train_dataloader = train_dataloader

    def setup(self, stage=None):
        # This is called after device placement
        device = self.device

        self.model_settings = Model.move_settings(self.model_settings, device)
        self.loss_settings = Model.move_settings(self.loss_settings, device)


        # self.model_settings.pdb_data = get_protein_data.get_protein_data(self.model_settings.protein_pdb_url)
        # self.model_settings.rac = ReciprocalASUGraph(*[ReciprocalASU(
        #     cell=self.model_settings.pdb_data["unit_cell"],
        #     spacegroup=self.model_settings.pdb_data["spacegroup"],
        #     dmin=float(self.model_settings.pdb_data["dmin"]),
        #     anomalous=True,
        # )])
        # self.model_settings.intensity_prior_distibution = self.model_settings.build_intensity_prior_distribution(self.model_settings.rac.to(device))
        # print("set intensity prior in setup")
        # self.surrogate_posterior = self.compute_intensity_distribution()

        # self.loss_function = loss_functionality.LossFunction(
        #     model_settings=self.model_settings,
        #     loss_settings=self.loss_settings
        # ).to(device)

    @property
    def device(self):
        return next(self.parameters()).device

    # def train_dataloader(self):
    #     return self._train_dataloader

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.model_settings = Model.move_settings(self.model_settings, self.device)
        self.loss_settings = Model.move_settings(self.loss_settings, self.device)
        # print("cuda=", self.model_settings.background_prior_distribution.concentration.device)

        
        if hasattr(self, 'surrogate_posterior'):
            self.surrogate_posterior.to(self.device)
            
        # Move distribution components
        if hasattr(self, 'distribution_for_shoebox_profile'):
            self.distribution_for_shoebox_profile.to(self.device)
            
        if hasattr(self, 'background_distribution'):
            self.background_distribution.to(self.device)

        if hasattr(self, 'scale_function'):
            self.scale_function.to(self.device)

        # self.loss_function = loss_functionality.LossFunction(model_settings=self.model_settings, loss_settings=self.loss_settings)

            
        return self

    @staticmethod
    def move_settings(settings: dataclasses.dataclass, device: torch.device):
        moved = {}
        for field in dataclasses.fields(settings):
            if not field.init:
                continue  # Skip fields that can't be set in __init__
            val = getattr(settings, field.name)
            if isinstance(val, torch.distributions.Distribution):
                param_names = val.arg_constraints.keys()
                params = {name: getattr(val, name).to(device) if hasattr(getattr(val, name), 'to') else getattr(val, name)
                        for name in param_names}
                moved[field.name] = type(val)(**params)
                # print("moved", val)

            elif hasattr(val, "to") and callable(getattr(val, "to")):
                try:
                    moved[field.name] = val.to(device)
                    # print("moved", val)
                except Exception as e:
                    # print(f"failed move with {e} for {val} ")
                    moved[field.name] = val  # fallback if .to() fails
            else:
                # print(f"{val} has not .to")
                moved[field.name] = val
        return dataclasses.replace(settings, **moved)

    def compute_intensity_distribution(self):
        initial_mean = self.model_settings.intensity_prior_distibution.distribution().mean() # self.model_settings.intensity_distribution_initial_location
        print("intensity folded normal mean shape", initial_mean.shape)
        initial_scale = 0.05 * initial_mean # self.model_settings.intensity_distribution_initial_scale
        surrogate_posterior = FrequencyTrackingPosterior.from_unconstrained_loc_and_scale(
            rac=self.model_settings.rac, loc=initial_mean, scale=initial_scale # epsilon=settings.epsilon
        )

        def check_grad_hook(grad):
            if grad is not None:
                print(f"Gradient stats: min={grad.min()}, max={grad.max()}, mean={grad.mean()}, any_nan={torch.isnan(grad).any()}, all_finite={torch.isfinite(grad).all()}")
            return grad
        surrogate_posterior.distribution.loc.register_hook(check_grad_hook)

        return surrogate_posterior
    
    def compute_scale(self, representations: list[torch.Tensor]) -> torch.distributions.Normal:
        joined_representation = sum(representations) if len(representations) > 1 else representations[0] #self._add_representation_(image_representation, metadata_representation) # (batch_size, dmodel)
        scale = self.scale_function(joined_representation)
        # self.logger.experiment.log({
        #     "Scale/ MLP mean": scale.network.mean().item(),
        #     "Scale/ MLP min": scale.network.min().item(),
        #     "Scale/ MLP max": scale.network.max().item()
        # })
        return scale.distribution
    
    def _add_representation_(self, representation1, representation2):
        return representation1 + representation2
    
    def compute_shoebox_profile(self, representations: list[torch.Tensor]):
        return self.distribution_for_shoebox_profile.compute_profile(*representations)

    def compute_background_distribution(self, shoebox_representation):
        return self.background_distribution(shoebox_representation)

    def compute_photon_rate(self, scale_distribution, background_distribution, profile, surrogate_posterior, ordered_miller_indices, verbose_output) -> torch.Tensor: 

        samples_surrogate_posterior = surrogate_posterior.rsample([self.model_settings.number_of_mc_samples]) # ( number of samples, rac_size)

        print("scale_distribution.rsample([self.model_settings.number_of_mc_samples])", scale_distribution.rsample([self.model_settings.number_of_mc_samples]).shape)
        samples_scale = scale_distribution.rsample([self.model_settings.number_of_mc_samples]).permute(1, 0, 2) #(batch_size, number_of_samples, 1)

        self.logger.experiment.log({"scale_samples_average": torch.mean(samples_scale)})

        samples_profile = profile.rsample([self.model_settings.number_of_mc_samples]).permute(1,0,2)
        print("compute photon rate with samples like", samples_profile.shape)
        samples_background = background_distribution.rsample([self.model_settings.number_of_mc_samples]).permute(1, 0, 2) #(batch_size, number_of_samples, 1)
        print("samples_background", samples_background.shape)
        rasu_ids = surrogate_posterior.rac.rasu_ids[0]
        samples_predicted_structure_factor = self.surrogate_posterior.rac.gather(
            source=samples_surrogate_posterior.T, rasu_id=rasu_ids, H=ordered_miller_indices
        ).unsqueeze(-1)
        print("gathered structure factor samples", samples_predicted_structure_factor.shape)

        if verbose_output:
            photon_rate = samples_scale * torch.square(samples_predicted_structure_factor) * samples_profile + samples_background
            return {
                'photon_rate': photon_rate,
                'samples_profile': samples_profile,
                'samples_surrogate_posterior': samples_surrogate_posterior,
                'samples_predicted_structure_factor': samples_predicted_structure_factor,
                'samples_scale': samples_scale,
                'samples_background': samples_background
            }
        else:
            photon_rate = samples_scale * torch.square(samples_predicted_structure_factor) * samples_profile + samples_background  # [batch_size, mc_samples, pixels]
            return photon_rate
    
    def _cut_metadata(self, metadata) -> torch.Tensor:
        return torch.index_select(metadata, dim=1, index=torch.tensor(self.model_settings.metadata_indices_to_keep, device=self.device))

    def _batch_to_representations(self, batch: tuple):
        shoeboxes_batch, metadata_batch, dead_pixel_mask_batch, counts_batch, hkl_batch = batch
        standardized_counts = shoeboxes_batch #[:,:,-1].reshape(shoeboxes_batch.shape[0], 1, 3, 21, 21)
        print("sb shape", standardized_counts.shape)

        shoebox_representation = self.shoebox_encoder(standardized_counts.reshape(shoeboxes_batch.shape[0], 1, 3, 21, 21), mask=dead_pixel_mask_batch)
        print("shoebox representation:", shoebox_representation.shape)


        metadata_representation = self.metadata_encoder(self._cut_metadata(metadata_batch).float()) # (batch_size, dmodel)
        print("metadata representation (batch size, dmodel)", metadata_representation.shape)

        joined_shoebox_representation = metadata_representation + shoebox_representation
        
        pooled_image_representation = torch.max(joined_shoebox_representation, dim=0, keepdim=True)[0] # (1, dmodel)
        image_representation = pooled_image_representation # + metadata_representation
        print("image rep (1, dmodel)", image_representation.shape)

        if torch.isnan(metadata_representation).any():
            raise ValueError("MLP metadata_representation produced NaNs!")
        if torch.isnan(shoebox_representation).any():
            raise ValueError("MLP shoebox_representation produced NaNs!")
        
        return shoebox_representation, metadata_representation, image_representation
    
    def forward(self, batch, verbose_output=False):
        try:
            shoebox_representation, metadata_representation, image_representation = self._batch_to_representations(batch=batch)
            shoeboxes_batch, metadata_batch, dead_pixel_mask_batch, counts_batch, hkl_batch = batch

            scale_distribution = self.compute_scale(representations=[image_representation, metadata_representation]) #  torch.distributions.Normal instance
            shoebox_profile = self.compute_shoebox_profile(representations=[shoebox_representation, image_representation])
            background_distribution = self.compute_background_distribution(shoebox_representation=shoebox_representation+image_representation)
            self.surrogate_posterior.update_observed(rasu_id=self.surrogate_posterior.rac.rasu_ids[0], H=hkl_batch)

            photon_rate = self.compute_photon_rate(
                scale_distribution=scale_distribution,
                background_distribution=background_distribution,
                profile=shoebox_profile,
                surrogate_posterior=self.surrogate_posterior,
                ordered_miller_indices=hkl_batch,
                verbose_output=verbose_output
            )
            if verbose_output:
                return (shoeboxes_batch, photon_rate, hkl_batch, counts_batch)

            return (counts_batch, dead_pixel_mask_batch, hkl_batch, photon_rate, background_distribution, scale_distribution, self.surrogate_posterior, 
                    self.distribution_for_shoebox_profile)
        except Exception as e:
            print(f"failed in forward: {e}")
    
    def training_step(self, batch):
        output = self(batch=batch)
        loss_output = self.loss_function(*output, self.model_settings, self.loss_settings)

        self.logger.experiment.log({
                "loss/kl_structure_factors": loss_output.kl_structure_factors.item(),
                "loss/kl_background": loss_output.kl_background.item(),
                "loss/kl_profile": loss_output.kl_profile.item(),
                # "loss/kl_scale": loss_output.kl_scale.item(),
                "loss/log_likelihood": loss_output.log_likelihood.item(),
            })

        self.loss = loss_output.loss
        # self.logger.experiment.log({"train_loss": self.loss})
        print("log loss step")
        self.log("train/loss_step", self.loss, on_step=True, on_epoch=True)
        norms = grad_norm(self, norm_type=2)
        for name, norm in norms.items():
            self.log(f"grad_norm/{name}", norm)

        print("Loss: ", self.loss)
        return self.loss
    
    def validation_step(self, batch, batch_idx):
        print(f"Validation step called for batch {batch_idx}")
        output = self(batch=batch)
        loss_output = self.loss_function(*output, self.model_settings, self.loss_settings)

        # self.log("validation_loss/kl_structure_factors", loss_output.kl_structure_factors.item(), on_step=False, on_epoch=True)
        self.log("validation_loss/kl_background", loss_output.kl_background.item(), on_step=False, on_epoch=True)
        self.log("validation_loss/kl_profile", loss_output.kl_profile.item(), on_step=False, on_epoch=True)
        self.log("validation_loss/log_likelihood", loss_output.log_likelihood.item(), on_step=False, on_epoch=True)
        # print("Validation Loss: ", loss_output.loss.item())
        # self.log("validation_loss/loss", loss_output.loss.item(), on_step=False, on_epoch=True)
        self.log("validation/loss_step_epoch", loss_output.loss.item(), on_step=False, on_epoch=True)


        return loss_output.loss

    def configure_optimizers(self):
        return self.optimizer
    
    # def val_dataloader(self):
    #     return self.dataloader.load_data_set_batched_by_image(
    #     data_set_to_load=self.dataloader.validation_data_set,
    #     )


        