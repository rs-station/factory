repo_root = os.path.abspath(os.path.join(__file__, os.pardir))
inner_pkg = os.path.join(repo_root, "abismal_torch")  

sys.path.insert(0, inner_pkg)
sys.path.insert(0, repo_root)

import sys, os
import cProfile

import pandas as pd

import data_loader
import torch
import numpy as np
import lightning as L
import dataclasses
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
from reciprocalspaceship.utils import apply_to_hkl, generate_reciprocal_asu
from rasu import *
from abismal_torch.likelihood import NormalLikelihood
from abismal_torch.surrogate_posterior import FoldedNormalPosterior
from wrap_folded_normal import FrequencyTrackingPosterior, SparseFoldedNormalPosterior
from abismal_torch.merging import VariationalMergingModel
from abismal_torch.scaling import ImageScaler
from positional_encoding import positional_encoding

from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
import wandb
# from pytorch_lightning.callbacks import ModelCheckpoint

from torchvision.transforms import ToPILImage
from lightning.pytorch.utilities import grad_norm
import matplotlib.pyplot as plt
import io
from PIL import Image

from settings import ModelSettings, LossSettings
import settings

import rs_distributions.modules as rsm
from masked_adam import MaskedAdam, make_lazy_adabelief_for_surrogate_posterior

from torch.optim.lr_scheduler import StepLR, MultiStepLR

from adabelief_pytorch import AdaBelief





import wandb

torch.set_float32_matmul_precision("medium")


import torch
 


class Model(L.LightningModule):

    automatic_optimization = True
    # shoebox_encoder: shoebox_encoder.ShoeboxEncoder #encoder.CNN_3d
    # metadata_encoder: metadata_encoder.MetadataEncoder #encoder.MLPMetadataEncoder
    # scale_function: MLPScale
    profile_distribution: distributions.LRMVN_Distribution
    background_distribution: distributions.HalfNormalDistribution
    surrogate_posterior: FoldedNormalPosterior

    def __init__(self, model_settings: ModelSettings, loss_settings: LossSettings):
        super().__init__()
        self.model_settings = model_settings
        self.profile_encoder = model_settings.profile_encoder #encoder.CNN_3d()
        self.metadata_encoder = model_settings.metadata_encoder
        self.intensity_encoder = model_settings.intensity_encoder
        self.scale_function = self.model_settings.scale_function
        self.profile_distribution = self.model_settings.build_shoebox_profile_distribution(torch.load(os.path.join(self.model_settings.data_directory, "concentration.pt"), weights_only=True), dmodel=self.model_settings.dmodel)
        self.background_distribution = self.model_settings.build_background_distribution(dmodel=self.model_settings.dmodel)
        self.surrogate_posterior = self.initialize_surrogate_posterior()
        self.dispersion_param = torch.tensor(loss_settings.dispersion_parameter) #torch.nn.Parameter(torch.tensor(0.001))torch.nn.Parameter(torch.tensor(0.001)) #torch.tensor(loss_settings.dispersion_parameter) #torch.nn.Parameter(torch.tensor(0.001))
        self.loss_settings = loss_settings
        self.loss_function = loss_functionality.LossFunction()
        self.loss = torch.Tensor(0)

        self.lr = self.model_settings.learning_rate 
        # self._train_dataloader = train_dataloader

        pdb_data = get_protein_data.get_protein_data(self.model_settings.protein_pdb_url)
        rac = ReciprocalASUGraph(*[ReciprocalASU(
            cell=pdb_data["unit_cell"],
            spacegroup=pdb_data["spacegroup"],
            dmin=float(pdb_data["dmin"]),
            anomalous=True,
        )])
        self.intensity_prior_distibution = self.model_settings.build_intensity_prior_distribution(rac)
        H_rasu = generate_reciprocal_asu(pdb_data["unit_cell"], pdb_data["spacegroup"], float(pdb_data["dmin"]), True)
        self.kl_structure_factors_loss = torch.tensor(0)

    def setup(self, stage=None):
        device = self.device

        self.model_settings = Model.move_settings(self.model_settings, device)
        self.loss_settings = Model.move_settings(self.loss_settings, device)


    @property
    def device(self):
        return next(self.parameters()).device

    # def train_dataloader(self):
    #     return self._train_dataloader

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.model_settings = Model.move_settings(self.model_settings, self.device)
        self.loss_settings = Model.move_settings(self.loss_settings, self.device)

        if hasattr(self, 'surrogate_posterior'):
            self.surrogate_posterior.to(self.device)
            
        if hasattr(self, 'profile_distribution'):
            self.profile_distribution.to(self.device)
            
        if hasattr(self, 'background_distribution'):
            self.background_distribution.to(self.device)

        if hasattr(self, 'scale_function'):
            self.scale_function.to(self.device)
        
        # Ensure dataloader is configured for the current device
        if hasattr(self, 'dataloader') and self.dataloader is not None:
            if hasattr(self.dataloader, 'pin_memory'):
                self.dataloader.pin_memory = True

        return self

    # def transfer_batch_to_device(self, batch, device, dataloader_idx=0):
    #     """Override to ensure all tensors in batch are moved to the correct device."""
    #     def _to_device(obj):
    #         if isinstance(obj, torch.Tensor):
    #             return obj.to(device)
    #         elif isinstance(obj, (list, tuple)):
    #             return type(obj)(_to_device(item) for item in obj)
    #         elif isinstance(obj, dict):
    #             return {key: _to_device(value) for key, value in obj.items()}
    #         else:
    #             return obj
        
    #     return _to_device(batch)

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
                except Exception as e:
                    moved[field.name] = val  
            else:
                moved[field.name] = val
        return dataclasses.replace(settings, **moved)

    # def set_dataloader(self, dataloader):
    #     self.dataloader = dataloader
        
    #     # Ensure dataloader is configured for GPU acceleration
    #     if hasattr(self.dataloader, 'pin_memory'):
    #         self.dataloader.pin_memory = True
        
    #     # If we have a device property, ensure proper configuration
    #     if hasattr(self, 'device'):
    #         # Configure for the current device
    #         if hasattr(self.dataloader, 'pin_memory'):
    #             self.dataloader.pin_memory = True
    #         if hasattr(self.dataloader, 'num_workers'):
    #             # Use multiple workers for faster data loading
    #             self.dataloader.num_workers = min(4, self.dataloader.num_workers if hasattr(self.dataloader, 'num_workers') else 0)
        
    #     return self

    def initialize_surrogate_posterior(self):
        initial_mean = self.model_settings.intensity_prior_distibution.distribution().mean() # self.model_settings.intensity_distribution_initial_location
        print("intensity folded normal mean shape", initial_mean.shape)
        initial_scale = 0.05 * initial_mean # self.model_settings.intensity_distribution_initial_scale

        surrogate_posterior = FrequencyTrackingPosterior.from_unconstrained_loc_and_scale(
            rac=self.model_settings.rac, loc=initial_mean, scale=initial_scale # epsilon=settings.epsilon
        )

        # def check_grad_hook(grad):
        #     if grad is not None:
        #         print(f"Gradient stats: min={grad.min()}, max={grad.max()}, mean={grad.mean()}, any_nan={torch.isnan(grad).any()}, all_finite={torch.isfinite(grad).all()}")
        #     return grad
        # surrogate_posterior.distribution.loc.register_hook(check_grad_hook)

        return surrogate_posterior
   
    def compute_scale(self, representations: list[torch.Tensor]) -> torch.distributions.Normal:
        joined_representation = sum(representations) if len(representations) > 1 else representations[0] #self._add_representation_(image_representation, metadata_representation) # (batch_size, dmodel)
        scale = self.scale_function(joined_representation)
        return scale.distribution
    
    def _add_representation_(self, representation1, representation2):
        return representation1 + representation2
    
    def compute_shoebox_profile(self, representations: list[torch.Tensor]):
        return self.profile_distribution.compute_profile(*representations)

    def compute_background_distribution(self, shoebox_representation):
        return self.background_distribution(shoebox_representation)

    def compute_photon_rate(self, scale_distribution, background_distribution, profile, surrogate_posterior, ordered_miller_indices, metadata, verbose_output) -> torch.Tensor: 

        _samples_predicted_structure_factor = surrogate_posterior.rsample([self.model_settings.number_of_mc_samples]).unsqueeze(-1).permute(1, 0, 2)
        samples_predicted_structure_factor = self.surrogate_posterior.rac.gather(_samples_predicted_structure_factor, torch.tensor([0]), ordered_miller_indices)

        samples_scale = scale_distribution.rsample([self.model_settings.number_of_mc_samples]).permute(1, 0, 2) #(batch_size, number_of_samples, 1)

        self.logger.experiment.log({"scale_samples_average": torch.mean(samples_scale)})

        samples_profile = profile.rsample([self.model_settings.number_of_mc_samples]).permute(1,0,2)
        samples_background = background_distribution.rsample([self.model_settings.number_of_mc_samples]).permute(1, 0, 2) #(batch_size, number_of_samples, 1)

        if verbose_output:
            photon_rate = samples_scale * torch.square(samples_predicted_structure_factor) * samples_profile + samples_background
            return {
                'photon_rate': photon_rate,
                'samples_profile': samples_profile,
                'samples_predicted_structure_factor': samples_predicted_structure_factor,
                'samples_scale': samples_scale,
                'samples_background': samples_background
            }
        else:
            photon_rate = samples_scale * torch.square(samples_predicted_structure_factor) * samples_profile + samples_background  # [batch_size, mc_samples, pixels]
            return photon_rate
    
    def _cut_metadata(self, metadata) -> torch.Tensor:
        # metadata_cut = torch.index_select(metadata, dim=1, index=torch.tensor(self.model_settings.metadata_indices_to_keep, device=self.device))
        if self.model_settings.use_positional_encoding:
            encoded_metadata = positional_encoding(X=metadata, L=self.model_settings.number_of_frequencies_in_positional_encoding)
            print("encoding dim metadata", encoded_metadata.shape)
            return encoded_metadata
        else:
            return metadata

    def _log_representation_stats(self, tensor: torch.Tensor, name: str):
        """Log mean, max, and min statistics for a tensor representation."""
        if tensor is None:
            return
        stats = {
            f"{name}/mean": tensor.mean().item(),
            f"{name}/min": tensor.min().item(),
            f"{name}/max": tensor.max().item(),
        }
        
        if hasattr(self, 'logger') and self.logger is not None:
            self.logger.experiment.log(stats)

    def get_surrogate_distribution(self, metadata):
        print("get_surrogate_distribution")

        if self.model_settings.use_surrogate_parameters:
            print("entered if")
            # params = [self.surrogate_posterior_parameters[metadata[b,-1].to(torch.int)] for b in range(len(metadat))]

            # idx = metadata[:, -1].long().tolist()
            # params = torch.stack([self.surrogate_posterior_parameters[i] for i in idx], dim=0)
            # return rsm.FoldedNormal(params[:,0], params[:,1])

            idx = metadata[:, -1].long()   # HKL ids
            print("idx", len(idx), idx[:10])
            params = self.surrogate_embed(idx)   # [B, 2]
            loc, raw_scale = params[:, 0], params[:, 1]
            scale = raw_scale.abs().clamp_min(1e-8)  # keep valid
            print("access param embedding", len(scale), len(loc))
            dist = rsm.FoldedNormal(loc, scale)
            
            return dist

    def _batch_to_representations(self, batch: tuple):
        shoeboxes_batch, metadata_batch, dead_pixel_mask_batch, counts_batch, hkl_batch, processed_metadata_batch = batch
        standardized_counts = shoeboxes_batch #[:,:,-1].reshape(shoeboxes_batch.shape[0], 1, 3, 21, 21)
        print("sb shape", standardized_counts.shape)

        shoebox_profile_representation = self.profile_encoder(standardized_counts.reshape(shoeboxes_batch.shape[0], 1, 3, 21, 21), mask=dead_pixel_mask_batch)

        shoebox_intensity_representation = self.intensity_encoder(standardized_counts.reshape(shoeboxes_batch.shape[0], 1, 3, 21, 21), mask=dead_pixel_mask_batch)


        metadata_representation = self.metadata_encoder(self._cut_metadata(processed_metadata_batch).float()) # (batch_size, dmodel)
        print("metadata representation (batch size, dmodel)", metadata_representation.shape)

        joined_shoebox_representation = shoebox_intensity_representation + metadata_representation 
        # joined_shoebox_representation = shoebox_profile_representation
        
        pooled_image_representation = torch.max(joined_shoebox_representation, dim=0, keepdim=True)[0] # (1, dmodel)
        image_representation = pooled_image_representation + metadata_representation
        print("image rep (1, dmodel)", image_representation.shape)

        if torch.isnan(metadata_representation).any():
            raise ValueError("MLP metadata_representation produced NaNs!")
        if torch.isnan(shoebox_representation).any():
            raise ValueError("MLP shoebox_representation produced NaNs!")
        
        return shoebox_representation, metadata_representation, image_representation, shoebox_profile_representation
    
    def forward(self, batch, verbose_output=False):

        try:
            shoebox_representation, metadata_representation, image_representation, shoebox_profile_representation = self._batch_to_representations(batch=batch)
            self._log_representation_stats(shoebox_representation, "sb_rep")
            self._log_representation_stats(metadata_representation, "metadata_rep")
            self._log_representation_stats(image_representation, "image_rep")
            
            shoeboxes_batch, metadata_batch, dead_pixel_mask_batch, counts_batch, hkl_batch, processed_metadata_batch = batch

            print("unique hkls", len(torch.unique(hkl_batch)))

            scale_distribution = self.compute_scale(representations=[metadata_representation, image_representation])#[image_representation, metadata_representation]) #  torch.distributions.Normal instance
            print("compute profile")
            shoebox_profile = self.compute_shoebox_profile(representations=[shoebox_profile_representation])#, image_representation])
            print("compute bg")
            background_distribution = self.compute_background_distribution(shoebox_representation=shoebox_representation)# shoebox_representation)#+image_representation)
            self.surrogate_posterior.update_observed(rasu_id=self.surrogate_posterior.rac.rasu_ids[0], H=hkl_batch)

            print("compute rate")
            photon_rate = self.compute_photon_rate(
                scale_distribution=scale_distribution,
                background_distribution=background_distribution,
                profile=shoebox_profile,
                surrogate_posterior=self.surrogate_posterior,
                ordered_miller_indices=hkl_batch,
                metadata=metadata_batch,
                verbose_output=verbose_output
            )
            if verbose_output:
                return (shoeboxes_batch, photon_rate, hkl_batch, counts_batch)
            return (counts_batch, dead_pixel_mask_batch, hkl_batch, photon_rate, background_distribution, scale_distribution, self.surrogate_posterior, 
                    self.profile_distribution)
        except Exception as e:
            print(f"failed in forward: {e}")
            for param in self.parameters():
                param.grad = None
            raise

    def training_step_(self, batch):
        opt_main, opt_loc = self.optimizers()
        self.eval()
        self.surrogate_posterior.train()
        for n, p in self.named_parameters():
            if not n.startswith("surrogate_posterior."):
                p.requires_grad_(False)
        local_runs = 3 if (self.current_epoch < 0.3*self.trainer.max_epochs) else 1
        for _ in range(local_runs):
            output = self(batch=batch)
            local_loss = self.loss_function(*output, self.model_settings, self.loss_settings, self.dispersion_param).loss
            opt_loc.zero_grad(set_to_none=True)
            self.manual_backward(local_loss)      # builds sparse grads on rows in idx
            opt_loc.step()
        self.train()
        for n, p in self.named_parameters():
            if not n.startswith("surrogate_posterior."):
                p.requires_grad_(True)
        output = self(batch=batch)
        loss_output = self.loss_function(*output, self.model_settings, self.loss_settings, self.dispersion_param)
        opt_main.zero_grad()
        loss = loss_output.loss
        loss.backward()
        opt_main.step()
        self.loss = loss.detach()
        self.logger.experiment.log({
                "loss/kl_structure_factors_step": loss_output.kl_structure_factors.item(),
                "loss/kl_background": loss_output.kl_background.item(),
                "loss/kl_profile": loss_output.kl_profile.item(),
                "loss/kl_scale": loss_output.kl_scale.item(),
                "loss/log_likelihood": loss_output.log_likelihood.item(),
            })
        self.log("disp_param", self.dispersion_param, on_step=True, on_epoch=True)
        self.log("loss/kl_structure_factors", loss_output.kl_structure_factors.item(),on_epoch=True)    
        self.loss = loss
        print("log loss step")
        self.log("train/loss_step", self.loss, on_step=True, on_epoch=True)
        norms = grad_norm(self, norm_type=2)
        for name, norm in norms.items():
            self.log(f"grad_norm/{name}", norm)
        return self.loss
    
    
    def training_step(self, batch):
        output = self(batch=batch)
        loss_output = self.loss_function(*output, self.model_settings, self.loss_settings, self.dispersion_param, self.current_epoch)
        self.loss = loss_output.loss.detach()
        self.logger.experiment.log({
                "loss/kl_structure_factors": loss_output.kl_structure_factors.item(),
                "loss/kl_background": loss_output.kl_background.item(),
                "loss/kl_profile": loss_output.kl_profile.item(),
                "loss/kl_scale": loss_output.kl_scale.item(),
                "loss/log_likelihood": loss_output.log_likelihood.item(),
            })
        self.log("loss/kl_structure_factors", loss_output.kl_structure_factors.item(),on_epoch=True)
        self.loss = loss_output.loss
        self.log("train/loss_step", self.loss, on_step=True, on_epoch=True)
        norms = grad_norm(self, norm_type=2)
        for name, norm in norms.items():
            self.log(f"grad_norm/{name}", norm)
        return self.loss

    
    def validation_step(self, batch, batch_idx): 
        print(f"Validation step called for batch {batch_idx}")
        output = self(batch=batch)
        loss_output = self.loss_function(*output, self.model_settings, self.loss_settings, self.dispersion_param, self.current_epoch)

        self.log("validation_loss/kl_structure_factors", loss_output.kl_structure_factors.item(), on_step=False, on_epoch=True)
        self.log("validation_loss/kl_background", loss_output.kl_background.item(), on_step=False, on_epoch=True)
        self.log("validation_loss/kl_profile", loss_output.kl_profile.item(), on_step=False, on_epoch=True)
        self.log("validation_loss/log_likelihood", loss_output.log_likelihood.item(), on_step=False, on_epoch=True)
        self.log("validation_loss/loss", loss_output.loss.item(), on_step=True, on_epoch=True)
        self.log("validation/loss_step_epoch", loss_output.loss.item(), on_step=False, on_epoch=True)

        return loss_output.loss



    def surrogate_full_sweep(self, dataloader=None, K=1):
        dl = self.dataloader
        opt_main, opt_loc = self.optimizers()
        for n, p in self.named_parameters():
            if not n.startswith("surrogate_posterior."):
                p.requires_grad_(False)
        self.eval()               
        self.surrogate_posterior.train()

        for xb in dl:
            xb = self.transfer_batch_to_device(xb, self.device)
            
            for _ in range(K):                      
                out = self(batch=xb)
                loss_obj = self.loss_function(*out, self.model_settings, self.loss_settings, self.dispersion_param)
                opt_loc.zero_grad(set_to_none=True)
                self.manual_backward(loss_obj.loss)
                opt_loc.step()

        self.train()  
        for n, p in self.named_parameters():
            if not n.startswith("surrogate_posterior."):
                p.requires_grad_(True)

    # def on_train_epoch_end(self):
    #     self.surrogate_full_sweep(K=1)

    def on_train_epoch_end(self):
        super().on_train_epoch_end()
        sched = self.lr_schedulers()

        # metric_value = self.trainer.callback_metrics.get("loss/structure_factor_amplitude", None)

        # # if metric_value is not None:
        # sched.step(metric_value.detach().cpu().item())

        sched.step()   

    def configure_optimizers(self):
        """Configure optimizer based on settings from config YAML."""
        
        # Get optimizer parameters from settings
        optimizer_name = self.model_settings.optimizer_name
        lr = self.model_settings.learning_rate
        betas = self.model_settings.optimizer_betas
        weight_decay = self.model_settings.weight_decay
        eps = float(self.model_settings.optimizer_eps)
        
        if optimizer_name == "AdaBelief":
            opt = AdaBelief( self.parameters(),
                # [   
                #     {'params': [p for n, p in self.named_parameters() if not n.startswith("surrogate_posterior.")], 'lr': 1e-4},
                #     {'params': self.surrogate_posterior.parameters(), 'lr': 1e-3}, 
                # ],
                lr=lr,
                eps=eps,
                betas=betas,
                weight_decouple=self.model_settings.weight_decouple,
                weight_decay=weight_decay,
            )
        elif optimizer_name == "AdamW":
            opt = torch.optim.AdamW(
                self.parameters(),
                lr=lr,
                betas=betas,
                weight_decay=weight_decay,
            )
        elif optimizer_name == "Adam":
            opt = torch.optim.Adam(
                self.parameters(),
                lr=lr,
                betas=betas,
                weight_decay=weight_decay,
            )
        elif optimizer_name == "SGD":
            opt = torch.optim.SGD(
                self.parameters(),
                lr=lr,
                momentum=betas[0] if len(betas) > 0 else 0.9,
            )
        elif optimizer_name == "LazyAdam":
            opt = MaskedAdam(
                self.parameters(),
                lr    = lr,
                betas = betas,
                surrogate_posterior_module=self.surrogate_posterior,
                weight_decay=weight_decay,
            )
        elif optimizer_name =="LazyAdaBelief":
            opt = make_lazy_adabelief_for_surrogate_posterior(
                self, lr=lr, weight_decay=weight_decay,
                decoupled_weight_decay=self.model_settings.weight_decouple, zero_tol=1e-12
            )
            print("self.model_settings.weight_decouple", self.model_settings.weight_decouple)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        # Configure scheduler

        # opt_local = AdaBelief(
        #         [   
        #             #{'params': [p for n, p in self.named_parameters() if not n.startswith("surrogate_posterior.")], 'lr': 1e-4},
        #             {'params': self.surrogate_posterior.parameters(), 'lr': 1e-3}, 
        #         ],
        #         lr=lr,
        #         eps=eps,
        #         betas=[0.9, 0.9],
        #         weight_decouple=self.model_settings.weight_decouple,
        #         weight_decay=weight_decay,
        #     )

        sch = MultiStepLR(opt, milestones=self.model_settings.milestones, gamma=self.model_settings.scheduler_gamma)

        # sch = DebugStepLR(opt, step_size=self.model_settings.scheduler_step, gamma=self.model_settings.scheduler_gamma)

        # sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
        # opt, mode="min", factor=0.1, patience=2, threshold=5e0,
        # cooldown=1, min_lr=1e-5,
        # )
        # return [opt], [{"scheduler": sch, "interval": "epoch", "monitor": "loss/kl_structure_factors"}]

        return [opt], [{"scheduler": sch, "interval": "epoch"}]
    
    def val_dataloader(self):
        return self.dataloader.load_data_set_batched_by_image(
        data_set_to_load=self.dataloader.validation_data_set,
        )


    def inference_to_intensities(self, dataloader, wandb_dir, output_path="predicted_intensities.mtz"):
        self.eval() 
        results = []
        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            # batch = self.transfer_batch_to_device(batch, self.device)
            shoeboxes_batch, metadata_batch, dead_pixel_mask_batch, counts_batch, hkl_batch, processed_metadata_batch = batch


            # Compute representations
            shoebox_representation, metadata_representation, image_representation, shoebox_profile_representation = self._batch_to_representations(batch=batch)

            # Get scale and structure factor distributions
            scale_distribution = self.compute_scale([metadata_representation, image_representation])
            S0 = scale_distribution.rsample([1]).squeeze()
            print("shape so", S0.shape)
            structure_factor_dist = self.surrogate_posterior

            params = self.surrogate_posterior.rac.gather(
                source=torch.stack([self.surrogate_posterior.distribution.loc, 
                                    self.surrogate_posterior.distribution.scale], dim=1), 
                rasu_id=torch.tensor([0], device=self.device), 
                H=hkl_batch
            )

            F_loc = params[:, 0]
            F_scale = params[:, 1].abs().clamp_min(1e-12)

            print("F_loc", F_loc.shape)
            print("F_scale", F_scale.shape)

            I_mean = S0 * (F_loc**2 + F_scale**2)
            I_sigma = S0 * torch.sqrt(2*F_scale**4 + 4*F_scale**2 * F_loc**2)

            # Collect HKL indices
            HKL = hkl_batch.cpu().detach().numpy()
            I_mean_np = I_mean.cpu().detach().numpy().flatten()
            I_sigma_np = I_sigma.cpu().detach().numpy().flatten()

            for i in range(HKL.shape[0]):
                results.append({
                    "H": int(HKL[i, 0]),
                    "K": int(HKL[i, 1]),
                    "L": int(HKL[i, 2]),
                    "I_mean": float(I_mean_np[i]),
                    "I_sigma": float(I_sigma_np[i])
                })

            print(f"Processed batch {batch_idx+1}/{len(dataloader)}")

        # Convert to DataFrame
        df = pd.DataFrame(results)

        # Save to MTZ
        ds = rs.DataSet(df)

        # Define column types according to MTZ conventions
        ds["H"] = ds["H"].astype(int)
        ds["K"] = ds["K"].astype(int)
        ds["L"] = ds["L"].astype(int)
        ds = ds.set_index(["H", "K", "L"])

        ds["I"] = ds["I_mean"].astype(float)
        ds["SIGI"] = ds["I_sigma"].astype(float)
        # ds.assign_dtypes({"I": "J", "SIGI": "Q"})

        if hasattr(ds, "assign_dtypes"):
            ds = ds.assign_dtypes({"I": "J", "SIGI": "Q"})
        elif hasattr(ds, "set_dtypes"):
            ds = ds.set_dtypes({"I": "J", "SIGI": "Q"})
        else:
            ds._mtz_dtypes = {"I": "J", "SIGI": "Q"}
        
        ds.cell = gemmi.UnitCell(79.1, 79.1, 38.4, 90, 90, 90)
        ds.spacegroup = gemmi.SpaceGroup("P43212")

        output_path = f"{wandb_dir}/{output_path}"
        ds.write_mtz(output_path, include_scale=True)
        print(f"Inference MTZ saved to {output_path}")

        return mtz_path

def compute_I_mean_sigma(F_loc, F_scale, S=1.0):
        """
        F_loc, F_scale: torch tensors of folded normal parameters (loc, scale)
        S: deterministic scale factor
        Returns: I_mean, I_sigma
        """
        I_mean = S * (F_loc**2 + F_scale**2)
        Var_F2 = 2*F_scale**4 + 4*F_scale**2 * F_loc**2
        I_sigma = S * torch.sqrt(Var_F2)
        return I_mean, I_sigma