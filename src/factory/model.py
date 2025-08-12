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




import wandb

torch.set_float32_matmul_precision("medium")


# import integrator
# from integrator import data_loaders
# from integrator.model.encoders import cnn_3d, fc_encoder



class Model(L.LightningModule):

    automatic_optimization = False
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
        self.intensity_encoder = model_settings.intensity_encoder
        self.scale_function = self.model_settings.scale_function
        self.distribution_for_shoebox_profile = self.model_settings.build_shoebox_profile_distribution(torch.load(os.path.join(self.model_settings.data_directory, "concentration.pt"), weights_only=True), dmodel=self.model_settings.dmodel)
        # self.distribution_for_shoebox_profile = self.model_settings.build_shoebox_profile_distribution(dmodel=self.model_settings.dmodel)
        self.background_distribution = self.model_settings.build_background_distribution(dmodel=self.model_settings.dmodel)
        self.surrogate_posterior = self.compute_intensity_distribution()
        # self.surrogate_posterior_params = self.compute_posterior_params()
                # self.optimizer = self.model_settings.optimizer(self.parameters(), betas=self.model_settings.optimizer_betas, lr=self.model_settings.learning_rate)
        self.loss_settings = loss_settings
        self.loss_function = loss_functionality.LossFunction()
        self.loss = torch.Tensor(0)

        self.lr = self.model_settings.learning_rate 
        # self._train_dataloader = train_dataloader

    def setup(self, stage=None):
        device = self.device

        self.model_settings = Model.move_settings(self.model_settings, device)
        self.loss_settings = Model.move_settings(self.loss_settings, device)

        pdb_data = get_protein_data.get_protein_data(self.model_settings.protein_pdb_url)
        rac = ReciprocalASUGraph(*[ReciprocalASU(
            cell=pdb_data["unit_cell"],
            spacegroup=pdb_data["spacegroup"],
            dmin=float(pdb_data["dmin"]),
            anomalous=True,
        )])
        self.intensity_prior_distibution = self.model_settings.build_intensity_prior_distribution(rac)

        initial_mean = self.intensity_prior_distibution.distribution().mean()
        initial_scale = 0.05 * initial_mean
        self.initial_surrogate_param = torch.tensor([initial_mean.mean(), initial_scale.mean()], device=self.device)
        H_rasu = generate_reciprocal_asu(pdb_data["unit_cell"], pdb_data["spacegroup"], float(pdb_data["dmin"]), True)

        # Create separate parameters for loc and scale for each HKL
        self.surrogate_posterior_parameters = torch.nn.ParameterDict()
        for i, hkl in enumerate(H_rasu):
            key = str(hkl.tolist())
            self.surrogate_posterior_parameters[f"{key}_loc"] = torch.nn.Parameter(
                torch.tensor(initial_mean[i], device=device)
            )
            self.surrogate_posterior_parameters[f"{key}_scale"] = torch.nn.Parameter(
                torch.tensor(initial_scale[i], device=device)
            )
        self.missing_hkls = 0
        # print("surrogate parans", self.surrogate_posterior_parameters)



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

        # surrogate_posterior = SparseFoldedNormalPosterior(
        #     rac=self.model_settings.rac, loc=initial_mean, scale=initial_scale
        # )

        # def check_grad_hook(grad):
        #     if grad is not None:
        #         print(f"Gradient stats: min={grad.min()}, max={grad.max()}, mean={grad.mean()}, any_nan={torch.isnan(grad).any()}, all_finite={torch.isfinite(grad).all()}")
        #     return grad
        # surrogate_posterior.distribution.loc.register_hook(check_grad_hook)

        return surrogate_posterior

    def compute_posterior_params(self):
        initial_mean = self.model_settings.intensity_prior_distibution.distribution().mean() # self.model_settings.intensity_distribution_initial_location
        initial_scale = 0.05 * initial_mean
        H_rasu = generate_reciprocal_asu()
    
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
        # self.distribution_for_shoebox_profile._update_prior_distributions()
        # dir_prior = self.distribution_for_shoebox_profile.dirichlet_prior.rsample()
        # print("shape dir prior", dir_prior.shape)
        # dir_prior = dir_prior.reshape(3,21,21)
        # plt.figure()
        # plt.imshow(dir_prior[1:2].squeeze().cpu().detach().numpy())
        # buf = io.BytesIO()
        # plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        # buf.seek(0)
        # plt.close()
        
        # # Convert buffer to PIL Image
        # pil_image = Image.open(buf)
        # self.logger.experiment.log({
        #     "concentration vector": wandb.Image(pil_image)
        # })
        return self.distribution_for_shoebox_profile.compute_profile(*representations)

    def compute_background_distribution(self, shoebox_representation):
        return self.background_distribution(shoebox_representation)

    def compute_photon_rate(self, scale_distribution, background_distribution, profile, surrogate_posterior, ordered_miller_indices, metadata, verbose_output) -> torch.Tensor: 

        print("ordered_miller_indices", ordered_miller_indices[0])
        if self.model_settings.use_surrogate_parameters:
            # print("use params")
            # loc = []
            # scale = []
            # n = 0
            # for batch in range(len(metadata)):
            #     print("m", metadata[batch])
            #     print("key", str(metadata[batch, -3:].int().tolist()), ordered_miller_indices[batch])
            #     try:
            #         _loc, _scale = self.surrogate_posterior_parameters[str(metadata[batch, -3:].int().tolist())]
            #         # Ensure parameters are on the correct device
            #         _loc = _loc.to(self.device)
            #         _scale = _scale.to(self.device)
            #         loc.append(_loc)
            #         scale.append(_scale)
            #     except Exception as e:
            #         n+=1
            #         print("failed", e)
                    
            # print("loc", len(loc), "scale", len(scale))
            # print(f"failed for {n} out of {len(metadata)}")
            # # Stack the tensors and ensure they're on the correct device
            # if loc and scale:
            #     loc_tensor = torch.stack(loc).to(self.device)
            #     scale_tensor = torch.stack(scale).to(self.device)
            samples_predicted_structure_factor = surrogate_posterior.rsample([self.model_settings.number_of_mc_samples]).unsqueeze(-1).permute(1, 0, 2)
            print("samples_predicted_structure_factor shape", samples_predicted_structure_factor.shape)

            

        elif isinstance(surrogate_posterior, SparseFoldedNormalPosterior):
            samples_predicted_structure_factor = surrogate_posterior.get_distribution(ordered_miller_indices).rsample([self.model_settings.number_of_mc_samples])
        else:
            samples_predicted_structure_factor = surrogate_posterior.rsample([self.model_settings.number_of_mc_samples]).unsqueeze(-1).permute(1, 0, 2) # ( number of samples, rac_size)
            # print("number of total hkl", len(samples_surrogate_posterior))
            # rasu_ids = surrogate_posterior.rac.rasu_ids[0]
            # samples_predicted_structure_factor = self.surrogate_posterior.rac.gather(
            #     source=samples_surrogate_posterior.T, rasu_id=rasu_ids, H=ordered_miller_indices
            # ).unsqueeze(-1).requires_grad_()
            
            print("gathered structure factor samples", samples_predicted_structure_factor.shape)

        print("scale_distribution.rsample([self.model_settings.number_of_mc_samples])", scale_distribution.rsample([self.model_settings.number_of_mc_samples]).shape)
        samples_scale = scale_distribution.rsample([self.model_settings.number_of_mc_samples]).permute(1, 0, 2) #(batch_size, number_of_samples, 1)

        self.logger.experiment.log({"scale_samples_average": torch.mean(samples_scale)})

        print("profile dim", profile.rsample().shape)

        samples_profile = profile.rsample([self.model_settings.number_of_mc_samples]).permute(1,0,2)
        print("compute photon rate with samples like", samples_profile.shape)
        samples_background = background_distribution.rsample([self.model_settings.number_of_mc_samples]).permute(1, 0, 2) #(batch_size, number_of_samples, 1)
        print("samples_background", samples_background.shape)

        # Memory optimization: Compute photon rate in-place to avoid creating intermediate tensors
        if verbose_output:
            photon_rate = samples_scale * torch.square(samples_predicted_structure_factor) * samples_profile + samples_background
            return {
                'photon_rate': photon_rate,
                'samples_profile': samples_profile,
                # 'samples_surrogate_posterior': samples_surrogate_posterior,
                'samples_predicted_structure_factor': samples_predicted_structure_factor,
                'samples_scale': samples_scale,
                'samples_background': samples_background
            }
        else:
            photon_rate = samples_scale * torch.square(samples_predicted_structure_factor) * samples_profile + samples_background  # [batch_size, mc_samples, pixels]
            return photon_rate
    
    def _cut_metadata(self, metadata) -> torch.Tensor:
        metadata_cut = torch.index_select(metadata, dim=1, index=torch.tensor(self.model_settings.metadata_indices_to_keep, device=self.device))
        if self.model_settings.use_positional_encoding:
            encoded_metadata = positional_encoding(X=metadata_cut, L=self.model_settings.number_of_frequencies_in_positional_encoding)
            print("encoding dim metadata", encoded_metadata.shape)
            return encoded_metadata
        else:
            return metadata_cut

    def _log_representation_stats(self, tensor: torch.Tensor, name: str):
        """Log mean, max, and min statistics for a tensor representation."""
        if tensor is None:
            return
            
        # Compute basic statistics
        stats = {
            f"{name}/mean": tensor.mean().item(),
            f"{name}/min": tensor.min().item(),
            f"{name}/max": tensor.max().item(),
        }
        
        # Log to wandb
        if hasattr(self, 'logger') and self.logger is not None:
            self.logger.experiment.log(stats)

    def get_surrogate_distribution(self, metadata):
        # print("ordered_miller_indices", ordered_miller_indices[0])
        if self.model_settings.use_surrogate_parameters:
            print("use params")
            loc = []
            scale = []
            n = 0
            for batch in range(len(metadata)):
                print("m", metadata[batch])
                # print("key", str(metadata[batch, -3:].int().tolist()), ordered_miller_indices[batch])
                try:
                    key = str(metadata[batch, -3:].int().tolist())
                    if f"{key}_loc" not in self.surrogate_posterior_parameters:
                        print("not in param dict")
                        # Create separate parameters for loc and scale
                        self.surrogate_posterior_parameters[f"{key}_loc"] = torch.nn.Parameter(
                            torch.tensor(self.initial_surrogate_param[0], device=self.device)
                        )
                        self.surrogate_posterior_parameters[f"{key}_scale"] = torch.nn.Parameter(
                            torch.tensor(self.initial_surrogate_param[1], device=self.device)
                        )
                        self.missing_hkls += 1
                        print()
                    
                    _loc = self.surrogate_posterior_parameters[f"{key}_loc"]
                    _scale = self.surrogate_posterior_parameters[f"{key}_scale"]
                    # Ensure parameters are on the correct device
                    _loc = _loc.to(self.device)
                    _scale = _scale.to(self.device)
                    loc.append(_loc)
                    scale.append(_scale)
                except Exception as e:
                    n+=1
                    print("failed", e)
                    
                    
            print("loc", len(loc), "scale", len(scale))
            print(f"failed for {n} out of {len(metadata)}")
            # Stack the tensors and ensure they're on the correct device
            # if loc and scale:
            loc_tensor = torch.stack(loc).to(self.device)
            scale_tensor = torch.stack(scale).to(self.device)
            return rsm.FoldedNormal(loc_tensor, scale_tensor)

    def _batch_to_representations(self, batch: tuple):
        shoeboxes_batch, metadata_batch, dead_pixel_mask_batch, counts_batch, hkl_batch = batch
        standardized_counts = shoeboxes_batch #[:,:,-1].reshape(shoeboxes_batch.shape[0], 1, 3, 21, 21)
        print("sb shape", standardized_counts.shape)

        shoebox_profile_representation = self.shoebox_encoder(standardized_counts.reshape(shoeboxes_batch.shape[0], 1, 3, 21, 21), mask=dead_pixel_mask_batch)
        # print("shoebox representation:", shoebox_representation.shape)

        shoebox_representation = self.intensity_encoder(standardized_counts.reshape(shoeboxes_batch.shape[0], 1, 3, 21, 21), mask=dead_pixel_mask_batch)


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
        
        return shoebox_representation, metadata_representation, image_representation, shoebox_profile_representation
    
    def forward(self, batch, verbose_output=False):

        try:
            shoebox_representation, metadata_representation, image_representation, shoebox_profile_representation = self._batch_to_representations(batch=batch)
            self._log_representation_stats(shoebox_representation, "sb_rep")
            self._log_representation_stats(metadata_representation, "metadata_rep")
            self._log_representation_stats(image_representation, "image_rep")
            
            shoeboxes_batch, metadata_batch, dead_pixel_mask_batch, counts_batch, hkl_batch = batch

            if self.model_settings.use_surrogate_parameters:
                surrogate_distributions_batch = self.get_surrogate_distribution(metadata_batch)
            else:
                surrogate_distributions_batch = self.surrogate_posterior.get_distribution(rasu_id=self.surrogate_posterior.rac.rasu_ids[0], H=hkl_batch)

            print("unique hkls", len(torch.unique(hkl_batch)))

            scale_distribution = self.compute_scale(representations=[metadata_representation])#[image_representation, metadata_representation]) #  torch.distributions.Normal instance
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
                surrogate_posterior=surrogate_distributions_batch,
                ordered_miller_indices=hkl_batch,
                metadata=metadata_batch,
                verbose_output=verbose_output
            )
            if verbose_output:
                return (shoeboxes_batch, photon_rate, hkl_batch, counts_batch)

            return (counts_batch, dead_pixel_mask_batch, hkl_batch, photon_rate, background_distribution, scale_distribution, surrogate_distributions_batch, 
                    self.distribution_for_shoebox_profile)
        except Exception as e:
            print(f"failed in forward: {e}")
    
    def training_step(self, batch):
        opt_dense, opt_sparse = self.optimizers()

        output = self(batch=batch)
        loss_output = self.loss_function(*output, self.model_settings, self.loss_settings)

        self.surrogate_posterior.parameters()

        opt_dense.zero_grad()
        opt_sparse.zero_grad()
        self.manual_backward(loss_output.loss)
        opt_dense.step()
        opt_sparse.step()
        self.logger.experiment.log({
                "loss/kl_structure_factors": loss_output.kl_structure_factors.item(),
                "loss/kl_background": loss_output.kl_background.item(),
                "loss/kl_profile": loss_output.kl_profile.item(),
                "loss/kl_scale": loss_output.kl_scale.item(),
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

        for name, param in self.named_parameters():
            if "alpha_layer" in name:
                print(f"{name}: grad = {param.grad}")


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
        if self.model_settings.use_surrogate_parameters:
            # When using surrogate parameters, include them in the surrogate parameters list
            all_surrogate = list(self.surrogate_posterior.parameters()) + list(self.surrogate_posterior_parameters.values())
        else:
            all_surrogate = list(self.surrogate_posterior.parameters())
            
        other_params          = []
        surrogate_dense       = []
        surrogate_sparse      = []

        for module in self.surrogate_posterior.modules():
            if isinstance(module, torch.nn.Embedding) and module.sparse:
                # embedding.weight will get sparse gradients
                surrogate_sparse.append(module.weight)
        
        surrogate_sparse_ids = {id(p) for p in surrogate_sparse}
        for p in all_surrogate:
            if id(p) in surrogate_sparse_ids:
                continue
            surrogate_dense.append(p)

        surrogate_ids = {id(p) for p in all_surrogate}
        for p in self.parameters():
            if id(p) not in surrogate_ids:
                other_params.append(p)

        opt_dense_main = torch.optim.Adam(
            other_params,
            lr    = self.model_settings.learning_rate,
            betas = self.model_settings.optimizer_betas,
            weight_decay = self.model_settings.weight_decay,  # Add weight decay
        )
        if len(surrogate_dense)>0:
            opt_dense_sur  = torch.optim.Adam(
                surrogate_dense,
                lr    = self.model_settings.surrogate_posterior_learning_rate,
                betas = self.model_settings.optimizer_betas,
                weight_decay = self.model_settings.surrogate_weight_decay,  # Add weight decay for surrogate
            )
            return [opt_dense_main, opt_dense_sur]
        if len(surrogate_sparse)>0:
            opt_sparse_sur = torch.optim.SparseAdam(
                surrogate_sparse,
                lr = self.model_settings.surrogate_posterior_learning_rate,
            )

            return [opt_dense_main, opt_sparse_sur]
    
    # def val_dataloader(self):
    #     return self.dataloader.load_data_set_batched_by_image(
    #     data_set_to_load=self.dataloader.validation_data_set,
    #     )


        