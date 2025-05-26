import sys, os

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
import simplest_encode
import torch.nn.functional as F
from networks import *
import distributions
import loss_functionality
from callbacks import LossLogging, Plotting, CorrelationPlotting, ScalePlotting
import get_protein_data
import reciprocalspaceship as rs
from abismal_torch.prior import WilsonPrior
# from abismal_torch.symmetry.reciprocal_asu import ReciprocalASU, ReciprocalASUGraph
from rasu import *
from abismal_torch.likelihood import NormalLikelihood
from abismal_torch.surrogate_posterior import FoldedNormalPosterior
from abismal_torch.merging import VariationalMergingModel
from abismal_torch.scaling import ImageScaler

from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
import wandb
# from pytorch_lightning.callbacks import ModelCheckpoint

from torchvision.transforms import ToPILImage
from lightning.pytorch.utilities import grad_norm
import matplotlib.pyplot as plt
wandb_logger = WandbLogger(project="full-model", name="testing-1")

# import integrator
# from integrator import data_loaders
# from integrator.model.encoders import cnn_3d, fc_encoder

@dataclasses.dataclass
class Settings():
    build_background_distribution = distributions.HalfNormalDistribution
    build_profile_distribution = distributions.Distribution
    build_shoebox_profile_distribution = distributions.LRMVN_Distribution
    background_prior_distribution: torch.distributions.HalfNormal = dataclasses.field(
        default_factory=lambda: torch.distributions.HalfNormal(1)
    )
    build_intensity_prior_distribution = WilsonPrior
    intensity_prior_distibution: WilsonPrior = dataclasses.field(init=False)
    # intensity_distribution_initial_location: float
    # intensity_distribution_initial_scale: float
    mvn_mean_prior_distribution = torch.distributions.normal.Normal(loc=torch.tensor(0.0), scale=torch.tensor(5.0))
    mvn_cov_factor_prior_distribution = torch.distributions.normal.Normal(loc=torch.tensor(0.0), scale=torch.tensor(0.1))
    mvn_cov_scale_prior_distribution = torch.distributions.half_normal.HalfNormal(scale=torch.tensor(0.1))
    optimizer = torch.optim.AdamW
    dmodel = 64
    # batch_size = 4
    number_of_epochs = 5
    number_of_mc_samples = 100

    # data_directory = r"C:\Users\Flavi\Desktop\Thesis_Data" #"/n/hekstra_lab/people/aldama/subset"
    # data_file_names: dict = dataclasses.field(default_factory=lambda: {
    #     "shoeboxes": "standardized_shoeboxes_subset.pt",
    #     "counts": "raw_counts_subset.pt",
    #     "metadata": "shoebox_features_subset.pt",
    #     "masks": "masks_subset.pt",
    #     "true_reference": "metadata_subset.pt",
    # })

    data_directory = "/n/hekstra_lab/people/aldama/subset"
    data_file_names = {
    "shoeboxes": "standardized_shoeboxes_subset.pt",
    "counts": "raw_counts_subset.pt",
    "metadata": "shoebox_features_subset.pt",
    "masks": "masks_subset.pt",
    "true_reference"
    : "metadata_subset.pt",
    }

    enable_checkpointing = True

    merged_mtz_file_path = "/n/hekstra_lab/people/aldama/subset/merged.mtz"
    unmerged_mtz_file_path = "/n/holylabs/LABS/hekstra_lab/Users/fgiehr/creat_dials_unmerged/unmerged.mtz"
    protein_pdb_url = "https://files.rcsb.org/download/9B7C.cif"
    rac: ReciprocalASUCollection = dataclasses.field(init=False)
    def __post_init__(self):
        pdb_data = get_protein_data.get_protein_data(self.protein_pdb_url)
        self.rac = ReciprocalASUGraph(*[ReciprocalASU(
            cell=pdb_data["unit_cell"],
            spacegroup=pdb_data["spacegroup"],
            dmin=float(pdb_data["dmin"]),
            anomalous=True,
        )])
        self.intensity_prior_distibution = self.build_intensity_prior_distribution(self.rac)


@dataclasses.dataclass
class LossSettings():
    prior_background_weight: float = 0.0001
    prior_intensity_weight: float = 0.0001
    mvn_mean_weight: float = 0.0001
    mvn_cov_factor_weight: float = 0.0001
    mvn_cov_scale_weight: float = 0.0001
    eps: float = 0.00001


class Model(L.LightningModule):

    dataloader: data_loader.CrystallographicDataLoader
    shoebox_encoder: simplest_encode.ShoeboxEncoder #encoder.CNN_3d
    metadata_encoder: simplest_encode.MLPMetadataEncoder #encoder.MLPMetadataEncoder
    scale_model: MLPScale
    lrmvn_distribution_for_shoebox_profile: distributions.LRMVN_Distribution
    background_distribution: distributions.HalfNormalDistribution
    surrogate_posterior: FoldedNormalPosterior

    def __init__(self, settings: Settings, loss_settings: LossSettings, dataloader: data_loader.CrystallographicDataLoader):
        super().__init__()
        self.settings = settings
        self.dataloader = dataloader
        self.shoebox_encoder = simplest_encode.ShoeboxEncoder() #encoder.CNN_3d()
        self.metadata_encoder = simplest_encode.MLPMetadataEncoder(feature_dim=2, output_dims=self.settings.dmodel) #encoder.MLPMetadataEncoder()
        self.scale_model = MLPScale(input_dimension=self.settings.dmodel, hidden_dimension=self.settings.dmodel)
        self.lrmvn_distribution_for_shoebox_profile = self.settings.build_shoebox_profile_distribution()
        self.background_distribution = self.settings.build_background_distribution(dmodel=self.settings.dmodel)
        self.surrogate_posterior = self.compute_intensity_distribution()
        self.optimizer = self.settings.optimizer(self.parameters())
        self.loss_settings = loss_settings
        self.loss_function = loss_functionality.LossFunction(settings=settings, loss_settings=self.loss_settings)
        self.loss = torch.Tensor(0)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        device = next(self.parameters()).device
        self._move_settings_(self.settings)
        self._move_settings_(self.loss_settings)
        return self

    def _move_settings_(self, settings:dataclasses.dataclass):
        device = next(self.parameters()).device
        moved = {}
        for field in dataclasses.fields(self.settings):
            val = getattr(self.settings, field.name)
            if isinstance(val, torch.Tensor):
                moved[field.name] = val.to(device)
        # 3) recreate a new dataclass with those moved fields
        self.settings = dataclasses.replace(self.settings, **moved)
        return self

    def compute_intensity_distribution(self):
        initial_mean = self.settings.intensity_prior_distibution.distribution().mean() # self.settings.intensity_distribution_initial_location
        print("intensity folded normal mean shape", initial_mean.shape)
        initial_scale = 0.05 * initial_mean # self.settings.intensity_distribution_initial_scale
        surrogate_posterior = FoldedNormalPosterior.from_unconstrained_loc_and_scale(
            rac=self.settings.rac, loc=initial_mean, scale=initial_scale # epsilon=settings.epsilon
        )

        def check_grad_hook(grad):
            if grad is not None:
                print(f"Gradient stats: min={grad.min()}, max={grad.max()}, mean={grad.mean()}, any_nan={torch.isnan(grad).any()}, all_finite={torch.isfinite(grad).all()}")
            return grad
        surrogate_posterior.distribution.loc.register_hook(check_grad_hook)

        return surrogate_posterior
    
    def compute_scale(self, image_representation, metadata_representation) -> torch.distributions.Normal:
        joined_representation = image_representation #self._add_representation_(image_representation, metadata_representation) # (batch_size, dmodel)
        scale = self.scale_model(joined_representation)
        self.logger.experiment.log({
            "Scale/ MLP mean": scale.network.mean().item(),
            "Scale/ MLP min": scale.network.min().item(),
            "Scale/ MLP max": scale.network.max().item()
        })
        return scale.distribution
    
    def _add_representation_(self, representation1, representation2):
        return representation1 + representation2
    
    def compute_shoebox_profile(self, shoebox_representation, image_representation):
        batch_size = shoebox_representation.shape[0]
        profile_mvn_distribution = self.lrmvn_distribution_for_shoebox_profile(
            shoebox_representation, image_representation
        )
        log_probs = profile_mvn_distribution.log_prob(
            self.lrmvn_distribution_for_shoebox_profile.pixel_positions.expand(batch_size, 1, 1323, 3)
        )

        log_probs_stable = log_probs - log_probs.max(dim=-1, keepdim=True)[0]

        profile = torch.exp(log_probs_stable)

        avg_profile = profile.mean(dim=1)
        avg_profile = avg_profile / (avg_profile.sum(dim=-1, keepdim=True) + 1e-10)

        return profile

    def compute_background_distribution(self, shoebox_representation):
        return self.background_distribution(shoebox_representation)

    def compute_photon_rate(self, scale_distribution, background_distribution, profile, surrogate_posterior, ordered_miller_indices, log_images) -> torch.Tensor: 

        samples_surrogate_posterior = surrogate_posterior.rsample([self.settings.number_of_mc_samples]) # ( number of samples, rac_size)

        samples_scale = scale_distribution.rsample([self.settings.number_of_mc_samples]).unsqueeze(-1).permute(1, 0, 2) #(batch_size, number_of_samples, 1)
        samples_profile = profile / profile.sum(dim=-1, keepdim=True) + 1e-10 # (batch_size, number of samples, pixels)

        samples_background = background_distribution.rsample([self.settings.number_of_mc_samples]).permute(1, 0, 2) #(batch_size, number_of_samples, 1)

        rasu_ids = surrogate_posterior.rac.rasu_ids[0]
        samples_predicted_structure_factor = self.surrogate_posterior.rac.gather(
            source=samples_surrogate_posterior.T, rasu_id=rasu_ids, H=ordered_miller_indices
        ).unsqueeze(-1)
        print("gathered structure factor samples", samples_predicted_structure_factor.shape)

        if log_images:
            return (samples_scale * torch.square(samples_predicted_structure_factor) * samples_profile + samples_background,
                    samples_profile, samples_surrogate_posterior, samples_predicted_structure_factor)
        else:
            photon_rate = samples_scale * torch.square(samples_predicted_structure_factor) * samples_profile + samples_background  # [batch_size, mc_samples, pixels]
            self.logger.experiment.log({
                "scale/mean_from_samples": samples_scale.mean().item(),
                "scale/max_from_samples": samples_scale.max().item(),
                "scale/min_from_samples": samples_scale.min().item(),
                "structure_factor/mean_from_samples": samples_predicted_structure_factor.mean().item(),
                "structure_factor/max_from_samples": samples_predicted_structure_factor.max().item(),
                "structure_factor/min_from_samples": samples_predicted_structure_factor.min().item(),
                "profile/mean_from_samples": samples_profile.mean().item(),
                "profile/max_from_samples": samples_profile.max().item(),
                "profile/min_from_samples": samples_profile.mean().min(),
                "background/mean_from_samples": samples_background.mean().item(),
                "background/max_from_samples": samples_background.max().item(),
                "background/min_from_samples": samples_background.min().item(),
                "photon_rate/mean_from_samples": photon_rate.mean().item(),
                "photon_rate/max_from_samples": photon_rate.max().item(),
                "photon_rate/min_from_samples": photon_rate.min().item(),
            })
            return photon_rate
    
    def _cut_metadata(self, metadata) -> torch.Tensor:
        device = next(self.parameters()).device
        return torch.index_select(metadata, dim=1, index=torch.tensor([4,5], device=device)) # 4,5 corresponds to 

    def forward(self, batch, log_images=False):
        shoeboxes_batch, metadata_batch, dead_pixel_mask_batch, counts_batch, hkl_batch, dials_reference_batch = batch
        print("PLACE shoeboxes on", shoeboxes_batch.device)

        # shoebox_representation = self.shoebox_encoder(shoeboxes_batch) # (batch_size, dmodel)
        standardized_counts = shoeboxes_batch[:,:,-1].reshape(shoeboxes_batch.shape[0], 1, 3, 21, 21)
        print("sb shape", standardized_counts.shape)
        shoebox_representation = self.shoebox_encoder(standardized_counts.reshape(shoeboxes_batch.shape[0], 1, 3, 21, 21), mask=dead_pixel_mask_batch)
        print("shoebox representation:", shoebox_representation.shape)

        metadata_representation = self.metadata_encoder(self._cut_metadata(metadata_batch).float()) # (batch_size, dmodel)
        print("metadata representation (batch size, dmodel)", metadata_representation.shape)

        joined_shoebox_representation = shoebox_representation + metadata_representation

        image_representation = torch.max(joined_shoebox_representation, dim=0, keepdim=True)[0] # (1, dmodel)
        print("image rep (1, dmodel)", image_representation.shape)

        if torch.isnan(image_representation).any():
            raise ValueError("MLP image_representation produced NaNs!")
        if torch.isnan(metadata_representation).any():
            raise ValueError("MLP metadata_representation produced NaNs!")

        scale_distribution = self.compute_scale(image_representation=image_representation, metadata_representation=metadata_representation) #  torch.distributions.Normal instance
        shoebox_profile = self.compute_shoebox_profile(shoebox_representation=shoebox_representation, image_representation=image_representation)
        background_distribution = self.compute_background_distribution(shoebox_representation=shoebox_representation)
        print("hkl batch", hkl_batch)
        self.surrogate_posterior.update_observed(rasu_id=self.surrogate_posterior.rac.rasu_ids[0], H=hkl_batch)

        photon_rate = self.compute_photon_rate(
            scale_distribution=scale_distribution,
            background_distribution=background_distribution,
            profile=shoebox_profile,
            surrogate_posterior=self.surrogate_posterior,
            ordered_miller_indices=hkl_batch,
            log_images=log_images
        )
        if log_images:
            return (shoeboxes_batch, photon_rate, hkl_batch, dials_reference_batch)


        return (counts_batch, dead_pixel_mask_batch, hkl_batch, photon_rate, background_distribution, self.surrogate_posterior, 
                self.lrmvn_distribution_for_shoebox_profile.mvn_mean_distribution, self.lrmvn_distribution_for_shoebox_profile.mvn_cov_factor_distribution, self.lrmvn_distribution_for_shoebox_profile.mvn_cov_scale_distribution)
    
    def training_step(self, batch):
        output = self(batch=batch)
        loss_output = self.loss_function(*output)

        self.logger.experiment.log({
                "loss/kl_structure_factors": loss_output.kl_structure_factors.item(),
                "loss/kl_background": loss_output.kl_background.item(),
                "loss/kl_profile": loss_output.kl_profile.item(),
                "loss/log_likelihood": loss_output.log_likelihood.item(),
            })

        self.loss = loss_output.loss
        # self.logger.experiment.log({"train_loss": self.loss})
        self.log("train/loss_step", self.loss, on_step=True, on_epoch=True)
        norms = grad_norm(self, norm_type=2)
        for name, norm in norms.items():
            self.log(f"grad_norm/{name}", norm)

        print("Loss: ", self.loss)
        return self.loss
    
    def validation_step(self, batch, batch_idx):
        output = self(batch=batch)
        loss_output = self.loss_function(*output)

        self.log("validation_loss/kl_structure_factors", loss_output.kl_structure_factors.item(), on_step=False, on_epoch=True)
        self.log("validation_loss/kl_background", loss_output.kl_background.item(), on_step=False, on_epoch=True)
        self.log("validation_loss/kl_profile", loss_output.kl_profile.item(), on_step=False, on_epoch=True)
        self.log("validation_loss/log_likelihood", loss_output.log_likelihood.item(), on_step=False, on_epoch=True)
        # print("Validation Loss: ", loss_output.loss.item())
        self.log("validation_loss/loss", loss_output.loss.item(), on_step=False, on_epoch=True)

        return loss_output.loss

    def configure_optimizers(self):
        return self.optimizer
    
    def val_dataloader(self):
        return self.dataloader.load_data_set_batched_by_image(
        data_set_to_load=self.dataloader.validation_data_set,
    )

def run():
    settings = Settings()
    loss_settings = LossSettings()

    data_loader_settings = data_loader.DataLoaderSettings(data_directory=settings.data_directory,
                                data_file_names=settings.data_file_names,
                                )
    dataloader = data_loader.CrystallographicDataLoader(settings=data_loader_settings)
    print("Loading data ...")
    dataloader.load_data_()
    print("Data loaded successfully.")

    model = Model(settings=settings, loss_settings=loss_settings, dataloader=dataloader)
    if settings.enable_checkpointing:
        checkpoint_callback = ModelCheckpoint(
            monitor="validation_loss/loss",
            save_top_k=1,
            mode="min",
            filename="best-{epoch:02d}-{validation_loss/loss:.2f}"
        )

    trainer = L.pytorch.Trainer(
        logger=wandb_logger, 
        max_epochs=300, 
        log_every_n_steps=5, 
        val_check_interval=3, 
        accelerator="auto", 
        enable_checkpointing=settings.enable_checkpointing, 
        callbacks=[Plotting(), LossLogging(), CorrelationPlotting(), ScalePlotting(), checkpoint_callback]
    )
    trainer.fit(model, train_dataloaders=dataloader.load_data_set_batched_by_image(
        data_set_to_load=dataloader.train_data_set,
    ))
    if settings.enable_checkpointing and checkpoint_callback.best_model_path:
        artifact = wandb.Artifact('best_model', type='model')
        artifact.add_file(checkpoint_callback.best_model_path)
        wandb_logger.experiment.log_artifact(artifact)
    else:
        print("No checkpoint to log — possibly because no validation occurred or no model was saved.")



def main():
    run()

if __name__ == "__main__":
    main()

        