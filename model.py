import sys, os

repo_root = os.path.abspath(os.path.join(__file__, os.pardir))
inner_pkg = os.path.join(repo_root, "abismal_torch")   # ← now points to factory/factory/abismal_torch

sys.path.insert(0, inner_pkg)
sys.path.insert(0, repo_root)



import data_loader
import torch
import numpy as np
import lightning as L
import dataclasses
import encoder
import torch.nn.functional as F
from networks import *
import distributions
import loss_functionality
import get_protein_data
from abismal_torch.prior import WilsonPrior
from abismal_torch.symmetry.reciprocal_asu import ReciprocalASU, ReciprocalASUGraph
from abismal_torch.likelihood import NormalLikelihood
from abismal_torch.surrogate_posterior import FoldedNormalPosterior
from abismal_torch.merging import VariationalMergingModel
from abismal_torch.scaling import ImageScaler

from lightning.pytorch.loggers import WandbLogger
from lightning import Callback
import wandb
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
    # intensity_distribution_initial_location: float
    # intensity_distribution_initial_scale: float
    mvn_mean_prior_distribution = torch.distributions.normal.Normal(loc=torch.tensor(0.0), scale=torch.tensor(5.0))
    mvn_cov_factor_prior_distribution = torch.distributions.normal.Normal(loc=torch.tensor(0.0), scale=torch.tensor(0.5))
    mvn_cov_scale_prior_distribution = torch.distributions.half_normal.HalfNormal(scale=torch.tensor(0.3))
    optimizer = torch.optim.AdamW
    dmodel = 64
    batch_size = 4
    number_of_epochs = 5
    number_of_mc_samples = 100

    data_directory = r"C:\Users\Flavi\Desktop\Thesis_Data" #"/n/hekstra_lab/people/aldama/subset"
    data_file_names: dict = dataclasses.field(default_factory=lambda: {
        "shoeboxes": "standardized_shoeboxes_subset.pt",
        "counts": "raw_counts_subset.pt",
        "metadata": "shoebox_features_subset.pt",
        "masks": "masks_subset.pt",
        "true_reference": "metadata_subset.pt",
    })
    protein_pdb_url = "https://files.rcsb.org/download/9B7C.cif"

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
    shoebox_encoder: encoder.CNN_3d
    metadata_encoder: encoder.MLPMetadataEncoder
    scale_model: MLPScale
    lrmvn_distribution_for_shoebox_profile: distributions.LRMVN_Distribution
    background_distribution: distributions.HalfNormalDistribution
    prior_intensity_distribution: WilsonPrior
    surrogate_posterior: FoldedNormalPosterior

    def __init__(self, settings: Settings, loss_settings: LossSettings, dataloader: data_loader.CrystallographicDataLoader):
        super().__init__()
        self.settings = settings
        self.dataloader = dataloader
        self.shoebox_encoder = encoder.CNN_3d()
        self.metadata_encoder = encoder.MLPMetadataEncoder()
        self.scale_model = MLPScale(input_dimension=self.settings.dmodel)
        self.lrmvn_distribution_for_shoebox_profile = self.settings.build_shoebox_profile_distribution()
        self.background_distribution = self.settings.build_background_distribution(dmodel=self.settings.dmodel)
        self.rac = self.build_rac()
        self.prior_intensity_distribution = WilsonPrior(self.rac)
        # print("rac size", self.rac.size)
        self.surrogate_posterior = self.compute_intensity_distribution()
        self.optimizer = self.settings.optimizer(self.parameters())
        self.loss_function = loss_functionality.LossFunction(settings=settings, loss_settings=loss_settings)


    def compute_scale(self, image_representation, metadata_representation) -> torch.distributions.Normal:
        joined_representation = self._add_representation_(image_representation, metadata_representation) # (batch_size, dmodel)
        return self.scale_model(joined_representation) # a  torch.distributions.Normal object
    
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
        print("shape intensity", samples_surrogate_posterior.shape)

        samples_scale = scale_distribution.rsample([self.settings.number_of_mc_samples]).unsqueeze(-1).permute(1, 0, 2)
        print("shape scale", samples_scale.shape) #(batch_size, number_of_samples, 1)
        samples_profile = profile / profile.sum(dim=-1, keepdim=True) + 1e-10
        print("profile shape", samples_profile.shape) # (batch_size, number of samples, pixels)

        samples_background = background_distribution.rsample([self.settings.number_of_mc_samples]).permute(1, 0, 2)
        print("shape backgr", samples_background.shape)

        print("ordered millder indices", ordered_miller_indices.shape)

        print("indices type", ordered_miller_indices)
        samples_predicted_structure_factor = self.surrogate_posterior.rac.reciprocal_asus[0].gather(
            source=samples_surrogate_posterior.T, H=ordered_miller_indices.to(torch.int)
        ).unsqueeze(-1)
        print("gathered structure factor samples", samples_predicted_structure_factor.shape)

        if log_images:
            return (samples_scale * torch.square(samples_predicted_structure_factor) * samples_profile + samples_background,
                    samples_profile)
        else:
            return samples_scale * torch.square(samples_predicted_structure_factor) * samples_profile + samples_background  # [batch_size, mc_samples, pixels]

        # return samples_scale * samples_profile + samples_background  # [batch_size, mc_samples, pixels]

    def compute_structure_factors(self):
        pass

    def build_rac(self):
        pdb_data = get_protein_data.get_protein_data(self.settings.protein_pdb_url)
        rasus = [ReciprocalASU(
            cell=pdb_data["unit_cell"],
            spacegroup=pdb_data["spacegroup"],
            dmin=float(pdb_data["dmin"]),
            anomalous=True,
        )]
        return ReciprocalASUGraph(*rasus)

    def compute_intensity_distribution(self):
        initial_mean = self.prior_intensity_distribution.distribution().mean() # self.settings.intensity_distribution_initial_location
        print("intensity folded normal mean shape", initial_mean.shape)
        initial_scale = 0.05 * initial_mean # self.settings.intensity_distribution_initial_scale
        surrogate_posterior = FoldedNormalPosterior.from_unconstrained_loc_and_scale(
            rac=self.rac, loc=initial_mean, scale=initial_scale # epsilon=settings.epsilon
        )

        #####
        def check_grad_hook(grad):
            if grad is not None:
                print(f"Gradient stats: min={grad.min()}, max={grad.max()}, mean={grad.mean()}, any_nan={torch.isnan(grad).any()}, all_finite={torch.isfinite(grad).all()}")
            return grad
        surrogate_posterior.distribution.loc.register_hook(check_grad_hook)
        ###

        return surrogate_posterior
    
    def _cut_metadata(self, metadata) -> torch.Tensor:
        return torch.index_select(metadata, dim=1, index=torch.tensor([4,5]))

    def forward(self, batch, log_images=False):
        shoeboxes_batch, metadata_batch, dead_pixel_mask_batch, counts_batch, hkl_batch, dials_reference_batch = batch

        shoebox_representation = self.shoebox_encoder(shoeboxes_batch) # (batch_size, dmodel)
        print("shoebox representation:", shoebox_representation.shape)

        metadata_representation = self.metadata_encoder(self._cut_metadata(metadata_batch).float()) # (batch_size, dmodel)
        print("metadata representation (batch size, dmodel)", metadata_representation.shape)

        joined_shoebox_representation = shoebox_representation + metadata_representation

        image_representation = torch.max(joined_shoebox_representation, dim=0, keepdim=True)[0] # (1, dmodel)
        print("image rep (1, dmodel)", image_representation.shape)

        scale_distribution = self.compute_scale(image_representation=image_representation, metadata_representation=metadata_representation) #  torch.distributions.Normal instance
        shoebox_profile = self.compute_shoebox_profile(shoebox_representation=shoebox_representation, image_representation=image_representation)
        background_distribution = self.compute_background_distribution(shoebox_representation=shoebox_representation)
        surrogate_posterior = self.surrogate_posterior 

        photon_rate = self.compute_photon_rate(
            scale_distribution=scale_distribution,
            background_distribution=background_distribution,
            profile=shoebox_profile,
            surrogate_posterior=surrogate_posterior,
            ordered_miller_indices=hkl_batch,
            log_images=log_images
        )
        if log_images:
            return (shoeboxes_batch, photon_rate, dials_reference_batch)

        return (counts_batch, dead_pixel_mask_batch, photon_rate, background_distribution, surrogate_posterior, 
                self.lrmvn_distribution_for_shoebox_profile.mvn_mean_distribution, self.lrmvn_distribution_for_shoebox_profile.mvn_cov_factor_distribution, self.lrmvn_distribution_for_shoebox_profile.mvn_cov_scale_distribution)
    
    def training_step(self, batch):
        output = self(batch=batch)
        loss = self.loss_function(*output)
        self.logger.experiment.log({"train_loss": loss})
        return loss
    
    def configure_optimizers(self):
        return self.optimizer
    
    def val_dataloader(self):
        return self.dataloader.load_data_set_batched_by_image(
        data_set_to_load=self.dataloader.validation_data_set,
    )
    
class Plotting(Callback):

    def __init__(self):
        self.fixed_batch = None

    @staticmethod
    def make_image(image):
        fig, ax = plt.subplots()
        ax.imshow(image.cpu().numpy())  # convert CHW to HWC
        ax.axis("off")
        return fig

    def on_fit_start(self, trainer, pl_module):
        self.fixed_batch = next(iter(pl_module.dataloader.load_data_for_logging_during_training()))

    def on_train_epoch_end(self, trainer, pl_module):
        with torch.no_grad():
            shoeboxes_batch, photon_rate_output, dials_reference = pl_module(self.fixed_batch, log_images=True)
            dials_id = dials_reference[:,-1]
            photon_rate, profile = photon_rate_output
            images = []
            
            batch_size = shoeboxes_batch.size(0)
            col_names = ["type"] + [f"ID {i}" for i in dials_id]
            table     = wandb.Table(columns=col_names)

            raw_images = [wandb.Image(self.make_image(shoeboxes_batch[batch, :, -1].reshape(3, 21, 21)[1:2,:].squeeze())) for batch in range(batch_size)]
            profile_images = [wandb.Image(self.make_image(profile[batch,0,:].reshape(3,21,21)[1:2].squeeze())) for batch in range(batch_size)] # (batch_size, number of samples, pixels)
            rate_images = [wandb.Image(self.make_image(photon_rate[batch,0,:].reshape(3,21,21)[1:2].squeeze())) for batch in range(batch_size)]

            table.add_data("raw shoebox",  *raw_images)
            table.add_data("profile",      *profile_images)
            table.add_data("photon rate",  *rate_images)

            # log the table
            pl_module.logger.experiment.log({"Image Grid": table},
                                            step=trainer.global_step)            
            print("logged images")

# class Plotting(Callback):

#     def __init__(self):
#         self.fixed_batch = None

#     @staticmethod
#     def make_image(image: torch.Tensor) -> plt.Figure:
#         fig, ax = plt.subplots()
#         ax.imshow(image.cpu().numpy(), cmap="gray")  # if single-channel
#         ax.axis("off")
#         return fig

#     def on_fit_start(self, trainer, pl_module):
#         # grab one batch for visualization
#         self.fixed_batch = next(
#             iter(pl_module.dataloader.load_data_for_logging_during_training())
#         )

#     def on_train_epoch_end(self, trainer, pl_module):
#         with torch.no_grad():
#             shoeboxes, (photon_rate, profile), dials_reference = pl_module(
#                 self.fixed_batch, log_images=True
#             )
#             dials_id = dials_reference[:, -1].long()
#             images = []

#             batch_size = shoeboxes.size(0)
#             for i in range(batch_size):
#                 example_id = int(dials_id[i])

#                 # pick out the center‐slice (channel 1) of the 3×21×21 shoebox
#                 raw_slice = shoeboxes[i, 1, :, :].unsqueeze(0)
#                 prof_slice = profile[i, 0].reshape(3, 21, 21)[1].unsqueeze(0)
#                 rate_slice = photon_rate[i, 0].reshape(3, 21, 21)[1].unsqueeze(0)

#                 # turn each into a figure
#                 fig_raw  = self.make_image(raw_slice)
#                 fig_prof = self.make_image(prof_slice)
#                 fig_rate = self.make_image(rate_slice)

#                 # now, **caption goes inside** wandb.Image()
#                 images.extend([
#                     wandb.Image(fig_raw,  caption=f"raw shoebox  — ID {example_id}"),
#                     wandb.Image(fig_prof, caption=f"profile       — ID {example_id}"),
#                     wandb.Image(fig_rate, caption=f"photon rate  — ID {example_id}")
#                 ])

#             # log them all at once, including the step so WandB knows where to put them
#             pl_module.logger.experiment.log(
#                 {"Profiles": images},
#                 step=trainer.global_step
#             )

#             print("logged images")

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
    # checkpoint_callback = L.Callback.ModelCheckpoint(
    #     monitor="val_loss",
    #     save_top_k=1,
    #     mode="min",
    #     filename="best-{epoch:02d}-{val_loss:.2f}"
    # )

    trainer = L.pytorch.Trainer(logger=wandb_logger, max_epochs=5, val_check_interval=5, accelerator="auto", enable_checkpointing=False, callbacks=[Plotting()])
    trainer.fit(model, train_dataloaders=dataloader.load_data_set_batched_by_image(
        data_set_to_load=dataloader.train_data_set,
    ))
    # artifact = wandb.Artifact('best_model', type='model')
    # artifact.add_file(checkpoint_callback.best_model_path)
    # wandb_logger.experiment.log_artifact(artifact)


def main():
    run()

if __name__ == "__main__":
    main()

        