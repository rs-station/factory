import sys, os

# __file__ is .../factory/factory/model.py
repo_root = os.path.abspath(os.path.join(__file__, os.pardir))
inner_pkg = os.path.join(repo_root, "abismal_torch")   # ← now points to factory/factory/abismal_torch

# insert both levels:
sys.path.insert(0, inner_pkg)
sys.path.insert(0, repo_root)



import data_loader
import torch
import numpy as np
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


class Model(torch.nn.Module):

    shoebox_encoder: encoder.CNN_3d
    metadata_encoder: encoder.MLPMetadataEncoder
    scale_model: MLPScale
    lrmvn_distribution_for_shoebox_profile: distributions.LRMVN_Distribution
    background_distribution: distributions.HalfNormalDistribution
    prior_intensity_distribution: WilsonPrior
    surrogate_posterior: FoldedNormalPosterior

    def __init__(self, settings: Settings):
        super().__init__()
        self.settings = settings
        self.shoebox_encoder = encoder.CNN_3d()
        self.metadata_encoder = encoder.MLPMetadataEncoder()
        self.scale_model = MLPScale(input_dimension=self.settings.dmodel)
        self.lrmvn_distribution_for_shoebox_profile = self.settings.build_shoebox_profile_distribution()
        self.background_distribution = self.settings.build_background_distribution(dmodel=self.settings.dmodel)
        self.rac = self.build_rac()
        self.prior_intensity_distribution = WilsonPrior(self.rac)
        # print("rac size", self.rac.size)
        self.surrogate_posterior = self.compute_intensity_distribution()
        

        

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

    def compute_photon_rate(self, scale_distribution, background_distribution, profile, intensity_distribution):  
        samples_scale = scale_distribution.rsample([self.settings.number_of_mc_samples]).unsqueeze(-1).permute(1, 0, 2)
        print("shape scale", samples_scale.shape) #(batch_size, number_of_samples, 1)
        samples_intensity = intensity_distribution.rsample([self.settings.number_of_mc_samples]).unsqueeze(-1).permute(1, 0, 2)
        print("shape intensity", samples_intensity.shape)
        samples_background = background_distribution.rsample([self.settings.number_of_mc_samples]).permute(1, 0, 2)
        print("shape backgr", samples_background.shape)
        samples_profile = profile / profile.sum(dim=-1, keepdim=True) + 1e-10
        print("profile shape", samples_profile.shape)

        # return samples_scale * samples_intensity * samples_profile + samples_background  # [batch_size, mc_samples, pixels]

        return samples_scale * samples_profile + samples_background  # [batch_size, mc_samples, pixels]

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

    def forward(self, batch):
        shoeboxes_batch, metadata_batch, dead_pixel_mask_batch, counts_batch = batch
        # shoebox_encoder = encoder.CNN_3d()
        # x = torch.clamp(shoeboxes_batch, 0, 1)
        shoebox_representation = self.shoebox_encoder(shoeboxes_batch) # (batch_size, dmodel)
        print("shoebox representation:", shoebox_representation.shape)

        # metadata_encoder = encoder.MLPMetadataEncoder()
        metadata_representation = self.metadata_encoder(self._cut_metadata(metadata_batch).float()) # (batch_size, dmodel)
        print("metadata representation (batch size, dmodel)", metadata_representation.shape)

        joined_shoebox_representation = shoebox_representation + metadata_representation

        image_representation = torch.max(joined_shoebox_representation, dim=0, keepdim=True)[0] # (1, dmodel)
        print("image rep (1, dmodel)", image_representation.shape)

        scale_distribution = self.compute_scale(image_representation=image_representation, metadata_representation=metadata_representation) #  torch.distributions.Normal instance
        shoebox_profile = self.compute_shoebox_profile(shoebox_representation=shoebox_representation, image_representation=image_representation)
        background_distribution = self.compute_background_distribution(shoebox_representation=shoebox_representation)
        intensity_distribution = self.surrogate_posterior #(shoebox_representation)

        photon_rate = self.compute_photon_rate(
            scale_distribution=scale_distribution,
            background_distribution=background_distribution,
            profile=shoebox_profile,
            intensity_distribution=intensity_distribution,
        )
        return (counts_batch, dead_pixel_mask_batch, photon_rate, background_distribution, intensity_distribution, 
                self.lrmvn_distribution_for_shoebox_profile.mvn_mean_distribution, self.lrmvn_distribution_for_shoebox_profile.mvn_cov_factor_distribution, self.lrmvn_distribution_for_shoebox_profile.mvn_cov_scale_distribution)

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
        
    train_data = dataloader.load_data_set_batched_by_image(
        data_set_to_load=dataloader.train_data_set,
    )
    print("Train data loaded successfully.")
    loss_function = loss_functionality.LossFunction(settings=settings, loss_settings=loss_settings)
    model = Model(settings=settings)
    optimizer = settings.optimizer(model.parameters())
    for epoch in range(settings.number_of_epochs):
        for batch in train_data:
            optimizer.zero_grad()
            output = model(batch=batch)
            loss = loss_function(*output)
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch}, Loss: {loss.item()}")
        
    print("Training completed.")


def main():
    run()

if __name__ == "__main__":
    main()

        