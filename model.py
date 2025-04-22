
import data_loader
import torch
import numpy as np
import dataclasses
import encoder
import torch.nn.functional as F
from networks import *
import distributions
import loss_function

# import integrator
# from integrator import data_loaders
# from integrator.model.encoders import cnn_3d, fc_encoder

@dataclasses.dataclass
class Settings():
    background_distribution = distributions.Distribution()
    profile_distribution = distributions.Distribution()
    optimizer = torch.optim.AdamW
    batch_size = 10
    number_of_epochs = 10
    number_of_mc_samples = 100
    data_directory = "/n/hekstra_lab/people/aldama/subset"
    data_file_names = {
    "shoeboxes": "standardized_shoeboxes_subset.pt",
    "counts": "raw_counts_subset.pt",
    "metadata": "shoebox_features_subset.pt",
    "masks": "masks_subset.pt",
    "true_reference": "metadata_subset.pt",
    }

class Model():
    def __init__(self, settings: Settings):
        self.settings = settings

    def compute_scale(self, image_representation, metadata_representation):
        joined_representation = self._add_representation_(image_representation, metadata_representation)
        return MLPScale(joined_representation)
    
    def _add_representation_(self, representation1, representation2):
        return representation1 + representation2
    
    def compute_shoebox_profile(self, shoebox_representation, image_representation):
        profile_mvn_model = distributions.LRMVN_Distribution(
            batch_size=self.settings.batch_size,
        )
        profile_mvn_distribution = profile_mvn_model(
            shoebox_representation, image_representation
        )
        log_probs = profile_mvn_distribution.log_prob(
            self.pixel_positions.expand(self.batch_size, 1, 1323, 3)
        )

        log_probs_stable = log_probs - log_probs.max(dim=-1, keepdim=True)[0]

        profile = torch.exp(log_probs_stable)

        avg_profile = profile.mean(dim=1)
        avg_profile = avg_profile / (avg_profile.sum(dim=-1, keepdim=True) + 1e-10)

        return profile

    def compute_background_distribution(self, shoebox_representation):
        return self.settings.background_distribution(shoebox_representation)

    def compute_photon_rate(self, background_distribution, profile, intensity_distribution):  
        # background_distribution: self.settings.background_distribution(shoebox_representation) instance, profile: torch.exp(log_probs_stable) inastance )is that a distribution?)
        # intensity distibution: sth alike self.qI(shoebox_representation, image_representation)
        samples_intensity = intensity_distribution.rsample([self.number_of_mc_samples]).unsqueeze(-1).permute(1, 0, 2)
        samples_background = background_distribution.rsample([self.number_of_mc_samples]).permute(1, 0, 2)
        samples_profile = profile / profile.sum(dim=-1, keepdim=True) + 1e-10

        return samples_intensity * samples_profile + samples_background  # [batch_size, mc_samples, pixels]

    def compute_structure_factors(self):
        pass

    def compute_intensity_distribution(self, representation):
        return self.settings.background_distribution(representation)

    def forward(self, batch):
        shoeboxes_batch, metadata_batch, dead_pixel_mask_batch, counts_batch = batch
        shoebox_encoder = encoder.CNN_3d()
        # x = torch.clamp(shoeboxes_batch, 0, 1)
        shoebox_representation = shoebox_encoder(shoeboxes_batch)

        metadata_encoder = encoder.MLPMetadataEncoder()
        metadata_representation = metadata_encoder(metadata_batch)

        joined_shoebox_representation = shoebox_representation + metadata_representation

        image_representation = torch.max(joined_shoebox_representation, dim=0, keepdim=True)[0]

        scale = self.compute_scale(image_representation=image_representation, metadata_representation=metadata_representation)
        shoebox_profile = self.compute_shoebox_profile(shoebox_representation=shoebox_representation, image_representation=image_representation)
        background_distribution = self.compute_background_distribution(shoebox_representation=shoebox_representation)
        intensity_distribution = self.compute_intensity_distribution(shoebox_representation=shoebox_representation)

        # predicted_structure_factors = self.compute_structure_factors()

        photon_rate = self.compute_photon_rate(
            background_distribution=background_distribution,
            profile=shoebox_profile,
            intensity_distribution=intensity_distribution,
        )

        return None # outputs


def run():
    settings = Settings()

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
    model = Model(settings=settings)
    for epoch in range(settings.number_of_epochs):
        for batch in train_data:
            settings.optimizer.zero_grad()
            output = model(batch=batch)
            loss = loss_function.LossFunction(output)
            loss.backward()
            settings.optimizer.step()
            print(f"Epoch {epoch}, Loss: {loss.item()}")
        
    print("Training completed.")

def main():
    model = Model()
    model.run()

if __name__ == "__main__":
    main()

        