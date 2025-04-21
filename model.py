
import data_loader
import torch
import numpy as np
import dataclasses
import encoder
# from integrator import data_loaders
# from integrator.model.encoders import cnn_3d, fc_encoder

@dataclasses.dataclass
class ModelSettings():
    data_directory = "/n/hekstra_lab/people/aldama/subset"
    data_file_names = {
    "shoeboxes": "standardized_shoeboxes_subset.pt",
    "counts": "raw_counts_subset.pt",
    "metadata": "shoebox_features_subset.pt",
    "masks": "masks_subset.pt",
    "true_reference": "metadata_subset.pt",
    }

class NormalDistributionLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bijector = torch.nn.Softplus()

    def forward(self, hidden_representation):
        loc, scale = torch.unbind(hidden_representation, dim=-1)
        scale = self.bijector(scale)
        return torch.distributions.Normal(loc=loc, scale=scale)

class MLPScale(torch.nn.Module):
    def __init__(self, hidden_dim=64, input_dim=7, number_of_layers=2):
        super().__init__()
        activation = torch.nn.ReLU()
        self.hidden_dimension 
        self.add_bias = True
        self._buid_mlp_(number_of_layers=number_of_layers)
        
    def _build_mlp_(self, number_of_layers):
        mlp_layers = []
        for i in range(number_of_layers):
            mlp_layers.append(torch.nn.Linear(
                in_features=self.input_dim if i == 0 else self.hidden_dim,
                out_features=self.hidden_dim,
                bias=self.add_bias,
            ))
            mlp_layers.append(self.activation)
        self.network = torch.nn.Sequential(*mlp_layers)

        map_to_distribution_layers = []
        map_to_distribution_layers.append(torch.nn.Linear(
            in_features=self.hidden_dim,
            out_features=2,
            bias=self.add_bias,
        ))
        map_to_distribution_layers.append(NormalDistributionLayer())

        self.distribution = torch.nn.Sequential(*map_to_distribution_layers)

    def forward(self, x):
        return self.distribution(self.network(x))

class Model():
    def __init__(self):

        pass

    def compute_scale(self, image_representation, metadata_representation):
        joined_representation = self._add_representation_(image_representation, metadata_representation)
        return MLPScale(joined_representation)
    
    def _add_representation_(self, representation1, representation2):
        return representation1 + representation2


    def compute_background(self, shoebox_representation, metadata_representation):
        joined_representation = self._add_representation_(shoebox_representation, metadata_representation)
        pass

    def compute_shoebox_profile(self, shoebox_representation, metadata_representation):
        pass

    def compute_structure_factors(self):
        pass

    def compute_intensity(self):
        pass

    def compute_photon_rate(self, predicted_structure_factors, background, profile, scale):
        pass

    def run(self):
        model_settings = ModelSettings()

        data_loader_settings = data_loader.DataLoaderSettings(data_directory=model_settings.data_directory,
                                    data_file_names=model_settings.data_file_names,
                                    )
        dataloader = data_loader.CrystallographicDataLoader(settings=data_loader_settings)
        print("Loading data ...")	
        dataloader.load_data_()
        print("Data loaded successfully.")
            
        train_data = dataloader.load_data_set_batched_by_image(
            data_set_to_load=dataloader.train_data_set,
        )
        print("Train data loaded successfully.")
        for batch in train_data:
            shoeboxes_batch, metadata_batch, dead_pixel_mask_batch, counts_batch = batch
            shoebox_encoder = encoder.CNN_3d()
            # x = torch.clamp(shoeboxes_batch, 0, 1)
            shoebox_representation = shoebox_encoder(shoeboxes_batch)

            metadata_encoder = encoder.MLPMetadataEncoder()
            metadata_representation = metadata_encoder(metadata_batch)

            joined_shoebox_representation = shoebox_representation + metadata_representation

            image_representation = torch.max(joined_shoebox_representation, dim=0, keepdim=True)[0]

            scale = self.compute_scale(image_representation, metadata_representation)
            background = self.compute_background(shoebox_representation, metadata_representation)
            shoebox_profile = self.compute_shoebox_profile(shoebox_representation, metadata_representation)
            intensity = self.compute_intensity()

            predicted_structure_factors = self.compute_structure_factors()

            photon_rate = self.compute_photon_rate(predicted_structure_factors, background, profile, scale)

            print("run for batch completed")


def main():
    model = Model()
    model.run()

if __name__ == "__main__":
    main()

        