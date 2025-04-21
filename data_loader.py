"""To load shoeboxes batched by image.
Call with 

python data_loader.py --data_directory <path_to_data_directory>.

"""

import torch
import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Sampler
import itertools
import random
import dataclasses
import argparse

@ dataclasses.dataclass
class DataLoaderSettings():
    data_directory: str 
    data_file_names: dict = dataclasses.field(default_factory=lambda: {
        "shoeboxes": "shoebox_subset.pt",
        "counts": "raw_counts_subset.pt",
        "metadata": "metadata_subset.pt",
        "masks": "mask_subset.pt",
        "true_reference": "true_reference_subset.pt",
    })
    metadata_indices: dict = dataclasses.field(default_factory=lambda: {
        "h": 0,
        "k": 1,
        "l": 2,
        "d": 3,
        "x": 4,
        "y": 5,
        "z": 6,
    })
    metadata_keys_to_keep: list = dataclasses.field(default_factory=lambda: [
        "x", "y"
    ])

    validation_set_split: float = 0.2
    test_set_split: float = 0.5
    number_of_images_per_batch: int = 20
    number_of_shoeboxes_per_batch: int = 16
    number_of_batches: int = 100
    number_of_workers: int = 3
    pin_memory: bool = True
    prefetch_factor: int|None = 2
    shuffle_indices: bool = True
    shuffle_groups: bool = True
    optimize_shoeboxes_per_batch: bool = True
    append_image_id_to_metadata: bool = False
    verbose: bool = True


class CrystallographicDataLoader():
    full_data_set: torch.utils.data.dataset.Subset
    train_data_set: torch.utils.data.dataset.Subset
    validation_data_set: torch.utils.data.dataset.Subset
    test_data_set: torch.utils.data.dataset.Subset
    settings: DataLoaderSettings

    def __init__(
            self, 
            settings: DataLoaderSettings
        ):
        self.settings = settings

    def _get_raw_shoebox_data_(self):
        metadata = torch.load(
            os.path.join(self.settings.data_directory, self.settings.data_file_names["metadata"]), weights_only=True
        )
        if len(self.settings.metadata_keys_to_keep) < len(self.settings.metadata_indices):
            metadata = self._cut_metadata(metadata)
        print("Metadata shape:", metadata.shape)

        counts = torch.load(
            os.path.join(self.settings.data_directory, self.settings.data_file_names["counts"]), weights_only=True
        )
        print("counts shape:", counts.shape)
        dead_pixel_mask = torch.load(
            os.path.join(self.settings.data_directory, self.settings.data_file_names["masks"]), weights_only=True
        )
        print("dead_pixel_mask shape:", dead_pixel_mask.shape)
        shoeboxes = torch.load(
            os.path.join(self.settings.data_directory, self.settings.data_file_names["shoeboxes"]), weights_only=True
        )
        print("shoeboxes shape:", shoeboxes.shape)
        self.full_data_set = TensorDataset(
            shoeboxes, metadata, dead_pixel_mask, counts
        )

    def append_image_id_to_metadata_(self) -> None:
            image_ids = self._get_image_ids_from_shoeboxes(
                shoebox_data_set=self.full_data_set.tensors[0]
            )
            data_as_list = list(self.full_data_set.tensors)
            data_as_list[1] = torch.cat(
                (data_as_list[1], torch.tensor(image_ids).unsqueeze(1)), dim=1
            )
            self.full_data_set.tensors = tuple(data_as_list)
    
    def _cut_metadata(self, metadata: torch.Tensor) -> torch.Tensor:
        indices_to_keep = torch.tensor([self.settings.metadata_indices[i] for i in self.settings.metadata_keys_to_keep], dtype=torch.long)
        return torch.index_select(metadata, dim=1, index=indices_to_keep)
    

    def _clean_data_(self):
        pass

    def _split_full_data_(self) -> None:
        full_data_set_length = len(self.full_data_set)
        validation_data_set_length = int(
            full_data_set_length * self.settings.validation_set_split
        )
        test_data_set_length = int(full_data_set_length * self.settings.test_set_split)
        train_data_set_length = (
            full_data_set_length - validation_data_set_length - test_data_set_length
        )
        self.train_data_set, self.validation_data_set, self.test_data_set = torch.utils.data.random_split(
            self.full_data_set,
            [train_data_set_length, validation_data_set_length, test_data_set_length],
            generator=torch.Generator().manual_seed(42)
        )

    def load_data_(self) -> None:
        self._get_raw_shoebox_data_()
        if self.settings.append_image_id_to_metadata:
            self.append_image_id_to_metadata_()
        self._clean_data_()
        self._split_full_data_()

    def _get_image_ids_from_shoeboxes(self, shoebox_data_set: torch.Tensor) -> list:
        """Returns the list of respective image ids for 
            all shoeboxes in the shoebox data set."""
        
        image_ids = []
        for shoebox in shoebox_data_set:
            minimum_dz_index = torch.argmin(shoebox[:, 5]) # 5th index is dz
            image_ids.append(shoebox[minimum_dz_index, 2].item()) # 2nd index is z
        
        if len(image_ids) != len(shoebox_data_set):
            print(len(image_ids), len(shoebox_data_set))
            raise ValueError(
                f"The number of shoeboxes {len(shoebox_data_set)} does not match the number of image ids {len(image_ids)}."
            )
        return image_ids
        
    def _map_images_to_shoeboxes(self, shoebox_data_set: torch.Tensor) -> dict:
        """Returns a dictionary with image ids as keys and indices of all shoeboxes 
            belonging to that image as values."""
        
        image_ids = self._get_image_ids_from_shoeboxes(
            shoebox_data_set=shoebox_data_set
        )
        images_to_shoebox_indices = {}
        for shoebox_index, image_id in enumerate(image_ids):
            if image_id not in images_to_shoebox_indices:
                images_to_shoebox_indices[image_id] = []
            images_to_shoebox_indices[image_id].append(shoebox_index)
        return images_to_shoebox_indices


    class BatchByImageSampler(Sampler):
        
        image_id_to_indices: dict
        settings: DataLoaderSettings

        def __init__(self, 
                     image_id_to_indices: dict, 
                     settings: DataLoaderSettings,
            ) -> None: 
            self.image_id_to_indices = image_id_to_indices
            self.settings = settings
            self.batches = self._get_batches()
            

        def _list_of_shoebox_indices_by_image(self, number_of_images_per_batch: int) -> list:
            shoebox_indices = list(itertools.chain.from_iterable(
                    random.sample(
                        list(self.image_id_to_indices.values()), 
                        number_of_images_per_batch
                    )
                )
            )
            if self.settings.shuffle_indices:
                random.shuffle(shoebox_indices)
            return shoebox_indices
        
        def _get_batches(
                self,
            ) -> list:
            batches = []
            if self.settings.optimize_shoeboxes_per_batch:
                average_shoeboxes_per_image = (
                    (sum(len(shoeboxes) for shoeboxes in self.image_id_to_indices.values()
                        ) // len(self.image_id_to_indices))
                )
                number_of_images_required = -(-self.settings.number_of_shoeboxes_per_batch//average_shoeboxes_per_image)

                if self.settings.verbose:
                    print("Average shoeboxes per image:", average_shoeboxes_per_image)
                    print("Number of images per batch:", number_of_images_required)
                if number_of_images_required > len(self.image_id_to_indices):
                    raise ValueError(
                        f"The number of images required = {number_of_images_required} "
                        f"is larger than the number of images = {len(self.image_id_to_indices)}."
                    )
                for _ in range(self.settings.number_of_batches):
                    batch = self._list_of_shoebox_indices_by_image(
                                    number_of_images_per_batch=number_of_images_required
                                        )[:self.settings.number_of_shoeboxes_per_batch]
                    while len(batch) < self.settings.number_of_shoeboxes_per_batch:
                        batch += self._list_of_shoebox_indices_by_image(
                                    number_of_images_per_batch=1
                                        )[:self.settings.number_of_shoeboxes_per_batch - len(batch)]
                    batches.append(batch)

                return batches
            else:
                return [self._list_of_shoebox_indices_by_image(
                    number_of_images_per_batch=self.settings.number_of_images_per_batch
                        ) 
                        for _ in range(self.settings.number_of_batches)]
                    

        def __iter__(self):
            if self.settings.shuffle_groups:
                random.shuffle(self.batches)
            for batch in self.batches:
                yield batch

    def load_data_set_batched_by_image(self, 
                                       data_set_to_load: torch.utils.data.dataset.Subset | torch.utils.data.TensorDataset,
        ) -> torch.utils.data.dataloader.DataLoader:

        if isinstance(data_set_to_load, torch.utils.data.dataset.Subset):
            image_id_to_indices = self._map_images_to_shoeboxes(
                shoebox_data_set=self.full_data_set.tensors[0][data_set_to_load.indices]
            )
        else:
            image_id_to_indices = self._map_images_to_shoeboxes(
                shoebox_data_set=self.full_data_set.tensors[0]
            )

        batch_by_image_sampler = self.BatchByImageSampler(
            image_id_to_indices=image_id_to_indices,
            settings=self.settings
            )
        return DataLoader(
            data_set_to_load,
            batch_sampler=batch_by_image_sampler,
            num_workers=self.settings.number_of_workers,
            pin_memory=self.settings.pin_memory,
            prefetch_factor=self.settings.prefetch_factor,
        )
    
def test(settings: DataLoaderSettings):
    print("enter test")
    dataloader = CrystallographicDataLoader(settings=settings)
    print("Loading data ...")	
    dataloader.load_data_()
    print("Data loaded successfully.")
        
    test_data = dataloader.load_data_set_batched_by_image(
        data_set_to_load=dataloader.test_data_set
    )
    print("Test data loaded successfully.")
    for batch in test_data:
        shoeboxes_batch, metadata_batch, dead_pixel_mask_batch, counts_batch = batch
        print("Batch shoeboxes shape:", shoeboxes_batch.shape)
    return test_data

def parse_args():
    parser = argparse.ArgumentParser(description="Pass DataLoader settings")
    parser.add_argument(
        "--data_directory",
        type=str,
        help="Path to the data directory",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    settings = DataLoaderSettings(data_directory=args.data_directory, test_set_split=0.91)
    test(settings=settings)

if __name__ == "__main__":
    main()
