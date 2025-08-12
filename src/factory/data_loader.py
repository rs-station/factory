"""To load shoeboxes batched by image.
Call with 

python data_loader.py --data_directory <path_to_data_directory>.

"""

import torch
import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset, Subset, Dataset
from torch.utils.data import Sampler
import itertools
import random
import dataclasses
import argparse
import settings



class ShoeboxTensorDataset(Dataset):
    def __init__(self, shoeboxes, metadata, dead_pixel_mask, counts, hkl, mean, var):
        self.shoeboxes = shoeboxes
        self.metadata = metadata
        self.dead_pixel_mask = dead_pixel_mask
        self.counts = counts
        self.hkl = hkl
        self.mean = mean
        self.var = var

    def __len__(self):
        return self.shoeboxes.shape[0]

    def __getitem__(self, idx):
        # Normalize on the fly
        counts = self.counts[idx]
        normed_shoebox = torch.log1p(counts)
        # normed_shoebox = (counts - self.mean) / torch.sqrt(self.var)
        return (
            normed_shoebox,
            self.metadata[idx],
            self.dead_pixel_mask[idx],
            self.counts[idx],
            self.hkl[idx],
        )


class CrystallographicDataLoader():
    full_data_set: torch.utils.data.dataset.Subset
    train_data_set: torch.utils.data.dataset.Subset
    validation_data_set: torch.utils.data.dataset.Subset
    test_data_set: torch.utils.data.dataset.Subset
    data_loader_settings: settings.DataLoaderSettings

    def __init__(
            self, 
            data_loader_settings: settings.DataLoaderSettings
        ):
        self.data_loader_settings = data_loader_settings
        self.use_standard_sampler = False

    def _get_raw_shoebox_data_(self):
        
        # counts = torch.load(
        #     os.path.join(self.settings.data_directory, self.settings.data_file_names["counts"]), weights_only=True
        # )
        # print("counts shape:", counts.shape)

        dead_pixel_mask = torch.load(
            os.path.join(self.data_loader_settings.data_directory, self.data_loader_settings.data_file_names["masks"]), weights_only=True
        )
        print("dead_pixel_mask shape:", dead_pixel_mask.shape)
        shoeboxes = torch.load(
            os.path.join(self.data_loader_settings.data_directory, self.data_loader_settings.data_file_names["counts"]), weights_only=True
        )
        # shoeboxes = shoeboxes[:,:,-1]
        # if len(shoeboxes[0]) != 7:
        #     self.use_standard_sampler = True
        
        counts = shoeboxes.clone()
        print("standard sampler:", self.use_standard_sampler)

        stats = torch.load(os.path.join(self.data_loader_settings.data_directory,"stats.pt"), weights_only=True)
        mean = stats[0]
        var = stats[1]

        print("mean", mean)
        print("var", var)

        shoeboxes = (counts - mean) / torch.sqrt(var)

        # shoeboxes,dead_pixel_mask, counts, metadata = self._clean_shoeboxes_(shoeboxes, dead_pixel_mask, counts, metadata)
        print("shoeboxes shape:", shoeboxes.shape)
        # dials_reference = torch.load(
        #     os.path.join(self.settings.data_directory, self.settings.data_file_names["true_reference"]), weights_only=True
        # )

        # dials_reference = torch.load(
        #     "/n/hekstra_lab/people/aldama/subset/small_dataset/pass1/reference.pt", weights_only=True
        # )
        # print("dials refercne shape", dials_reference.shape)


        metadata = torch.load(
            os.path.join(self.data_loader_settings.data_directory, self.data_loader_settings.data_file_names["metadata"]), weights_only=True
        )



        print("Metadata shape:", metadata.shape) #(d, h,k, l ,x, y, z)
        # metadata = torch.zeros(shoeboxes.shape[0], 7)

        # hkl = metadata[:,1:4].to(torch.int)
        # print("hkl shape", hkl.shape)

        hkl = metadata[:,6:9].to(torch.int)
        print("hkl shape", hkl.shape)

        # Use custom dataset for on-the-fly normalization
        self.full_data_set = ShoeboxTensorDataset(
            shoeboxes=shoeboxes,
            metadata=metadata,
            dead_pixel_mask=dead_pixel_mask,
            counts=counts,
            hkl=hkl,
            mean=mean,
            var=var,
        )
        print("Metadata shape full tensor:", self.full_data_set.metadata.shape) #(d, h,k, l ,x, y, z)

        # print("concentration shape", torch.load(
        #     os.path.join(self.settings.data_directory, "concentration.pt"), weights_only=True
        # ).shape)

    def append_image_id_to_metadata_(self) -> None:
            image_ids = self._get_image_ids_from_shoeboxes(
                shoebox_data_set=self.full_data_set.shoeboxes
            )
            data_as_list = list(self.full_data_set.tensors)
            data_as_list[1] = torch.cat(
                (data_as_list[1], torch.tensor(image_ids).unsqueeze(1)), dim=1
            )
            self.full_data_set.tensors = tuple(data_as_list)
    
    # def _cut_metadata(self, metadata: torch.Tensor) -> torch.Tensor:
    #     # indices_to_keep = torch.tensor([self.settings.metadata_indices[i] for i in self.settings.metadata_keys_to_keep], dtype=torch.long)
    #     indices_to_keep = torch.tensor(0,1)
    #     return torch.index_select(metadata, dim=1, index=indices_to_keep)
    

    def _clean_shoeboxes_(self, shoeboxes: torch.Tensor, dead_pixel_mask, counts, metadata):
        shoebox_mask = (shoeboxes[..., -1].sum(dim=1) < 150000)
        return (shoeboxes[shoebox_mask], dead_pixel_mask[shoebox_mask], counts[shoebox_mask], metadata[shoebox_mask])

    def _split_full_data_(self) -> None:
        full_data_set_length = len(self.full_data_set)
        validation_data_set_length = int(
            full_data_set_length * self.data_loader_settings.validation_set_split
        )
        test_data_set_length = int(full_data_set_length * self.data_loader_settings.test_set_split)
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
        if not self.use_standard_sampler and self.data_loader_settings.append_image_id_to_metadata:
            self.append_image_id_to_metadata_()
        # self._clean_data_()
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
        
    def _map_images_to_shoeboxes(self, shoebox_data_set: torch.Tensor, metadata:torch.Tensor) -> dict:
        """Returns a dictionary with image ids as keys and indices of all shoeboxes 
            belonging to that image as values."""
        
        # image_ids = self._get_image_ids_from_shoeboxes(
        #     shoebox_data_set=shoebox_data_set
        # )
        print("metadata shape", metadata.shape)
        image_ids = metadata[:,2].round().to(torch.int64)  # Convert to integer type
        print("image ids", image_ids)
        import numpy as np
        unique_ids = torch.unique(image_ids)
        # print("Number of unique image IDs:", len(unique_ids))
        # print("First few unique image IDs:", unique_ids[:10])
        # print("Total number of image IDs:", len(image_ids))
        # print("Min image ID:", image_ids.min().item())
        # print("Max image ID:", image_ids.max().item())
        # print("Number of negative values:", (image_ids < 0).sum().item())
        # print("Number of zero values:", (image_ids == 0).sum().item())
        
        images_to_shoebox_indices = {}
        for shoebox_index, image_id in enumerate(image_ids):
            image_id_int = image_id.item()  # Convert tensor to Python int
            if image_id_int not in images_to_shoebox_indices:
                images_to_shoebox_indices[image_id_int] = []
            images_to_shoebox_indices[image_id_int].append(shoebox_index)
        
        # print("Number of keys in dictionary:", len(images_to_shoebox_indices))
        return images_to_shoebox_indices


    class BatchByImageSampler(Sampler):
        def __init__(self, image_id_to_indices: dict, data_loader_settings: settings.DataLoaderSettings):
            self.data_loader_settings = data_loader_settings
            # each element is a list of all shoebox-indices for one image
            self.image_indices_list = list(image_id_to_indices.values())
            self.batch_size = data_loader_settings.number_of_shoeboxes_per_batch
            self.num_batches = data_loader_settings.number_of_batches
            self.shuffle_groups = getattr(data_loader_settings, "shuffle_groups", True)

        def __iter__(self):
            images = self.image_indices_list.copy()
            # print("len images", len(images))
            if self.shuffle_groups:
                random.shuffle(images)

            batch = []
            batch_count = 0
            image_number = 0
            for image in images:
                image_number +=1
                for shoebox in image:
                    batch.append(shoebox)
                    if len(batch) == self.batch_size:
                        print("number of images", image_number)
                        image_number = 0
                        yield batch
                        batch = []
                        batch_count += 1
                        if batch_count >= self.num_batches:
                            return  

        def __len__(self):
            return self.num_batches

    def load_data_set_batched_by_image(self, 
                                       data_set_to_load: torch.utils.data.dataset.Subset | torch.utils.data.TensorDataset,
        ) -> torch.utils.data.dataloader.DataLoader:
        if self.use_standard_sampler:
            return DataLoader(
                data_set_to_load,
                batch_size=self.data_loader_settings.number_of_shoeboxes_per_batch,
                shuffle=self.data_loader_settings.shuffle_indices,
                num_workers=self.data_loader_settings.number_of_workers,
                pin_memory=self.data_loader_settings.pin_memory,
            )

        if isinstance(data_set_to_load, torch.utils.data.dataset.Subset):
            print("Metadata shape full tensor in loadeing:", self.full_data_set.metadata.shape) #(d, h,k, l ,x, y, z)

            image_id_to_indices = self._map_images_to_shoeboxes(
                shoebox_data_set=self.full_data_set.shoeboxes[data_set_to_load.indices],
                metadata=self.full_data_set.metadata[data_set_to_load.indices]
            )
        else:
            image_id_to_indices = self._map_images_to_shoeboxes(
                shoebox_data_set=self.full_data_set.shoeboxes
            )

        batch_by_image_sampler = self.BatchByImageSampler(
            image_id_to_indices=image_id_to_indices,
            data_loader_settings=self.data_loader_settings
            )
        return DataLoader(
            data_set_to_load,
            batch_sampler=batch_by_image_sampler,
            num_workers=self.data_loader_settings.number_of_workers,
            pin_memory=self.data_loader_settings.pin_memory,
            prefetch_factor=self.data_loader_settings.prefetch_factor,
        )
    
    def load_data_for_logging_during_training(self, number_of_shoeboxes_to_log: int = 5) -> torch.utils.data.dataloader.DataLoader:
        if self.use_standard_sampler:
            subset = Subset(self.train_data_set, indices=range(number_of_shoeboxes_to_log))
            return DataLoader(subset, batch_size=number_of_shoeboxes_to_log)

        image_id_to_indices = self._map_images_to_shoeboxes(
                shoebox_data_set=self.full_data_set.shoeboxes[self.train_data_set.indices],
                metadata=self.full_data_set.metadata[self.train_data_set.indices]
            )
        data_loader_settings = dataclasses.replace(self.data_loader_settings, number_of_shoeboxes_per_batch=number_of_shoeboxes_to_log, number_of_batches=1)
        batch_by_image_sampler = self.BatchByImageSampler(
            image_id_to_indices=image_id_to_indices,
            data_loader_settings=data_loader_settings
            )
        return DataLoader(
            self.train_data_set,
            batch_sampler=batch_by_image_sampler,
            num_workers=self.data_loader_settings.number_of_workers,
            pin_memory=self.data_loader_settings.pin_memory,
            prefetch_factor=self.data_loader_settings.prefetch_factor,
            persistent_workers=True
        )


    

