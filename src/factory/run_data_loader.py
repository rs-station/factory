import argparse
import json

import CNN_3d
from data_loader import *


def parse_args():
    parser = argparse.ArgumentParser(description="Pass DataLoader settings")
    parser.add_argument(
        "--data_directory",
        type=str,
        help="Path to the data directory",
    )
    parser.add_argument(
        "--data_file_names",
        type=json.loads,
        default=None,
        help='JSON string of file names, e.g. \'{"shoeboxes": "shoebox.pt", "counts": "counts.pt"}\'',
    )
    return parser.parse_args()


def main():
    print("main function")
    # args = parse_args()
    # if args.data_file_names is not None:
    #     settings = DataLoaderSettings(data_directory=args.data_directory,
    #                                 data_file_names=args.data_file_names,
    #                                 test_set_split=0.01
    #                                 )
    # else:
    # data_directory = "/n/hekstra_lab/projects/shoeboxes"
    # data_file_names = {
    # "shoeboxes": "standardized_shoeboxes.pt",
    # "counts": "raw_counts.pt",
    # "metadata": "shoebox_features.pt",
    # "masks": "weak_masks.pt",
    # "true_reference": "metadata.pt",
    # }

    data_directory = "/n/hekstra_lab/people/aldama/subset"
    data_file_names = {
        "shoeboxes": "standardized_shoeboxes_subset.pt",
        "counts": "raw_counts_subset.pt",
        "metadata": "shoebox_features_subset.pt",
        "masks": "masks_subset.pt",
        "true_reference": "metadata_subset.pt",
    }

    print("load settings")
    settings = DataLoaderSettings(
        data_directory=data_directory,
        data_file_names=data_file_names,
        test_set_split=0.5,
    )
    print("load settings done")
    test_data = test(settings=settings)
    batch = next(iter(test_data))
    shoeboxes_batch, metadata_batch, dead_pixel_mask_batch, counts_batch = batch
    print("push into CNN")
    model = CNN_3d.CNN_3d()
    x = torch.clamp(shoeboxes_batch, 0, 1)

    out = model(x)

    print(out.shape)


if __name__ == "__main__":
    main()
