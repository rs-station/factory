from data_loader import *
import argparse
import json

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
    args = parse_args()
    if args.data_file_names is not None:
        settings = DataLoaderSettings(data_directory=args.data_directory, 
                                    data_file_names=args.data_file_names,
                                    test_set_split=0.01
                                    )
    else:
        data_file_names = {
        "shoeboxes": "standardized_shoeboxes_subset.pt",
        "counts": "raw_counts_subset.pt.pt",
        "metadata": "shoebox_features_subset.pt",
        "masks": "masks_subset.pt",
        "true_reference": "metadata_subset.pt",
        }
        settings = DataLoaderSettings(data_directory=args.data_directory,
                                    test_set_split=0.01
                                    )
    test(settings=settings)

if __name__ == "__main__":
    main()

