import settings
from data_loader import CrystallographicDataLoader


def test_dataloader(data_loader_settings: settings.DataLoaderSettings):
    print("enter test")
    dataloader = CrystallographicDataLoader(data_loader_settings=data_loader_settings)
    print("Loading data ...")
    dataloader.load_data_()
    print("Data loaded successfully.")

    test_data = dataloader.load_data_set_batched_by_image(
        data_set_to_load=dataloader.train_data_set
    )
    print("Test data loaded successfully.")
    for batch in test_data:
        shoeboxes_batch, metadata_batch, dead_pixel_mask_batch, counts_batch, hkl = (
            batch
        )
        print("Batch shoeboxes :", shoeboxes_batch.shape)
        print("image ids", metadata_batch[:, 2])
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
    # args = parse_args()
    data_loader_settings = settings.DataLoaderSettings()
    test_dataloader(data_loader_settings=data_loader_settings)


if __name__ == "__main__":
    main()
