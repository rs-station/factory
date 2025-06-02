import torch
import pandas as pd
import numpy as np
import subprocess
import os
import matplotlib.pyplot as plt

import wandb
import pytorch_lightning as L

import reciprocalspaceship as rs
import rs_distributions as rsd

from model import *
import data_loader
import get_protein_data


wandb.init(project="anomalous peaks")

artifact = wandb.use_artifact("flaviagiehr-harvard-university/full-model/best_model:latest", type="model")
artifact_dir = artifact.download()

ckpt_files = [f for f in os.listdir(artifact_dir) if f.endswith(".ckpt")]
assert len(ckpt_files) == 1, "Expected only one checkpoint file."
checkpoint_path = os.path.join(artifact_dir, ckpt_files[0])

settings = Settings()
loss_settings = LossSettings()
# data_loader_settings = data_loader.DataLoaderSettings(data_directory=settings.data_directory,
#                             data_file_names=settings.data_file_names,
#                             )
# _dataloader = data_loader.CrystallographicDataLoader(settings=data_loader_settings)
# _dataloader.load_data_()
# dataloader = _dataloader.load_data_set_batched_by_image(data_set_to_load=_dataloader.train_data_set)



model = Model.load_from_checkpoint(checkpoint_path, settings=settings, loss_settings=loss_settings)#, dataloader=dataloader)
model.eval()



# hkl_batch = []
# _surrogate_posteriors_for_all_batches = []
# device = next(model.parameters()).device

# # with torch.no_grad():
# #     for batch in dataloader:
# #         # batch = [batch_item.to(device) for batch_item in batch]
# #         shoeboxes_batch, metadata_batch, dead_pixel_mask_batch, counts_batch, _hkl_batch = batch
# #         # output = model(batch)
# #         print("hkl batch shape:", _hkl_batch.shape)
# #         hkl_batch.append(_hkl_batch.to(torch.int).to(device))
#         # _surrogate_posteriors_for_all_batches.append(output[5])



# surrogate_posterior_all_reflections = model.surrogate_posterior.mean
# print("hkl batch shape:", len(hkl_batch), hkl_batch[0].shape)
# _hkls_with_duplicates = torch.cat(hkl_batch, dim=0)  # This will give us (90x8, 3)
# hkls = torch.unique(_hkls_with_duplicates, dim=0)
# print("hkls shape:", hkls.shape)
# print("surrogate posterior shape:", surrogate_posterior_all_reflections.shape)

# rasu_ids = model.surrogate_posterior.rac.rasu_ids[0].to(device)

# structure_factors = model.surrogate_posterior.rac.gather(
#     source=surrogate_posterior_all_reflections.T, rasu_id=rasu_ids, H=hkls.to(device)
# )

# H, K, L = hkls[:,0].cpu().numpy(), hkls[:,1].cpu().numpy(), hkls[:,2].cpu().numpy()
# F = structure_factors.cpu().detach().numpy()

# df = pd.DataFrame({
#     "H": H,
#     "K": K,
#     "L": L,
#     "F": F,
# })
# ds = rs.DataSet(df)
# ds.set_index(["H", "K", "L"], inplace=True)
# ds.spacegroup = settings.pdb_data["spacegroup"]  
# ds.cell = settings.pdb_data["unit_cell"]  

# ds["F"] = rs.DataSeries(ds["F"], dtype="F")

# ds.rename(columns={"F": "F_obs"}, inplace=True) #
# ds.rename(columns={"SIGF": "SIGF_obs"}, inplace=True) #


# Get absolute paths for all files
repo_dir = "/n/holylabs/LABS/hekstra_lab/Users/fgiehr/jobs/anomalous_peaks_files"
mtz_output_path = os.path.join(repo_dir, "phenix_ready.mtz")

# Get the dataset and verify its contents
surrogate_posterior_dataset = list(model.surrogate_posterior.to_dataset(only_observed=True))[0]

# Print dataset information
print("\nDataset Information:")
print(f"Number of reflections: {len(surrogate_posterior_dataset)}")
print(f"Columns available: {surrogate_posterior_dataset.columns.tolist()}")
print(f"Spacegroup: {surrogate_posterior_dataset.spacegroup}")
print(f"Cell parameters: {surrogate_posterior_dataset.cell}")

# Try to get wavelength from the dataset
wavelength = None
if hasattr(surrogate_posterior_dataset, 'wavelength'):
    wavelength = surrogate_posterior_dataset.wavelength
    print(f"\nWavelength from dataset: {wavelength}Å")
else:
    print("\nNo wavelength found in dataset, using default of 1.0Å")
    wavelength = "1.0"

# Print some statistics for available columns
print("\nStructure Factor Statistics:")
# Check if we have anomalous data
is_anomalous = any(col in surrogate_posterior_dataset.columns for col in ['F(+)', 'F(-)'])

if is_anomalous:
    print("Anomalous data detected:")
    for col in ['F(+)', 'F(-)', 'SIGF(+)', 'SIGF(-)']:
        if col in surrogate_posterior_dataset.columns:
            print(f"{col} mean: {surrogate_posterior_dataset[col].mean():.2f}")
            print(f"{col} std: {surrogate_posterior_dataset[col].std():.2f}")
else:
    print("Non-anomalous data:")
    for col in ['F', 'SIGF']:
        if col in surrogate_posterior_dataset.columns:
            print(f"{col} mean: {surrogate_posterior_dataset[col].mean():.2f}")
            print(f"{col} std: {surrogate_posterior_dataset[col].std():.2f}")

print(f"\nWriting MTZ file to: {mtz_output_path}")

# Write the MTZ file
surrogate_posterior_dataset.write_mtz(mtz_output_path)

# Verify the file was written
if os.path.exists(mtz_output_path):
    print(f"Successfully wrote MTZ file to: {mtz_output_path}")
    print(f"File size: {os.path.getsize(mtz_output_path) / 1024:.2f} KB")
    
    # Read and inspect the MTZ file
    print("\nInspecting MTZ file contents:")
    mtz_dataset = rs.read_mtz(mtz_output_path)
    print(f"Columns in MTZ file: {mtz_dataset.columns.tolist()}")
    print(f"Spacegroup: {mtz_dataset.spacegroup}")
    print(f"Cell parameters: {mtz_dataset.cell}")
    if hasattr(mtz_dataset, 'wavelength'):
        print(f"Wavelength: {mtz_dataset.wavelength}")
else:
    print("Error: Failed to write MTZ file")


# phenix_dir = str(Path(mtz_file).parent) + "/phenix_out"
# Path(phenix_dir).mkdir(parents=True, exist_ok=True)

# # Construct the phenix.refine command with proper escaping
# refine_command = (
#     f"phenix.refine {Path(phenix_eff).resolve()} {Path(mtz_file).resolve()} overwrite=true"
# )

# # Construct the find_peaks command
# peaks_command = (
#     f"rs.find_peaks *[0-9].mtz *[0-9].pdb " f"-f ANOM -p PANOM -z 5.0 -o peaks.csv"
# )

# full_command = (
#     f"source {phenix_env} && cd {phenix_dir} && {refine_command} && {peaks_command}"
# )

# try:
#     # Use subprocess.run instead of Popen for better error handling
#     result = subprocess.run(
#         full_command,
#         shell=True,
#         executable="/bin/bash",
#         capture_output=True,  # Capture both stdout and stderr
#         text=True,  # Convert output to string
#         check=True,  # Raise CalledProcessError on non-zero exit
#     )
#     print("Phenix command completed successfully")
#     return result
# except subprocess.CalledProcessError as e:
#     print(f"Command failed with error code: {e.returncode}")
#     print(
#         "Command that failed:", full_command
#     )  # Print the actual command for debugging
#     print("Working directory:", phenix_dir)
#     print("Standard Output:")
#     print(e.stdout)
#     print("Error Output:")
#     print(e.stderr)
#     raise


phenix_env = "/n/hekstra_lab/garden_backup/phenix-1.21/phenix-1.21.1-5286/phenix_env.sh"
phenix_output_dir = os.path.join(repo_dir, "phenix_output")
os.makedirs(phenix_output_dir, exist_ok=True)
cmd = f"source {phenix_env} && cd {repo_dir} && phenix.autosol {mtz_output_path} seq_file={settings.lysozyme_sequence_file_path} labels='F(+) SIGF(+) F(-) SIGF(-)' skip_xtriage=True general.top_output_dir={phenix_output_dir} verbose=True"

# First verify input files exist and are readable
if not os.path.exists(mtz_output_path):
    raise FileNotFoundError(f"MTZ file not found at {mtz_output_path}")
if not os.path.exists(settings.lysozyme_sequence_file_path):
    raise FileNotFoundError(f"Sequence file not found at {settings.lysozyme_sequence_file_path}")

# Check if files are readable
try:
    with open(mtz_output_path, 'rb') as f:
        pass
    with open(settings.lysozyme_sequence_file_path, 'r') as f:
        pass
except PermissionError:
    raise PermissionError(f"Cannot read one or more input files. Please check file permissions.")

print("\nRunning phenix.autosol with command:")
print(cmd)

try:
    # Run the command and capture both stdout and stderr
    result = subprocess.run(
        ["bash", "-c", cmd], 
        check=True, 
        capture_output=True, 
        text=True,
        env=os.environ.copy()  # Pass current environment
    )
    print("\nPhenix ran successfully.")
    print("Output:", result.stdout)
except subprocess.CalledProcessError as e:
    print("\nPhenix.autosol failed!")
    print(f"Error code: {e.returncode}")
    print("\nCommand output:")
    print(e.stdout)
    print("\nError output:")
    print(e.stderr)
    print("\nFull command that failed:")
    print(cmd)
    raise

# Check if the solved.mtz file exists before trying to add it to wandb
solved_mtz_path = os.path.join(phenix_output_dir, "solved.mtz")
if not os.path.exists(solved_mtz_path):
    print(f"\nWarning: Expected output file not found: {solved_mtz_path}")
    print("Checking phenix output directory contents:")
    if os.path.exists(phenix_output_dir):
        print("Files in output directory:")
        for f in os.listdir(phenix_output_dir):
            print(f"  {f}")
    else:
        print(f"Output directory {phenix_output_dir} does not exist")
    raise FileNotFoundError(f"Expected Phenix output file not found: {solved_mtz_path}")

solved_artifact = wandb.Artifact("phenix_solved", type="mtz")
solved_artifact.add_file(solved_mtz_path)
wandb.log_artifact(solved_artifact)


ds = rs.read_mtz(os.path.join(phenix_output_dir, "solved.mtz"))


# check columns
print("check columns", ds.columns)

# "F(+)", "F(-)", "PHIF(+)", "PHIF(-)"

peaks = rsd.peak_finder(
    ds,
    map_column="F(+) - F(-)",  # anomalous difference column
    dmin=2.5,
    sigma=3.5
)

# Save peaks
peaks.to_csv("anomalous_peaks.csv", index=False)

peaks_artifact = wandb.Artifact("anomalous_peaks", type="peaks")
peaks_artifact.add_file("anomalous_peaks.csv")
wandb.log_artifact(peaks_artifact)