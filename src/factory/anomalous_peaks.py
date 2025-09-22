import os
import subprocess

import data_loader
import get_protein_data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as L
import reciprocalspaceship as rs
import rs_distributions as rsd
import torch
import wandb
from model import *

print(
    rs.read_mtz(
        "/n/holylabs/LABS/hekstra_lab/Users/fgiehr/creat_dials_unmerged/merged.mtz"
    )
)

wandb.init(project="anomalous peaks", name="anomalous-peaks")

artifact = wandb.use_artifact(
    "flaviagiehr-harvard-university/full-model/best_model:latest", type="model"
)
artifact_dir = artifact.download()

ckpt_files = [f for f in os.listdir(artifact_dir) if f.endswith(".ckpt")]
assert len(ckpt_files) == 1, "Expected only one checkpoint file."
checkpoint_path = os.path.join(artifact_dir, ckpt_files[0])

settings = Settings()
loss_settings = LossSettings()

model = Model.load_from_checkpoint(
    checkpoint_path, settings=settings, loss_settings=loss_settings
)  # , dataloader=dataloader)
model.eval()

repo_dir = "/n/holylabs/LABS/hekstra_lab/Users/fgiehr/jobs/anomalous_peaks_files"
mtz_output_path = os.path.join(repo_dir, "phenix_ready.mtz")

surrogate_posterior_dataset = list(
    model.surrogate_posterior.to_dataset(only_observed=True)
)[0]

# Additional validation checks
if len(surrogate_posterior_dataset) == 0:
    raise ValueError("Dataset has no reflections!")

# Check if we have the required columns
required_columns = ["F(+)", "F(-)", "SIGF(+)", "SIGF(-)"]
missing_columns = [
    col for col in required_columns if col not in surrogate_posterior_dataset.columns
]
if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

# Check for any NaN or infinite values
for col in required_columns:
    if surrogate_posterior_dataset[col].isna().any():
        print(f"Warning: Column {col} contains NaN values")
    if np.isinf(surrogate_posterior_dataset[col]).any():
        print(f"Warning: Column {col} contains infinite values")

# Ensure d_min is set
if (
    not hasattr(surrogate_posterior_dataset, "d_min")
    or surrogate_posterior_dataset.d_min is None
):
    print("Setting d_min to 2.5Å")
    surrogate_posterior_dataset.d_min = 2.5

# Try to get wavelength from the dataset
wavelength = None
if hasattr(surrogate_posterior_dataset, "wavelength"):
    wavelength = surrogate_posterior_dataset.wavelength
    print(f"\nWavelength from dataset: {wavelength}Å")
else:
    print("\nNo wavelength found in dataset, using default of 1.0Å")
    wavelength = "1.0"

print(f"\nWriting MTZ file to: {mtz_output_path}")

# Write the MTZ file
surrogate_posterior_dataset.write_mtz(mtz_output_path)

# Verify the file was written
if os.path.exists(mtz_output_path):
    print(f"Successfully wrote MTZ file to: {mtz_output_path}")
else:
    print("Error: Failed to write MTZ file")


phenix_env = "/n/hekstra_lab/garden_backup/phenix-1.21/phenix-1.21.1-5286/phenix_env.sh"
phenix_output_dir = os.path.join(repo_dir, "phenix_output")
os.makedirs(phenix_output_dir, exist_ok=True)
path_to_pdb_model = "/n/holylabs/LABS/hekstra_lab/Users/fgiehr/jobs/anomalous_peaks_files/pdb_model/9b7c.pdb"

# mtz_output_path = "/n/holylabs/LABS/hekstra_lab/Users/fgiehr/creat_dials_unmerged/merged.mtz"  # for the reference peaks

phenix_refine_eff_file_path = (
    "/n/holylabs/LABS/hekstra_lab/Users/fgiehr/jobs/anomalous_peaks_files/phenix.eff"
)


cmd = f"source {phenix_env} && cd {repo_dir} && phenix.refine {phenix_refine_eff_file_path} {mtz_output_path} overwrite=True"

print("\nRunning phenix with command:", flush=True)
print(cmd, flush=True)

# try:
#     result = subprocess.run(
#             cmd,
#             shell=True,
#             executable="/bin/bash",
#             capture_output=True,  # Capture both stdout and stderr
#             text=True,  # Convert output to string
#             check=True,  # Raise CalledProcessError on non-zero exit
#         )
#     print("Phenix command completed successfully")

# except subprocess.CalledProcessError as e:
#     print("\nPhenix.refine failed!", flush=True)
#     print(f"Error code: {e.returncode}", flush=True)
#     print("\nCommand output:", flush=True)
#     print(e.stdout, flush=True)
#     print("\nError output:", flush=True)
#     print(e.stderr, flush=True)
#     print("\nFull command that failed:", flush=True)
#     print(cmd, flush=True)
#     raise

solved_artifact = wandb.Artifact("phenix_solved", type="mtz")
# solved_artifact.add_file(solved_mtz_path)
# wandb.log_artifact(solved_artifact)

# ds = rs.read_mtz("/n/holylabs/LABS/hekstra_lab/Users/fgiehr/jobs/anomalous_peaks_files/AutoSol_run_3_/exptl_fobs_phases_freeR_flags_2_with_anom_data_with_hl_anom.mtz")
# print(ds.columns)
# ds = rs.read_mtz("/n/holylabs/LABS/hekstra_lab/Users/fgiehr/jobs/anomalous_peaks_files/AutoSol_run_3_/F_ano.mtz")
# print(ds.columns)
# print("full length", len(ds))
# ds = ds.dropna(subset=["I(+)", "I(-)"])
# print("cleaned length", len(ds))

# ds["F(+)"] = np.where(ds["I(+)"] > 0, np.sqrt(ds["I(+)"]), 0)
# ds["F(-)"] = np.where(ds["I(-)"] > 0, np.sqrt(ds["I(-)"]), 0)
# ds["F(+) - F(-)"] = ds["F(+)"] - ds["F(-)"]

# ds = rs.read_mtz(os.path.join(phenix_output_dir, "solved.mtz"))


# # check columns
# print("check columns", ds.columns)

# "F(+)", "F(-)", "PHIF(+)", "PHIF(-)"
# from rs_distributions.peak_finding import peak_finder

subprocess.run(
    "rs.find_peaks *[0-9].mtz *[0-9].pdb " f"-f ANOM -p PANOM -z 5.0 -o peaks.csv",
    shell=True,
    cwd=f"{repo_dir}",
)

# Save peaks
print("saved peaks")

peaks_df = pd.read_csv(
    "/n/holylabs/LABS/hekstra_lab/Users/fgiehr/jobs/anomalous_peaks_files/peaks.csv"
)
print(peaks_df.columns)
# print(peaks_df.iloc[0])   #
print(peaks_df)

# Create a nice table of peak heights
peak_heights_table = pd.DataFrame(
    {"Residue": peaks_df["residue"], "Peak_Height": peaks_df["peakz"]}
)

# Sort by peak height in descending order for better visualization
peak_heights_table = peak_heights_table.sort_values(
    "Peak_Height", ascending=False
).reset_index(drop=True)

# Add rank column for easier reference
peak_heights_table.insert(0, "Rank", range(1, len(peak_heights_table) + 1))

# Round peak heights to 3 decimal places for cleaner display
peak_heights_table["Peak_Height"] = peak_heights_table["Peak_Height"].round(3)

print("\n=== Anomalous Peak Heights ===")
print(peak_heights_table.to_string(index=False))
print(f"\nTotal peaks found: {len(peak_heights_table)}")
print(f"Highest peak: {peak_heights_table['Peak_Height'].max():.3f}")
print(f"Average peak height: {peak_heights_table['Peak_Height'].mean():.3f}")

# Log the peak heights table to wandb
wandb.log({"peak_heights": wandb.Table(dataframe=peak_heights_table)})


peaks_artifact = wandb.Artifact("anomalous_peaks", type="peaks")
peaks_artifact.add_file(os.path.join(repo_dir, "peaks.csv"))
wandb.log_artifact(peaks_artifact)
