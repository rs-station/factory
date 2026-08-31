import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
from Bio.PDB import PDBParser
from scipy.spatial import cKDTree

wandb.init(project="plot anomolous peaks")

peaks_df = pd.read_csv(
    "/n/holylabs/LABS/hekstra_lab/Users/fgiehr/jobs/anomalous_peaks_files/peaks.csv"
)
print(peaks_df.columns)
# print(peaks_df.iloc[0])   #
print(peaks_df)

# Create a table of peak heights
peak_heights_table = pd.DataFrame({"Peak_Height": peaks_df["peakz"]})

# Log the peak heights table to wandb
wandb.log({"peak_heights": wandb.Table(dataframe=peak_heights_table)})

peak_coords = peaks_df[["cenx", "ceny", "cenz"]].values
model_file = "/n/holylabs/LABS/hekstra_lab/Users/fgiehr/jobs/anomalous_peaks_files/shaken_9b7c_refine_001.pdb"
parser = PDBParser(QUIET=True)
model = parser.get_structure("model", model_file)

atoms = []
atom_coords = []

for atom in model.get_atoms():
    atoms.append(atom)
    atom_coords.append(atom.coord)

atom_coords = np.array(atom_coords)

tree = cKDTree(atom_coords)

# Match peaks to atoms
matches = []
for i, peak in enumerate(peak_coords):
    dist, idx = tree.query(peak)
    atom = atoms[idx]
    matches.append(
        {
            "Peak_X": peak[0],
            "Peak_Y": peak[1],
            "Peak_Z": peak[2],
            "Peak_Height": peaks_df.loc[i, "peakz"],  # adjust if column name differs
            "Atom_Name": atom.get_name(),
            "Residue": atom.get_parent().get_resname(),
            "Residue_ID": atom.get_parent().get_id()[1],
            "Chain": atom.get_parent().get_parent().id,
            "B_factor": atom.get_bfactor(),
            "Occupancy": atom.get_occupancy(),
            "Distance": dist,
        }
    )

# Create output DataFrame
results_df = pd.DataFrame(matches)

# Optionally filter by distance
results_df = results_df[results_df["Distance"] < 1.5]  # example cutoff

# Save or print table
results_df.to_csv("peak_atom_matches.csv", index=False)
print(results_df)
wandb.log({"peaks_vs_model": results_df.to_csv("peak_atom_matches.csv", index=False)})

# peak_coords = np.array([atom.coord for atom in model.get_atoms()])
# === Load model atoms ===

# model_file = "/n/holylabs/LABS/hekstra_lab/Users/fgiehr/jobs/anomalous_peaks_files/pdb_model/shaken_9b7c.pdb"
# parser = PDBParser(QUIET=True)
# model = parser.get_structure("model", model_file)
# model_coords = np.array([atom.coord for atom in model.get_atoms()])

# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111)

# # Plot RS-Boost peaks
# # ax.scatter(peak_coords[:, 0], peak_coords[:, 1], peak_coords[:, 2],
# #            c='red', s=20, label='RS-Boost Peaks')
# ax.plot(peak_coords)#,model_coords[:,2])
# # Plot model atoms
# # ax.scatter(model_coords[:, 0], model_coords[:, 1], model_coords[:, 2],
# #            c='blue', s=2, alpha=0.5, label='Model Atoms')

# # ax.set_xlabel("X (Å)")
# # ax.set_ylabel("Y (Å)")
# # ax.set_zlabel("Z (Å)")
# # ax.set_title("RS-Boost Peaks vs Model Atoms")
# # ax.legend()
# # plt.tight_layout()

# # Log the figure to wandb
# wandb.log({"peaks_vs_model": wandb.Image(fig)})
# plt.show()
