import torch
import pandas as pd
import numpy as np
import subprocess

import wandb
import pytorch_lightning as L

import reciprocalspaceship as rs

from model import *
import data_loader
import get_protein_data

wandb.init(project="anomalous peaks")

artifact = wandb.use_artifact("flaviagiehr/full-model/best_model:latest", type="model")
artifact_dir = artifact.download()

ckpt_files = [f for f in os.listdir(artifact_dir) if f.endswith(".ckpt")]
assert len(ckpt_files) == 1, "Expected only one checkpoint file."
checkpoint_path = os.path.join(artifact_dir, ckpt_files[0])

model = Model.load_from_checkpoint(checkpoint_path)
model.eval()

settings = Settings()

data_loader_settings = data_loader.DataLoaderSettings(data_directory=settings.data_directory,
                            data_file_names=settings.data_file_names,
                            )
dataloader = data_loader.CrystallographicDataLoader(settings=data_loader_settings)
dataloader.load_data_()

results = []

with torch.no_grad():
    for batch in dataloader:
        output = model(batch)
        results.append(output.cpu())

hkl_batch = results[3]

surrogate_posterior = results[5]

_surrogate_posterior_samples = surrogate_posterior.rsample([500])

rasu_ids = surrogate_posterior.rac.rasu_ids[0]
structure_factors = surrogate_posterior.rac.gather(
    source=_surrogate_posterior_samples.T, rasu_id=rasu_ids, H=hkl_batch.to(torch.int)
).mean(dim=1)

H, K, L = hkl_batch[:, 0].cpu().numpy(), hkl_batch[:, 1].cpu().numpy(), hkl_batch[:, 2].cpu().numpy()
F = structure_factors.cpu().numpy()

df = pd.DataFrame({
    "H": H,
    "K": K,
    "L": L,
    "F": F,
})
ds = rs.DataSet(df)

pdb_data =  get_protein_data.get_protein_data(settings.protein_pdb_url)
ds.set_index(["H", "K", "L"], inplace=True)
ds.set_spacegroup(pdb_data["spacegroup"])  
ds.set_unitcell(pdb_data["unit_cell"])  

ds["F"].set_observation_type("intensity")
ds["F"].set_mtztype("F")

ds.rename(columns={"F": "F_obs", "SIGF": "SIGF_obs"}, inplace=True)
ds.write_mtz("phenix_ready.mtz")
# ds.write_mtz("model_output.mtz")

import subprocess

phenix_command = [
    "phenix.autosol", 
    "phenix_ready.mtz",
    f"sequence_file={settings.lysozime_sequence_file_path}",  
]

subprocess.run(phenix_command, check=True)
