from model import *
from settings import *
from parse_yaml import load_settings_from_yaml
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import Callback, ModelCheckpoint

import glob
import os
import phenix_callback
import wandb
from datetime import datetime
from pytorch_lightning.tuner.tuning import Tuner

def run():

    model_alias = "v53"

    now_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    wandb_logger = WandbLogger(project="full-model", name=f"{now_str}_phenix_from_{model_alias}", save_dir="/n/holylabs/LABS/hekstra_lab/Users/fgiehr/jobs/lightning_logs")
    wandb.init(
        project="full-model",
        name=f"{now_str}_phenix_from_{model_alias}",
        dir="/n/holylabs/LABS/hekstra_lab/Users/fgiehr/jobs/lightning_logs"
    )
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    if slurm_job_id is not None:
        wandb.config.update({"slurm_job_id": slurm_job_id})
    print("wandb_logger.experiment.dir",wandb_logger.experiment.dir)

    artifact = wandb.use_artifact(f"flaviagiehr-harvard-university/full-model/best_models:{model_alias}", type="model")
    artifact_dir = artifact.download()

    print("Files in artifact_dir:", os.listdir(artifact_dir))

    # config_artifact = wandb.use_artifact(f"flaviagiehr-harvard-university/full-model/best_models:{model_alias}", type="model")
    # config_path = os.path.join(config_artifact.download(), "config.yaml")
    # model_settings, loss_settings, dataloader_settings = load_settings_from_yaml(path=config_path)

    model_settings, loss_settings, dataloader_settings = load_settings_from_yaml(path="/n/holylabs/LABS/hekstra_lab/Users/fgiehr/factory/configs/config_example.yaml")


    ckpt_file = [f for f in os.listdir(artifact_dir) if f.endswith(".ckpt")][-1]

    print(f"downloaded {len(ckpt_files)} checkpoint files.")

    

def main():
    # cProfile.run('run()')
    run()

if __name__ == "__main__":
    main()