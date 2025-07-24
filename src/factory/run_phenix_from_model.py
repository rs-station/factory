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

    model_alias = "v10"
    config_alias = "v13"

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

    artifact = wandb.use_artifact(f"flaviagiehr-harvard-university/full-model/models:{model_alias}", type="model")
    artifact_dir = artifact.download()

    print("Files in artifact_dir:", os.listdir(artifact_dir))

    config_artifact = wandb.use_artifact(f"flaviagiehr-harvard-university/full-model/config:{config_alias}", type="config")
    config_artifact_dir = config_artifact.download()
    config_path = os.path.join(config_artifact_dir, "config_example.yaml")
    # config_path="/n/holylabs/LABS/hekstra_lab/Users/fgiehr/factory/configs/config_example.yaml"
    model_settings, loss_settings, phenix_settings, dataloader_settings = load_settings_from_yaml(path=config_path)

    # model_settings, loss_settings, dataloader_settings = load_settings_from_yaml(path="/n/holylabs/LABS/hekstra_lab/Users/fgiehr/factory/configs/config_example.yaml")


    checkpoint_paths = [os.path.join(artifact_dir, f) for f in os.listdir(artifact_dir) if f.endswith(".ckpt")]
    # artifact_dir = "/n/holylabs/hekstra_lab/Users/fgiehr/jobs/lightning_logs/wandb/run-20250718_124342-22j3mlzs/files/checkpoints"
    print(f"downloaded {len(checkpoint_paths)} checkpoint files.")

    print("run phenix")
    phenix_callback.run_phenix_over_all_checkpoints(
        model_settings=model_settings, 
        loss_settings=loss_settings, 
        phenix_settings=phenix_settings,
        artifact_dir=artifact_dir, 
        checkpoint_paths=checkpoint_paths,
        wandb_directory=wandb_logger.experiment.dir,#"/n/holylabs/LABS/hekstra_lab/Users/fgiehr/jobs/lightning_logs/wandb/run-20250720_200430-py737ahw/files",#wandb_logger.experiment.dir
        )

def main():
    # cProfile.run('run()')
    run()

if __name__ == "__main__":
    main()
