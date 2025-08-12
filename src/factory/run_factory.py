from model import *
from settings import *
from parse_yaml import load_settings_from_yaml, _log_settings_from_yaml
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import Callback, ModelCheckpoint

import glob
import os
import phenix_callback
import wandb
from pytorch_lightning.tuner.tuning import Tuner
import argparse
from datetime import datetime
from callbacks import *
import yaml

from omegaconf import OmegaConf

def configure_settings(config_path=None):
    
    if config_path is None:
        config_path = "/n/holylabs/LABS/hekstra_lab/Users/fgiehr/factory/configs/base_for_sweep.yaml"
    
    model_settings, loss_settings, dataloader_settings = load_settings_from_yaml(config_path)
    return model_settings, loss_settings, dataloader_settings
    

def run(config_path=None, run_from_version=False):
    
    if config_path is None:
        config_path = "/n/holylabs/LABS/hekstra_lab/Users/fgiehr/factory/configs/base_for_sweep.yaml"

    

    now_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    if run_from_version is False:

        run_name = f"{now_str}_from-start"
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        base_cfg = OmegaConf.load(config_path)

        wandb_logger = WandbLogger(project="full-model", name=run_name, save_dir="/n/holylabs/LABS/hekstra_lab/Users/fgiehr/jobs/lightning_logs")
        wandb_run = wandb.init(
            project="full-model",
            name=run_name,
            dir="/n/holylabs/LABS/hekstra_lab/Users/fgiehr/jobs/lightning_logs",
            config=config_dict #OmegaConf.to_container(base_cfg, resolve=True),
        )

        override = OmegaConf.create(dict(wandb_run.config))
        wandb_config = OmegaConf.merge(base_cfg, override)

        wandb_config = wandb_run.config

        model_settings, loss_settings, phenix_settings, dataloader_settings = load_settings_from_yaml(wandb_config) #wandb_config)
        print("model settings", model_settings)
        slurm_job_id = os.environ.get("SLURM_JOB_ID")
        if slurm_job_id is not None:
            wandb.config.update({"slurm_job_id": slurm_job_id})
        print("wandb_logger.experiment.dir",wandb_logger.experiment.dir)
        # _log_settings_from_yaml(config_path, logger=wandb_logger)
        model = Model(model_settings=model_settings, loss_settings=loss_settings)
        
        config_artifact = wandb.Artifact("config", type="config")
        config_artifact.add_file(config_path)  # Path to your config file
        logged_artifact = wandb_logger.experiment.log_artifact(config_artifact)

    else:
        run_name = f"{now_str}_continue-run-44"
        wandb_logger = WandbLogger(project="full-model", name=run_name, save_dir="/n/holylabs/LABS/hekstra_lab/Users/fgiehr/jobs/lightning_logs")
        wandb.init(
            project="full-model",
            name=run_name,
            dir="/n/holylabs/LABS/hekstra_lab/Users/fgiehr/jobs/lightning_logs"
        )
        config_path = "/n/holylabs/LABS/hekstra_lab/Users/fgiehr/factory/configs/config_example.yaml"

        # config_artifact = wandb.use_artifact(f"flaviagiehr-harvard-university/full-model/config:v25", type="config")
        # config_artifact_dir = config_artifact.download()
        # config_path = os.path.join(config_artifact_dir, "config_example.yaml")
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        model_settings, loss_settings, phenix_settings, dataloader_settings = load_settings_from_yaml(config_dict)

        
        slurm_job_id = os.environ.get("SLURM_JOB_ID")
        if slurm_job_id is not None:
            wandb.config.update({"slurm_job_id": slurm_job_id})
        artifact = wandb.use_artifact(f"flaviagiehr-harvard-university/full-model/models:v18", type="model")
        artifact_dir = artifact.download()

        ckpt_files = [f for f in os.listdir(artifact_dir) if f.endswith(".ckpt")]
        checkpoint_path = os.path.join(artifact_dir, ckpt_files[-1])
        # config_path = os.path.join(artifact_dir, "config.yaml")
        # model_settings, loss_settings, dataloader_settings = load_settings_from_yaml(path=config_path, logger=wandb_logger)
        # _log_settings_from_yaml(config_path)
        model = Model.load_from_checkpoint(checkpoint_path, model_settings=model_settings, loss_settings=loss_settings)

        # config_artifact = wandb.Artifact("model_config", type="config")
        # config_artifact.add_file(checkpoint_path)  # Path to your config file

    # if dataloader_settings is None:
    #     data_loader_settings = settings.DataLoaderSettings(
    #         data_directory=model_settings.data_directory,
    #         data_file_names=model_settings.data_file_names)
   
    dataloader = data_loader.CrystallographicDataLoader(data_loader_settings=dataloader_settings)
    print("Loading data ...")
    dataloader.load_data_()
    print("Data loaded successfully.")

    # Create train and validation dataloaders
    train_dataloader = dataloader.load_data_set_batched_by_image(
        data_set_to_load=dataloader.train_data_set,
    )
    val_dataloader = dataloader.load_data_set_batched_by_image(
        data_set_to_load=dataloader.validation_data_set,
    )


    # if model_settings.enable_checkpointing:
    checkpoint_callback = ModelCheckpoint(
        dirpath=wandb_logger.experiment.dir + "/checkpoints",
        # monitor="validation_loss/loss",
        save_top_k=-1,
        every_n_epochs=1,
        # mode="min",
        filename="{epoch:02d}",#-{step}-{validation_loss/loss:.2f}",
        # save_weights_only=True,
        # every_n_train_steps=None,  # Disable step-based checkpointing
        # save_last=False  # Don't save the last checkpoint
    )

    trainer = L.pytorch.Trainer(
        logger=wandb_logger, 
        max_epochs=20, 
        log_every_n_steps=40, 
        val_check_interval=40, 
        # limit_val_batches=21,
        accelerator="auto", 
        enable_checkpointing=True, #model_settings.enable_checkpointing, 
        # default_root_dir="/tmp",
        # profiler="simple",
        # limit_train_batches=20,
        callbacks=[
            Plotting(dataloader=dataloader),
            LossLogging(),
            CorrelationPlottingBinned(dataloader=dataloader), 
            CorrelationPlotting(dataloader=dataloader), 
            ScalePlotting(dataloader=dataloader),
            checkpoint_callback,
            # PeakHeights(),
        ]
    )

    
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    
    if model_settings.enable_checkpointing:
        checkpoint_dir = checkpoint_callback.dirpath if hasattr(checkpoint_callback, "dirpath") else "/tmp"

        print("checkpoint_dir", checkpoint_dir)
        checkpoint_pattern = os.path.join(checkpoint_dir, "**", "*.ckpt")
        checkpoint_paths = glob.glob(checkpoint_pattern, recursive=True)

        print("checkpoint_paths", checkpoint_paths)

        print("model_settings.enable_checkpointing",model_settings.enable_checkpointing)
        print("checkpoint_callback.best_model_path",checkpoint_callback.best_model_path)

        config_artifact_ref = logged_artifact.name  # e.g., 'your-entity/your-project/config:v0'
        version_alias = config_artifact_ref.split(":")[-1]

        artifact = wandb.Artifact('models', type='model')
        for ckpt_file in checkpoint_paths:
            artifact.add_file(ckpt_file)
        print(f"Logged {len(checkpoint_paths)} checkpoints to W&B artifact.")

        wandb_logger.experiment.log_artifact(artifact, aliases=[version_alias])

        print("run phenix")
        phenix_callback.run_phenix_over_all_checkpoints(
            model_settings=model_settings, 
            loss_settings=loss_settings, 
            phenix_settings=phenix_settings,
            artifact_dir=checkpoint_dir, 
            checkpoint_paths=checkpoint_paths,
            wandb_directory=wandb_logger.experiment.dir
        )


        # for ckpt_file in checkpoint_paths:
        #     try:
        #         os.remove(ckpt_file)
        #         print(f"Deleted checkpoint file: {ckpt_file}")
        #     except Exception as e:
        #         print(f"Failed to delete {ckpt_file}: {e}")

    wandb.finish()

# def run(config_path):
#     run(config_path=args.config)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run factory with config file')
    parser.add_argument('--config', type=str, help='Path to config YAML file')
    args, _ = parser.parse_known_args()
    
    # cProfile.run('run()')
    run(config_path=args.config)

if __name__ == "__main__":
    main()
