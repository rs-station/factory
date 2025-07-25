import yaml
import os
import wandb
from settings import ModelSettings, LossSettings, DataLoaderSettings, PhenixSettings
from abismal_torch.prior import WilsonPrior
import distributions
import networks
from lightning.pytorch.loggers import WandbLogger
import metadata_encoder
import shoebox_encoder
import torch

REGISTRY = {
    "HalfNormalDistribution": distributions.HalfNormalDistribution,
    "TorchGamma": torch.distributions.Gamma,
    "DirichletProfile": distributions.DirichletProfile,
    "WilsonPrior": WilsonPrior,
    "LogNormalDistributionLayer": networks.LogNormalDistributionLayer,
    "FoldedNormalDistributionLayer": networks.FoldedNormalDistributionLayer,
    "GammaDistributionLayer": networks.GammaDistributionLayer,
    "DeltaDistributionLayer": networks.DeltaDistributionLayer,
    "BaseShoeboxEncoder": shoebox_encoder.BaseShoeboxEncoder,
    "BaseMetadataEncoder": metadata_encoder.BaseMetadataEncoder,
    "SimpleShoeboxEncoder": shoebox_encoder.SimpleShoeboxEncoder,
    "SimpleMetadataEncoder": metadata_encoder.SimpleMetadataEncoder,
    "MLPScale": networks.MLPScale,
    "LogNormalDistribution": torch.distributions.LogNormal,
    "TorchUniform": torch.distributions.Uniform,
}

def _log_settings_from_yaml(path: str, logger: WandbLogger):
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    logger.experiment.config.update(config)

def instantiate_from_config(config: dict):
    def resolve_value(v):
        if isinstance(v, dict):
            if "name" in v and v["name"] in REGISTRY:
                params = v.get("params", {})
                resolved_params = {k: resolve_value(val) for k, val in params.items()}
                return REGISTRY[v["name"]](**resolved_params)
            else:
                return {k: resolve_value(val) for k, val in v.items()}
        elif isinstance(v, str) and v in REGISTRY:
            return REGISTRY[v]
        else:
            return v

    return {k: resolve_value(v) for k, v in config.items()}


def consistancy_check(model_settings_dict: dict):
    
    dmodel = model_settings_dict.get("dmodel")
    metadata_indices = model_settings_dict.get("metadata_indices_to_keep", [])

    if "shoebox_encoder" in model_settings_dict:
        model_settings_dict["shoebox_encoder"].setdefault("params", {})["out_dim"] = dmodel

    if "metadata_encoder" in model_settings_dict:
        model_settings_dict["metadata_encoder"].setdefault("params", {})["feature_dim"] = len(metadata_indices)
        model_settings_dict["metadata_encoder"]["params"]["output_dims"] = dmodel

    if "scale_function" in model_settings_dict:
        model_settings_dict["scale_function"].setdefault("params", {})["input_dimension"] = dmodel


def load_settings_from_yaml(config):
    # with open(path, 'r') as file:
    #     config = yaml.safe_load(file)

    model_settings_dict = config.get("model_settings", {})
    
    for key, value in model_settings_dict.items():
        if value == "None":
            model_settings_dict[key] = None

    consistancy_check(model_settings_dict)

    model_settings_dict = instantiate_from_config(model_settings_dict)

    model_settings = ModelSettings(**model_settings_dict)

    loss_settings = LossSettings(**config.get("loss_settings", {}))

    phenix_settings = PhenixSettings(**config.get("phenix_settings", {}))

    dataloader_settings_dict = config.get("dataloader_settings", {})
    dataloader_settings = DataLoaderSettings(**dataloader_settings_dict) if dataloader_settings_dict else None

    return model_settings, loss_settings, phenix_settings, dataloader_settings


def resolve_builders(settings_dict):
    for field in [
        "build_background_distribution",
        "build_shoebox_profile_distribution",
        "build_intensity_prior_distribution",
        "shoebox_encoder",
        "metadata_encoder",
    ]:
        if field in settings_dict:
            settings_dict[field] = REGISTRY[settings_dict[field]]

    return settings_dict