import os

import distributions
import metadata_encoder
import networks
import shoebox_encoder
import torch
import wandb
import yaml
from abismal_torch.prior import WilsonPrior
from lightning.pytorch.loggers import WandbLogger
from settings import DataLoaderSettings, LossSettings, ModelSettings, PhenixSettings

REGISTRY = {
    "HalfNormalDistribution": distributions.HalfNormalDistribution,
    "ExponentialDistribution": distributions.ExponentialDistribution,
    "TorchGamma": torch.distributions.Gamma,
    "DirichletProfile": distributions.DirichletProfile,
    "LRMVN_Distribution": distributions.LRMVN_Distribution,
    "WilsonPrior": WilsonPrior,
    "LogNormalDistributionLayer": networks.LogNormalDistributionLayer,
    "FoldedNormalDistributionLayer": networks.FoldedNormalDistributionLayer,
    "GammaDistributionLayer": networks.GammaDistributionLayer,
    "DeltaDistributionLayer": networks.DeltaDistributionLayer,
    "SoftplusNormalDistributionLayer": networks.SoftplusNormalDistributionLayer,
    "TruncatedNormalDistributionLayer": networks.TruncatedNormalDistributionLayer,
    "BaseShoeboxEncoder": shoebox_encoder.BaseShoeboxEncoder,
    "BaseMetadataEncoder": metadata_encoder.BaseMetadataEncoder,
    "SimpleShoeboxEncoder": shoebox_encoder.SimpleShoeboxEncoder,
    "SimpleMetadataEncoder": metadata_encoder.SimpleMetadataEncoder,
    "IntensityEncoder": shoebox_encoder.IntensityEncoder,
    "MLPScale": networks.MLPScale,
    "LogNormalDistribution": torch.distributions.LogNormal,
    "TorchUniform": torch.distributions.Uniform,
    "TorchHalfNormal": torch.distributions.HalfNormal,
    "TorchExponential": torch.distributions.Exponential,
    "rsFoldedNormal": networks.FoldedNormal,
}


def _log_settings_from_yaml(path: str, logger: WandbLogger):
    with open(path, "r") as file:
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
        model_settings_dict["shoebox_encoder"].setdefault("params", {})[
            "out_dim"
        ] = dmodel

    if "metadata_encoder" in model_settings_dict:
        if model_settings_dict.get("use_positional_encoding", False) == True:
            number_of_frequencies = model_settings_dict.get(
                "number_of_frequencies_in_positional_encoding", 2
            )
            model_settings_dict["metadata_encoder"].setdefault("params", {})[
                "feature_dim"
            ] = (len(metadata_indices) * number_of_frequencies * 2)
        else:
            model_settings_dict["metadata_encoder"].setdefault("params", {})[
                "feature_dim"
            ] = len(metadata_indices)
        model_settings_dict["metadata_encoder"]["params"]["output_dims"] = dmodel

    if "scale_function" in model_settings_dict:
        model_settings_dict["scale_function"].setdefault("params", {})[
            "input_dimension"
        ] = dmodel


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
    dataloader_settings = (
        DataLoaderSettings(**dataloader_settings_dict)
        if dataloader_settings_dict
        else None
    )

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
