import dataclasses
import torch
import torch
import torch.nn.functional as F
from networks import *
import distributions
import get_protein_data
import reciprocalspaceship as rs
from abismal_torch.prior import WilsonPrior

from rasu import *
from abismal_torch.likelihood import NormalLikelihood
from abismal_torch.surrogate_posterior import FoldedNormalPosterior
from wrap_folded_normal import FrequencyTrackingPosterior

import shoebox_encoder
import metadata_encoder


@dataclasses.dataclass
class ModelSettings():
    run_from_version: str | None = None

    build_background_distribution: type = distributions.HalfNormalDistribution
    build_profile_distribution: type = distributions.Distribution
    # build_shoebox_profile_distribution = distributions.LRMVN_Distribution
    build_shoebox_profile_distribution: type = distributions.DirichletProfile
    # background_prior_distribution: torch.distributions.Gamma = dataclasses.field(
    #     default_factory=lambda: torch.distributions.Gamma(concentration=torch.tensor(1.98), rate=torch.tensor(1/75.4))
    # )
    background_prior_distribution: torch.distributions.Gamma = dataclasses.field(
        default_factory=lambda: torch.distributions.HalfNormal(0.5)
    )
    # scale_prior_distibution: torch.distributions.HalfNormal = dataclasses.field(
    #     default_factory=lambda: torch.distributions.HalfNormal(2.0)
    # )
    scale_prior_distibution: torch.distributions.Gamma = dataclasses.field(
        default_factory=lambda: torch.distributions.Gamma(concentration=torch.tensor(6.68), rate=torch.tensor(6.4463))
    )
    scale_function: MLPScale = dataclasses.field(
        default_factory=lambda: MLPScale(
            input_dimension=64,
            scale_distribution=LogNormalDistributionLayer(hidden_dimension=64),
            hidden_dimension=64,
            number_of_layers=1,
            initial_scale_guess=2/140
        )
    )
    build_intensity_prior_distribution: WilsonPrior = WilsonPrior
    intensity_prior_distibution: WilsonPrior = dataclasses.field(init=False)
    
    shoebox_encoder: type = shoebox_encoder.BaseShoeboxEncoder()
    metadata_encoder: type = metadata_encoder.BaseMetadataEncoder()
    metadata_depth: int = 10
    metadata_indices_to_keep: list = dataclasses.field(default_factory=lambda: [0, 1])



    optimizer: type = torch.optim.AdamW
    learning_rate: float = 0.001
    dmodel: int = 64
    # batch_size = 4
    number_of_epochs: int = 5
    number_of_mc_samples: int = 20

    data_directory: str = "/n/hekstra_lab/people/aldama/subset/small_dataset/pass1"
    data_file_names: dict = dataclasses.field(default_factory=lambda: {
        # "shoeboxes": "standardized_counts.pt",
        "counts": "counts.pt",
        "metadata": "reference.pt",
        "masks": "masks.pt",
    })

    enable_checkpointing: bool = True
    lysozyme_sequence_file_path: str = "/n/holylabs/LABS/hekstra_lab/Users/fgiehr/factory/data/lysozyme.seq"

    merged_mtz_file_path: str = "/n/holylabs/LABS/hekstra_lab/Users/fgiehr/creat_dials_unmerged/merged.mtz"
    unmerged_mtz_file_path: str = "/n/holylabs/LABS/hekstra_lab/Users/fgiehr/creat_dials_unmerged/unmerged.mtz"
    protein_pdb_url: str = "https://files.rcsb.org/download/9B7C.cif"
    rac: ReciprocalASUCollection = dataclasses.field(init=False)
    pdb_data: dict = dataclasses.field(init=False)
    def __post_init__(self):
        self.pdb_data = get_protein_data.get_protein_data(self.protein_pdb_url)
        self.rac = ReciprocalASUGraph(*[ReciprocalASU(
            cell=self.pdb_data["unit_cell"],
            spacegroup=self.pdb_data["spacegroup"],
            dmin=float(self.pdb_data["dmin"]),
            anomalous=True,
        )])
        self.intensity_prior_distibution = self.build_intensity_prior_distribution(self.rac)


@dataclasses.dataclass
class LossSettings():
    prior_background_weight: float = 0.0001
    prior_structure_factors_weight: float = 0.0001
    prior_scale_weight: float = 0.0001
    prior_profile_weight: list[float] = dataclasses.field(default_factory=lambda: [0.0001, 0.0001, 0.0001])
    eps: float = 0.00001

@dataclasses.dataclass
class PhenixSettings():
    r_values_reference_path: str = "/n/holylabs/LABS/hekstra_lab/Users/fgiehr/jobs/anomalous_peaks_files/pdb_model/refine_001.log"


@ dataclasses.dataclass
class DataLoaderSettings():
    # data_directory: str = "/n/hekstra_lab/people/aldama/subset"
    # data_file_names: dict = dataclasses.field(default_factory=lambda: {
    #     "shoeboxes": "shoebox_subset.pt",
    #     "counts": "raw_counts_subset.pt",
    #     "metadata": "metadata_subset.pt",
    #     "masks": "mask_subset.pt",
    #     "true_reference": "true_reference_subset.pt",
    # })
    metadata_indices: dict = dataclasses.field(default_factory=lambda: {
        "d": 0,
        "h": 1,
        "k": 2,
        "l": 3,
        "x": 4,
        "y": 5,
        "z": 6,
    })
    metadata_keys_to_keep: list = dataclasses.field(default_factory=lambda: [
        "x", "y"
    ])


    data_directory: str = "/n/hekstra_lab/people/aldama/subset/small_dataset/pass1"
    data_file_names: dict = dataclasses.field(default_factory=lambda: {
        # "shoeboxes": "standardized_counts.pt",
        "counts": "counts.pt",
        "metadata": "reference.pt",
        "masks": "masks.pt",
    })

    validation_set_split: float = 0.2
    test_set_split: float = 0
    number_of_images_per_batch: int = 1
    number_of_shoeboxes_per_batch: int = 3000
    number_of_batches: int = 1444
    number_of_workers: int = 16
    pin_memory: bool = True
    prefetch_factor: int|None = 2
    shuffle_indices: bool = True
    shuffle_groups: bool = True
    optimize_shoeboxes_per_batch: bool = True
    append_image_id_to_metadata: bool = False
    verbose: bool = True