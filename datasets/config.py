from dataclasses import dataclass
from typing import Dict, Type

from gragod import Datasets


@dataclass
class Paths:
    base_path: str


@dataclass
class SWATPaths(Paths):
    base_path: str = "datasets_files/swat"
    name_train: str = "SWaT_data_train.csv"
    name_val: str = "SWaT_data_val.csv"
    name_test: str = "SWaT_data_test.csv"
    edge_index_path: str = "datasets_files/swat/edge_index.pt"


@dataclass
class TELCOPaths(Paths):
    base_path: str = "datasets_files/telco_v1"


@dataclass
class DatasetConfig:
    normalize: bool
    paths: Type[Paths]


@dataclass
class SWATConfig(DatasetConfig):
    normalize: bool = True
    paths: Type[Paths] = SWATPaths


@dataclass
class TELCOConfig(DatasetConfig):
    normalize: bool = True
    paths: Type[Paths] = TELCOPaths


def get_dataset_config(dataset: Datasets) -> DatasetConfig:
    DATASET_CONFIGS: Dict[Datasets, DatasetConfig] = {
        Datasets.SWAT: SWATConfig(),
        Datasets.TELCO: TELCOConfig(),
    }
    return DATASET_CONFIGS[dataset]
