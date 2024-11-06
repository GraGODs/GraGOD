from typing import Dict

from pydantic import BaseModel

from gragod import Datasets


class Paths(BaseModel):
    base_path: str


class SWATPaths(Paths):
    base_path: str = "datasets_files/swat"
    name_train: str = "SWaT_data_train.csv"
    name_val: str = "SWaT_data_val.csv"


class TELCOPaths(Paths):
    base_path: str = "datasets_files/telco"


class DatasetConfig(BaseModel):
    normalize: bool
    paths: type[Paths]


class SWATConfig(DatasetConfig):
    normalize: bool = False
    paths: type[Paths] = SWATPaths


class TELCOConfig(DatasetConfig):
    normalize: bool = False
    paths: type[Paths] = TELCOPaths


def get_dataset_config(dataset: Datasets) -> DatasetConfig:
    DATASET_CONFIGS: Dict[Datasets, DatasetConfig] = {
        Datasets.SWAT: SWATConfig(),
        Datasets.TELCO: TELCOConfig(),
    }
    return DATASET_CONFIGS[dataset]
