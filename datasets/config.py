from dataclasses import dataclass
from typing import Dict, Generic, TypeVar

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
    name_train: str = "TELCO_data_train.csv"
    name_train_labels: str = "TELCO_labels_train.csv"
    name_val: str = "TELCO_data_val.csv"
    name_val_labels: str = "TELCO_labels_val.csv"
    name_test: str = "TELCO_data_test.csv"
    name_test_labels: str = "TELCO_labels_test.csv"


P = TypeVar("P", bound=Paths)


@dataclass
class DatasetConfig(Generic[P]):
    normalize: bool
    paths: P
    timestamp_column: str


@dataclass
class SWATConfig(DatasetConfig[SWATPaths]):
    normalize: bool = True
    paths: SWATPaths = SWATPaths()
    timestamp_column: str = " Timestamp"


@dataclass
class TELCOConfig(DatasetConfig[TELCOPaths]):
    normalize: bool = True
    paths: TELCOPaths = TELCOPaths()
    timestamp_column: str = "time"


def get_dataset_config(dataset: Datasets) -> DatasetConfig:
    DATASET_CONFIGS: Dict[Datasets, DatasetConfig] = {
        Datasets.SWAT: SWATConfig(),
        Datasets.TELCO: TELCOConfig(),
    }
    return DATASET_CONFIGS[dataset]
