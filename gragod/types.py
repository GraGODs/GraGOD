import os
from enum import Enum

PathType = str | os.PathLike


class Datasets(Enum):
    TELCO = "telco"
    MIHAELA = "mihaela"
    SWAT = "swat"


def cast_dataset(dataset: str) -> Datasets:
    try:
        return Datasets(dataset)
    except ValueError:
        raise ValueError(f"{dataset} is not a valid dataset")


class ParamFileTypes(Enum):
    YAML = "yaml"
    JSON = "json"


class InterPolationMethods(Enum):
    LINEAR = "linear"
    SPLINE = "spline"


class CleanMethods(Enum):
    NONE = "none"
    INTERPOLATE = "interpolate"
    DROP = "drop"


class Models(Enum):
    GRU = "gru"
    GCN = "gcn"
    GDN = "gdn"
    MTAD_GAT = "mtad_gat"
    DC_VAE = "dc_vae"


def cast_model(model: str) -> Models:
    try:
        return Models(model)
    except ValueError:
        raise ValueError(f"{model} is not a valid model")
