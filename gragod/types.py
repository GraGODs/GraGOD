import os
from enum import Enum
from typing import Literal

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


MODEL_NAMES = Literal["gru", "gcn", "gdn", "mtad_gat"]
