import json

import pytorch_lightning as pl
import torch
import yaml

from datasets import (
    load_swat_training_data,
    load_telco_training_data,
    load_ute_training_data,
)
from gragod import Datasets, InterPolationMethods, ParamFileTypes
from gragod.utils import get_logger

logger = get_logger()


def set_seeds(seed):
    """
    Set the seeds for the random number generators on pytorch, numpy and python.
    Args:
        seed: The seed to set.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    pl.seed_everything(seed, workers=True)


def load_params(base_path: str, file_type: ParamFileTypes) -> dict:
    """
    Load the parameters from the given file.
    Args:
        base_path: The path to the parameters file.
        type: The enum with the type of the parameters file.
    Returns:
        The parameters as a dictionary.
    """
    if file_type == ParamFileTypes.YAML:
        with open(base_path, "r") as yaml_file:
            params = yaml.safe_load(yaml_file)
    elif file_type == ParamFileTypes.JSON:
        with open(base_path, "r") as json_file:
            params = json.load(json_file)
    else:
        raise ValueError(f"Type must be one of {ParamFileTypes.__members__.keys()}")

    return params


def load_training_data(
    dataset: Datasets,
    test_size: float = 0.1,
    val_size: float = 0.1,
    normalize: bool = False,
    clean: bool = False,
    interpolate_method: InterPolationMethods | None = None,
    down_len: int | None = None,
    max_std: float | None = None,
    labels_widening: bool = False,
    cutoff_value: float | None = None,
):
    """
    Load the training data for the given dataset.
    Args:
        dataset: The dataset to load.
        test_size: The size of the test set, from 0 to 1.
        val_size: The size of the validation set, from 0 to 1.
        normalize: Whether to normalize the data.
        clean: Whether to remove anomalies from the data.
        interpolate_method: The method to use to interpolate the missing values.
        down_len: The length of the downsample window.
                If None, no downsampling is performed.
        max_std: Maximum standard deviation for data cleaning
        labels_widening: Whether to widen the labels.
        cutoff_value: The cutoff value for data cleaning.
    Returns:
        The training, validation and test data as torch tensors.
    """
    # TODO:
    # - Fix datasets loading path
    # - Load all services in Miahela

    # if dataset == "mihaela":
    #     from datasets.mihaela import load_mihaela_service_training_data

    #     X, Y = load_mihaela_service_training_data(
    #         service_name="Clash_of_Clans",
    #         normalize=normalize,
    #         clean=clean,
    #         interpolate_method=interpolate_method,
    #     )

    #     return _split_train_val_test(
    #         X=X,
    #         Y=Y,
    #         test_size=test_size,
    #         val_size=val_size,
    #         shuffle=shuffle,
    #         random_state=random_state,
    #     )

    if dataset == Datasets.SWAT:
        return load_swat_training_data(
            normalize=normalize,
            clean=clean,
            interpolate_method=interpolate_method,
            down_len=down_len,
            max_std=max_std,
            labels_widening=labels_widening,
            cutoff_value=cutoff_value,
        )
    elif dataset == Datasets.TELCO:
        logger.warning(
            "Telco data is already splited, ignoring arguments: 'test_size'"
            ", 'val_size', 'shuffle' and 'random_state'"
        )

        return load_telco_training_data(
            normalize=normalize,
            clean=clean,
            interpolate_method=interpolate_method,
            down_len=down_len,
            max_std=max_std,
            labels_widening=labels_widening,
            cutoff_value=cutoff_value,
        )
    elif dataset == Datasets.UTE:
        return load_ute_training_data(
            normalize=normalize,
            clean=clean,
            interpolate_method=interpolate_method,
            down_len=down_len,
        )
    else:
        raise ValueError(f"{dataset} is an unkown dataset")
