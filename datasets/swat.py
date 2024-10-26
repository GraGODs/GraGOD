import os
from typing import Tuple

import pandas as pd
import torch

from datasets.config import SWATPaths
from datasets.data_processing import InterPolationMethods, preprocess_df


def load_swat_df_train(
    name: str = SWATPaths.NAME_TRAIN.value,
    path_to_dataset: str = SWATPaths.BASE_PATH.value,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads the training dataset from the given path and returns a pandas DataFrame.
    Args:
        path_to_dataset: Path to the dataset files.
    Returns:
        A pandas DataFrame containing the training dataset.
    """
    file = os.path.join(path_to_dataset, name)
    df_train = pd.read_csv(file)
    df_train_labels = (df_train["Normal/Attack"] == "Attack").astype(int)
    df_train_labels = df_train_labels.to_frame()
    df_train = df_train.drop(columns=["Normal/Attack"])

    return df_train, df_train_labels


def load_swat_df_val(
    name: str = SWATPaths.NAME_VAL.value,
    path_to_dataset: str = SWATPaths.BASE_PATH.value,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads the validation dataset from the given path and returns a pandas DataFrame.
    Args:
        path_to_dataset: Path to the dataset files.
    Returns:
        A pandas DataFrame containing the validation dataset.
    """
    file = os.path.join(path_to_dataset, name)
    df_val = pd.read_csv(file)
    df_val_labels = (df_val["Normal/Attack"] == "Attack").astype(int)
    df_val_labels = df_val_labels.to_frame()
    df_val = df_val.drop(columns=["Normal/Attack"])
    return df_val, df_val_labels


def split_val_df(
    df_val: pd.DataFrame, df_val_labels: pd.DataFrame, val_size: float = 0.6
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the validation dataset into two parts: validation and test.
    Args:
        df_val: The DataFrame with the validation data.
        df_val_labels: The labels of the validation data.
        val_size: The size of the validation set.
    Returns:
        The validation and test datasets.
    """
    val_size = int(val_size * len(df_val))
    df_test = df_val.iloc[val_size:]
    df_test_labels = df_val_labels.iloc[val_size:]
    df_val = df_val.iloc[:val_size]
    df_val_labels = df_val_labels.iloc[:val_size]
    return df_val, df_val_labels, df_test, df_test_labels


def load_swat_df(
    path_to_dataset: str = SWATPaths.BASE_PATH.value, val_size: float = 0.6
) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    """
    Loads the dataset from the given path and returns a pandas DataFrame.
    Args:
        names: List of names of the files to load.
        path_to_dataset: Path to the dataset files.
    Returns:
        A pandas DataFrame containing the dataset.
    """
    df_train, df_train_labels = load_swat_df_train(path_to_dataset=path_to_dataset)
    df_val, df_val_labels = load_swat_df_val(path_to_dataset=path_to_dataset)
    df_val, df_val_labels, df_test, df_test_labels = split_val_df(
        df_val, df_val_labels, val_size=val_size
    )

    return df_train, df_train_labels, df_val, df_val_labels, df_test, df_test_labels


def load_swat_training_data(
    path_to_dataset: str = SWATPaths.BASE_PATH.value,
    normalize: bool = False,
    clean: bool = False,
    scaler=None,
    interpolate_method: InterPolationMethods | None = None,
) -> Tuple[torch.Tensor, ...]:
    """
    Loads the training dataset from the given path and returns a pandas DataFrame.
    Args:
        path_to_dataset: Path to the dataset files.
        normalize: Whether to normalize the data. Default is False.
        clean: Whether to clean the data. Default is False.
        scaler: The scaler to use for normalization.
        interpolate_method: The method to use for interpolation.
    Returns:
        A pandas DataFrame containing the training dataset.
    """
    (
        df_train,
        df_train_labels,
        df_val,
        df_val_labels,
        df_test,
        df_test_labels,
    ) = load_swat_df(path_to_dataset=path_to_dataset)
    X_train, X_train_labels = preprocess_df(
        data_df=df_train,
        labels_df=df_train_labels,
        normalize=normalize,
        clean=clean,
        scaler=scaler,
        interpolate_method=interpolate_method,
    )
    X_val, X_val_labels = preprocess_df(
        data_df=df_val,
        labels_df=df_val_labels,
        normalize=normalize,
        clean=False,
        scaler=scaler,
        interpolate_method=interpolate_method,
    )
    X_test, X_test_labels = preprocess_df(
        data_df=df_test,
        labels_df=df_test_labels,
        normalize=normalize,
        clean=False,
        scaler=scaler,
        interpolate_method=interpolate_method,
    )

    if X_train_labels is None or X_test_labels is None or X_val_labels is None:
        raise ValueError("SWAT labels are not being loaded.")

    return X_train, X_val, X_test, X_train_labels, X_val_labels, X_test_labels


def get_swat_column_names_list() -> list[str]:
    return list(load_swat_df_train()[0].columns)
