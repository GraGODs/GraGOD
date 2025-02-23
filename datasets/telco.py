import os
from typing import Tuple

import pandas as pd
import torch

from datasets.config import TELCOConfig
from datasets.data_processing import InterPolationMethods, downsample, preprocess_df


def load_telco_df(
    base_path: str | os.PathLike = TELCOConfig.paths.base_path,
) -> Tuple[pd.DataFrame, ...]:
    """
    Load the TELCO datasets as pandas DataFrames from the given path.
    Args:
        base_path: The path where the datasets are stored.
    Returns:
        Tuple of DataFrames for train, validation, and test datasets.
    """
    df_train = pd.read_csv(os.path.join(base_path, str(TELCOConfig.paths.name_train)))
    df_train_labels = pd.read_csv(
        os.path.join(base_path, str(TELCOConfig.paths.name_train_labels))
    )
    df_val = pd.read_csv(os.path.join(base_path, str(TELCOConfig.paths.name_val)))
    df_val_labels = pd.read_csv(
        os.path.join(base_path, str(TELCOConfig.paths.name_val_labels))
    )
    df_test = pd.read_csv(os.path.join(base_path, str(TELCOConfig.paths.name_test)))
    df_test_labels = pd.read_csv(
        os.path.join(base_path, str(TELCOConfig.paths.name_test_labels))
    )

    return df_train, df_train_labels, df_val, df_val_labels, df_test, df_test_labels


def load_telco_tp(base_path: str | os.PathLike = TELCOConfig.paths.base_path):
    """
    Load the TELCO datasets as Temporian EventSets from the given path.
    Args:
        base_path: The path where the datasets are stored.
    Returns:
        Tuple of EventSets for train, validation, and test datasets.
    """
    import temporian as tp

    es_train = tp.from_csv(
        os.path.join(base_path, "TELCO_data_train.csv"), timestamps="time"
    )
    es_label_train = tp.from_csv(
        os.path.join(base_path, "TELCO_labels_train.csv"), timestamps="time"
    )
    es_val = tp.from_csv(
        os.path.join(base_path, "TELCO_data_val.csv"), timestamps="time"
    )
    es_label_val = tp.from_csv(
        os.path.join(base_path, "TELCO_labels_val.csv"), timestamps="time"
    )
    es_test = tp.from_csv(
        os.path.join(base_path, "TELCO_data_test.csv"), timestamps="time"
    )
    es_label_test = tp.from_csv(
        os.path.join(base_path, "TELCO_labels_test.csv"), timestamps="time"
    )

    return es_train, es_label_train, es_val, es_label_val, es_test, es_label_test


def load_telco_training_data(
    base_path: str | os.PathLike = TELCOConfig.paths.base_path,
    normalize: bool = False,
    clean: bool = False,
    scaler=None,
    interpolate_method: InterPolationMethods | None = None,
    down_len: int | None = None,
    max_std: float | None = None,
    labels_widening: bool = True,
    cutoff_value: float | None = None,
) -> Tuple[torch.Tensor, ...]:
    """
    Load the data for the telco dataset, splitted into train, val and test.
    Args:
        base_path: The path where the datasets are stored.
        normalize: Whether to normalize the data. Default is False.
        clean: Whether to clean the data. Default is False.
        scaler: The scaler to use for normalization.
        interpolate_method: The method to use for interpolation.
        down_len: The length of the downsample window.
                If None, no downsampling is performed.
        max_std: Maximum standard deviation for data cleaning. Default is 0.0.
        labels_widening: Whether to widen the labels. Default is True.
        cutoff_value: The cutoff value for data cleaning. Default is 30.0.
    Returns:
        Tuple of training data, training labels, validation data, validation labels,
        and test data.
    """
    (
        df_train,
        df_train_labels,
        df_val,
        df_val_labels,
        df_test,
        df_test_labels,
    ) = load_telco_df(base_path=base_path)

    # Drop timestamps from the dataframes (TODO: Add this to dataset config)
    columns_to_drop = [TELCOConfig.timestamp_column]
    df_train.drop(columns=columns_to_drop, inplace=True)
    df_train_labels.drop(columns=columns_to_drop, inplace=True)
    df_val.drop(columns=columns_to_drop, inplace=True)
    df_val_labels.drop(columns=columns_to_drop, inplace=True)
    df_test.drop(columns=columns_to_drop, inplace=True)
    df_test_labels.drop(columns=columns_to_drop, inplace=True)

    X_train, X_train_labels, scaler = preprocess_df(
        data_df=df_train,
        labels_df=df_train_labels,
        normalize=normalize,
        clean=clean,
        scaler=scaler,
        interpolate_method=interpolate_method,
        max_std=max_std,
        labels_widening=labels_widening,
        cutoff_value=cutoff_value,
    )
    X_val, X_val_labels, _ = preprocess_df(
        data_df=df_val,
        labels_df=df_val_labels,
        normalize=normalize,
        clean=clean,
        scaler=scaler,
        interpolate_method=interpolate_method,
        max_std=max_std,
        labels_widening=labels_widening,
        cutoff_value=cutoff_value,
    )
    X_test, X_test_labels, _ = preprocess_df(
        data_df=df_test,
        labels_df=df_test_labels,
        normalize=normalize,
        clean=False,
        scaler=scaler,
        interpolate_method=interpolate_method,
        max_std=max_std,
        labels_widening=labels_widening,
        cutoff_value=cutoff_value,
    )

    if X_train_labels is None or X_test_labels is None or X_val_labels is None:
        raise ValueError("Telco labels are not being loaded.")

    if down_len is not None:
        if down_len < 1:
            raise ValueError("Downsample length must be greater than 0")

        print(f"Downsampling data by {down_len}")
        X_train, X_train_labels = downsample(X_train, X_train_labels, down_len)
        X_val, X_val_labels = downsample(X_val, X_val_labels, down_len)
        X_test, X_test_labels = downsample(X_test, X_test_labels, down_len)

    return (
        X_train,
        X_val,
        X_test,
        X_train_labels,
        X_val_labels,
        X_test_labels,
    )
