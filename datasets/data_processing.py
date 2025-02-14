from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator
from sklearn.preprocessing import MinMaxScaler

from gragod import InterPolationMethods


# TODO:
#   - Check if swat labels are working, missing timestamps
#   - Improve data cleaning
def convert_df_to_tensor(df: pd.DataFrame) -> np.ndarray:
    """
    Convert a pandas DataFrame to a numpy array, exluding the timestamps.
    Args:
        df: The DataFrame to convert.
    Returns:
        The converted numpy array.
    """
    if df.shape[1] == 1:
        X = np.array(df.values[:, 0])
    else:
        X = np.array(df.values[:, 1:])
    X = np.vstack(X).astype(float)  # type:ignore

    return X


def interpolate_data(
    data: np.ndarray, method: InterPolationMethods | None = None
) -> np.ndarray:
    """
    Interpolate the missing values in the given data.
    Args:
        data: The data to interpolate.
        method: The interpolation method to use. Default is InterPolationMethods.LINEAR.
    Returns:
        The interpolated data.
    """

    method_str = (method or InterPolationMethods.LINEAR).value
    print(f"Interpolating data with method: {method_str}")
    df = pd.DataFrame(data)

    df.interpolate(method=method_str, inplace=True, order=3)
    interpolated_data = df.to_numpy()

    return interpolated_data


def normalize_data(data, scaler=None) -> Tuple[np.ndarray, BaseEstimator]:
    """
    Normalize the given data.
    Args:
        data: The data to normalize.
        scaler: The scaler to use for normalization.
    Returns:
        The normalized data and the scaler used.
    """
    data = np.asarray(data, dtype=np.float32)
    if np.any(sum(np.isnan(data))):
        data = np.nan_to_num(data)

    if scaler is None:
        scaler = MinMaxScaler()
        scaler.fit(data)
    data = scaler.transform(data)
    print("Data normalized")

    return data, scaler


def widen_labels(labels: np.ndarray, window_size: int = 2) -> np.ndarray:
    """
    Apply a rolling window to make the labels wider.
    Args:
        labels: The labels DataFrame to widen.
        window_size: The size of the window to apply.
    Returns:
        The widened labels DataFrame.
    """
    labels_df = pd.DataFrame(labels)
    labels_buffer = labels_df.copy()
    for col in labels_df.columns:
        if labels_df[col].sum() > 0:
            labels_buffer.loc[:, col] = (
                labels_df.loc[:, col]
                .rolling(2 * window_size + 1, center=True, min_periods=1)
                .sum()
                > 0
            ).astype(int)

    return labels_buffer.to_numpy()


def preprocess_df(
    data_df: pd.DataFrame,
    labels_df: pd.DataFrame | None = None,
    normalize: bool = False,
    clean: bool = False,
    scaler=None,
    interpolate_method: InterPolationMethods | None = None,
    max_std: float | None = None,
    labels_widening: bool = False,
    cutoff_value: float | None = None,
) -> Tuple[torch.Tensor, torch.Tensor | None, BaseEstimator | None]:
    """
    Preprocess the given data DataFrame.
    Args:
        data_df: The data DataFrame to preprocess.
        labels_df: The labels DataFrame to preprocess.
        normalize: Whether to normalize the data. Default is False.
        clean: Whether to clean the data. Default is False.
        scaler: The scaler to use for normalization.
    Returns:
        The preprocessed data and labels DataFrames.
    """
    # TODO: Improve this code, it's a mess

    data_df_copy = data_df.copy()
    labels_df_copy = labels_df.copy() if labels_df is not None else None

    if cutoff_value is not None:
        data_df_copy = data_df_copy.clip(upper=cutoff_value)

    data = interpolate_data(
        convert_df_to_tensor(data_df_copy), method=interpolate_method
    )
    labels = (
        interpolate_data(
            convert_df_to_tensor(labels_df_copy), method=interpolate_method
        )
        if labels_df_copy is not None
        else None
    )

    data_df_copy = pd.DataFrame(data)
    labels_df_copy = pd.DataFrame(labels)

    # Remove outliers
    if max_std is not None and max_std > 0.0:
        print(f"Using max_std: {max_std}")
        mean = data_df_copy.mean()
        std = data_df_copy.std()
        z_scores = (data_df_copy - mean) / std
        abs_z_scores = np.abs(z_scores)
        filtered_entries = abs_z_scores > max_std
        data_df_copy = data_df_copy.mask(filtered_entries)

        # Fill masked values with a cutoff value
        cutoff_value = mean + max_std * std
        data_df_copy = data_df_copy.fillna(cutoff_value)

    data = convert_df_to_tensor(data_df_copy)
    labels = (
        convert_df_to_tensor(labels_df_copy) if labels_df_copy is not None else None
    )

    if normalize:
        data, scaler = normalize_data(data, scaler)

    if clean:
        if labels is None:
            print("Skipping data cleaning, no labels provided")
        else:
            mask = labels == 1.0
            data[mask] = np.nan

    if labels_widening and labels is not None:
        print("Widening labels")
        labels = widen_labels(labels)

    print("Data cleaned!")

    data = torch.tensor(data).to(torch.float32)
    labels = torch.tensor(labels).to(torch.float32) if labels is not None else None

    return data, labels, scaler


def downsample_data(
    data: torch.Tensor, down_len: int, mode: str = "median"
) -> torch.Tensor:
    """
    Downsample the data by taking the median or mode of each downsample window.

    Args:
        data: The data to downsample (n_samples, n_features)
        down_len: The length of the downsample window.
        mode: The mode to use for downsampling. Default is "median".
    Returns:
        The downsampled data (n_samples // down_len, n_features)
    """
    # Reshape to (n_windows, window_size, n_features) and take median
    n_samples, n_features = data.shape
    n_windows = n_samples // down_len
    reshaped = data[: n_windows * down_len].reshape(n_windows, down_len, n_features)
    if mode == "median":
        return torch.median(reshaped, dim=1).values
    elif mode == "mode":
        return torch.mode(reshaped, dim=1).values
    else:
        raise ValueError(f"Invalid mode: {mode}")


def downsample(
    data: torch.Tensor, labels: torch.Tensor, down_len: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Downsample the data and labels.
    Args:
        data: The data to downsample (n_samples, n_features)
        labels: The labels to downsample (n_samples, n_features)
        down_len: The length of the downsample window.
    Returns:
        The downsampled data and labels (n_samples // down_len, n_features)
    """
    data_downsampled = downsample_data(data, down_len, mode="median")
    labels_downsampled = downsample_data(labels, down_len, mode="mode").round()
    if labels_downsampled.shape[0] != data_downsampled.shape[0]:
        raise ValueError(
            f"""Downsampled data and labels have different lengths
            Data shape {data_downsampled.shape},
            Labels shape {labels_downsampled.shape}"""
        )
    return data_downsampled, labels_downsampled
