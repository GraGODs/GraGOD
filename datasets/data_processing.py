from typing import Literal, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

# from sklearn.impute import KNNImputer, SimpleImputer

INTERPOLATION_METHODS = Literal["linear", "spline"]


def convert_df_to_tensor(df: pd.DataFrame) -> np.ndarray:
    """
    Convert a pandas DataFrame to a numpy array, exluding the timestamps.
    Args:
        df: The DataFrame to convert.
    Returns:
        The converted numpy array.
    """
    X = np.array(df.values[:, 1:])
    X = np.vstack(X).astype(float)  # type:ignore

    return X


def interpolate_data(
    data: np.ndarray, method: Optional[INTERPOLATION_METHODS] = None
) -> np.ndarray:
    """
    Interpolate the missing values in the given data.
    Args:
        data: The data to interpolate.
        method: The interpolation method to use. Default is "spline".
    Returns:
        The interpolated data.
    """

    method = method or "spline"
    df = pd.DataFrame(data)

    df.interpolate(method=method, inplace=True, order=3)
    interpolated_data = df.to_numpy()

    return interpolated_data


def normalize_data(data, scaler=None) -> Tuple[np.ndarray, MinMaxScaler]:
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


def preprocess_df(
    data_df: pd.DataFrame,
    labels_df: Optional[pd.DataFrame] = None,
    normalize: bool = False,
    clean: bool = False,
    scaler=None,
    interpolate_method: Optional[INTERPOLATION_METHODS] = None,
) -> Tuple[torch.Tensor, torch.Tensor | None]:
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
    data = convert_df_to_tensor(data_df)
    labels = convert_df_to_tensor(labels_df) if labels_df is not None else None

    if normalize:
        data, scaler = normalize_data(data, scaler)

    if clean:
        if labels is None:
            print("Skipping data cleaning, no labels provided")
        else:
            mask = labels == 1.0
            data[mask] = np.nan

    data = interpolate_data(data, method=interpolate_method)
    print("Data cleaned!")

    data = torch.tensor(data).to(torch.float32)
    labels = torch.tensor(labels).to(torch.float32) if labels is not None else None

    return data, labels
