import os
from typing import Tuple

import pandas as pd
import torch

from datasets.config import SWATPaths
from datasets.data_processing import InterPolationMethods, downsample, preprocess_df


def load_swat_df_train(
    name: str = SWATPaths.name_train,
    path_to_dataset: str = SWATPaths.base_path,
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
    name: str = SWATPaths.name_val,
    path_to_dataset: str = SWATPaths.base_path,
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


def load_swat_df_test(
    name: str = SWATPaths.name_test,
    path_to_dataset: str = SWATPaths.base_path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    file = os.path.join(path_to_dataset, name)
    df_test = pd.read_csv(file)
    df_test_labels = (df_test["Normal/Attack"] == "Attack").astype(int)
    df_test_labels = df_test_labels.to_frame()
    df_test = df_test.drop(columns=["Normal/Attack"])
    return df_test, df_test_labels


def load_swat_df(
    path_to_dataset: str = SWATPaths.base_path, val_size: float = 0.6
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
    df_test, df_test_labels = load_swat_df_test(path_to_dataset=path_to_dataset)

    return df_train, df_train_labels, df_val, df_val_labels, df_test, df_test_labels


def load_swat_training_data(
    path_to_dataset: str = SWATPaths.base_path,
    normalize: bool = False,
    clean: bool = False,
    scaler=None,
    interpolate_method: InterPolationMethods | None = None,
    down_len: int | None = None,
    max_std: float | None = None,
    labels_widening: bool = False,
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
    ) = load_swat_df(path_to_dataset=path_to_dataset)

    # Drop timestamps from the dataframes (TODO: Add this to dataset config)
    columns_to_drop = [" Timestamp"]
    df_train = df_train.drop(columns=columns_to_drop)
    df_val = df_val.drop(columns=columns_to_drop)
    df_test = df_test.drop(columns=columns_to_drop)

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
        raise ValueError("SWAT labels are not being loaded.")

    # The first 6 hours worth of training data is discarded since the
    # system takes 5-6 hours to stabilize.
    six_hours_in_seconds = 6 * 60 * 60
    X_train = X_train[six_hours_in_seconds:]
    X_train_labels = X_train_labels[six_hours_in_seconds:]

    if down_len is not None:
        if down_len < 1:
            raise ValueError("Downsample length must be greater than 0")

        print(f"Downsampling data by {down_len}")
        X_train, X_train_labels = downsample(X_train, X_train_labels, down_len)
        X_val, X_val_labels = downsample(X_val, X_val_labels, down_len)
        X_test, X_test_labels = downsample(X_test, X_test_labels, down_len)

    return X_train, X_val, X_test, X_train_labels, X_val_labels, X_test_labels


def build_swat_edge_index(
    X: torch.Tensor, device: str, path: str = SWATPaths.edge_index_path
) -> torch.Tensor:
    """
    Build the edge index based on the topological structure of SWaT system.

    The function looks for pairs of node names that are identical
    except for their last character, and creates edges between their
    corresponding indices in the graph.

    This way, sensors that measure similar signals in similar steps
    are connected together.

    Args:
        X: Input tensor containing the node features (not used in edge construction)
        device: Device to place the resulting edge_index tensor on ('cpu' or 'cuda')

    Returns:
        torch.Tensor: A tensor of shape [2, num_edges] containing the edge indices.
    """
    df, _ = load_swat_df_val(path_to_dataset="../datasets_files/swat")
    node_names = df.columns.tolist()

    edges = []
    for i, name1 in enumerate(node_names):
        for j, name2 in enumerate(node_names):
            if i != j:
                stripped_name1 = name1.strip()
                stripped_name2 = name2.strip()

                # Check if names are identical except for last character
                if (
                    len(stripped_name1) == len(stripped_name2)
                    and stripped_name1[:-1] == stripped_name2[:-1]
                ):
                    edges.append([i, j])

    edge_index = torch.tensor(edges, dtype=torch.long).t().to(device)

    torch.save(edge_index, path)

    return edge_index
