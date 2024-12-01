import datetime
import os
from enum import Enum
from typing import List, Tuple

import networkx as nx
import pandas as pd
import torch
from sklearn.base import BaseEstimator

from datasets.data_processing import InterPolationMethods, preprocess_df

BASE_PATH_DEFAULT = "datasets_files/mihaela"
DEFAULT_DATES = [str(day) for day in range(20190501, 20190532)]


class SuffixName(Enum):
    DL = "DL"
    UL = "UL"


class Cities(Enum):
    Nancy = "Nancy"


def load_mihaela_service_df(
    service_name: str = "Clash_of_Clans",
    dates: List[str] = DEFAULT_DATES,
    base_path: str = BASE_PATH_DEFAULT,
    city_name: Cities = Cities.Nancy,
    suffix_name: SuffixName = SuffixName.DL,
) -> pd.DataFrame:
    """
    Loads the service data for a given city, service and suffix.
    Args:
        service_name: The name of the service.
        dates: The list of dates to load.
        base_path: The path to the datasets.
        city_name: The name of the city.
        suffix_name: The suffix of the file name.
    Returns:
        The DataFrame with the service data. In the first column is the
        is the time, in the other columns are the traffic values for each tile_id.
    """
    city_name = city_name.value
    all_dfs = []
    for day_str in dates:
        traffic_file = os.path.join(
            base_path,
            city_name,
            service_name,
            day_str,
            f"{city_name}_{service_name}_{day_str}_{suffix_name.value}.txt",
        )
        day = datetime.datetime.strptime(day_str, "%Y%m%d")
        times = [day + datetime.timedelta(minutes=15 * i) for i in range(96)]
        day_times_str = [str(day) + t.strftime("%H:%M") for t in times]

        column_names = ["tile_id"] + day_times_str

        df_traffic_dn = pd.read_csv(traffic_file, sep=" ", names=column_names)
        df_pivot = df_traffic_dn.melt(
            id_vars=["tile_id"], var_name="time", value_name="traffic"
        )
        df_pivot = df_pivot.pivot(
            index="time", columns="tile_id", values="traffic"
        ).reset_index()
        all_dfs.append(df_pivot)

    final_df = pd.concat(all_dfs, axis=0)
    return final_df


def load_mihaela_service_training_data(
    service_name: str,
    dates: List[str] = DEFAULT_DATES,
    base_path: str = BASE_PATH_DEFAULT,
    city_name: Cities = Cities.Nancy,
    normalize: bool = False,
    clean: bool = False,
    scaler=None,
    interpolate_method: InterPolationMethods | None = None,
) -> Tuple[torch.Tensor, torch.Tensor | None, BaseEstimator | None]:
    """
    Load the training data for the given service from Mihaela dataset.
    Args:
        service_name: The name of the service.
        dates: The list of dates to load.
        base_path: The path to the datasets.
        city_name: The enum of the city.
        normalize: Whether to normalize the data. Default is False.
        clean: Whether to clean the data. Default is False.
        scaler: The scaler to use for normalization.
        interpolate_method: The method to use for interpolation.
    """
    city_name = city_name.value
    df = load_mihaela_service_df(
        service_name=service_name,
        dates=dates,
        base_path=base_path,
        city_name=city_name,
    )

    return preprocess_df(
        data_df=df,
        normalize=normalize,
        clean=clean,
        scaler=scaler,
        interpolate_method=interpolate_method,
    )


def load_all_services(
    dates: list[str] = [str(day) for day in range(20190501, 20190532)],
    path_to_datasets: str = "../datasets_files/mihaela",
    city_name: str = "Nancy",
) -> pd.DataFrame:
    """
    Loads all the services for a given city and merges them into one DataFrame.
    Args:
        dates: The list of dates to load.
        path_to_datasets: The path to the datasets.
        city_name: The name of the city.
    Returns:
        The DataFrame with the service data. In the first column is the
        is the time, in the other columns are the traffic values for each tile_id.
    """
    service_dirs = os.listdir(os.path.join(path_to_datasets, city_name))
    all_data = []
    for service_name in service_dirs:
        for suffix in SuffixName:
            service_data = load_mihaela_service_df(
                service_name,
                dates=dates,
                path_to_datasets=path_to_datasets,
                city_name=city_name,
                suffix_name=suffix,
            )
            all_data.append(service_data)
    return all_data


def create_graph(df: pd.DataFrame, city_dims: tuple[int, int] = (151, 165)) -> nx.Graph:
    """
    Creates a graph from the DataFrame. Each tile_id is a node and the edges are
    created between the nodes that are next to each other.
    Args:
        df: The DataFrame with the service data.
        city_dims: The dimensions of the city.
    Returns:
        The networkx graph.
    """
    n_rows, n_cols = city_dims
    G = nx.Graph()
    for index, row in df[:10].iterrows():
        for tile_id, traffic_value in row.items():
            if tile_id == "time":
                continue
            tile_id = int(tile_id)
            row_index = tile_id // n_cols
            col_index = tile_id % n_cols
            node = (row_index, col_index)

            G.add_node(tile_id, traffic=traffic_value)

            # if the nodes are next to each other, add an edge
            # TODO: eventually try more sophisticated edge creation
            for adj in [
                (row_index - 1, col_index),
                (row_index + 1, col_index),
                (row_index, col_index - 1),
                (row_index, col_index + 1),
            ]:
                if adj in G.nodes:
                    G.add_edge(node, adj)
    return G
