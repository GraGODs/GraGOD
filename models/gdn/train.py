import argparse

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from datasets.config import get_dataset_config
from gragod import InterPolationMethods, ParamFileTypes
from gragod.training import (
    get_column_names_list,
    load_params,
    load_training_data,
    set_seeds,
)
from gragod.types import cast_dataset
from models.gdn.dataset import TimeDataset
from models.gdn.evaluate import print_score
from models.gdn.model import GDN
from models.gdn.preprocess import build_graph_network
from models.gdn.test import test


def _get_attack_or_not_attack(labels: torch.Tensor) -> torch.Tensor:
    """Gets the attack or not attack tensor.

    It's a tensor with 1 if the attack is present in any of the columns and 0 otherwise.

    Args:
        labels: The labels tensor.

    Returns:
        The attack or not attack tensor.
    """
    return (labels.sum(dim=1) > 0).float()


def get_dataloader(
    X: torch.Tensor,
    y: torch.Tensor,
    edge_index: torch.Tensor,
    batch_size: int,
    n_workers: int,
    config: dict,
    is_train: bool = False,
    shuffle: bool = True,
):
    """
    Creates and returns a DataLoader for the given dataset.

    Args:
        X: The input features.
        y: The target labels.
        edge_index: The edge indices for the graph structure.
        batch_size: The batch size for the DataLoader.
        n_workers: The number of worker processes for data loading.
        config: Configuration parameters for the TimeDataset.
        is_train: Whether this is a training dataset.

    Returns:
        The created DataLoader object.
    """
    dataset = TimeDataset(X, y, edge_index, is_train=is_train, config=config)
    return DataLoader(
        dataset, batch_size=batch_size, num_workers=n_workers, shuffle=shuffle
    )


def main(
    dataset_name: str,
    model_params: dict,
    test_size: float = 0.1,
    val_size: float = 0.1,
    clean: bool = True,
    interpolate_method: InterPolationMethods | None = None,
    shuffle: bool = True,
    batch_size: int = 264,
    n_workers: int = 0,
    init_lr: float = 0.001,
    weight_decay: float = 0.0,
    n_epochs: int = 30,
    device: str = "mps",
    params: dict = {},
):
    """
    Main function to train and evaluate the GDN model.

    Args:
        dataset_name: Name of the dataset to use.
        model_params: Parameters for the GDN model.
        test_size: Proportion of data to use for testing.
        val_size: Proportion of data to use for validation.
        clean: Whether to clean the data.
        interpolate_method: Method to use for interpolation.
        batch_size: Batch size for training.
        n_workers: Number of worker processes for data loading.
        init_lr: Initial learning rate.
        weight_decay: Weight decay for optimization.
        n_epochs: Number of training epochs.
        device: Device to use for training.
        params: Additional parameters.
    """
    dataset = cast_dataset(dataset_name)
    dataset_config = get_dataset_config(dataset=dataset)

    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_training_data(
        dataset=dataset,
        test_size=test_size,
        val_size=val_size,
        normalize=dataset_config.normalize,
        clean=clean,
        interpolate_method=interpolate_method,
    )
    y_train = _get_attack_or_not_attack(y_train)
    y_val = _get_attack_or_not_attack(y_val)
    y_test = _get_attack_or_not_attack(y_test)

    column_names_list = get_column_names_list(dataset)

    # Create a fully connected graph
    base_graph_structure = {
        ft: [other_ft for other_ft in column_names_list if other_ft != ft]
        for ft in column_names_list
    }

    fc_edge_index = build_graph_network(base_graph_structure, list(column_names_list))

    cfg = {
        "slide_win": params["model_params"]["window_size"],
        "slide_stride": params["model_params"]["stride"],
    }

    train_loader = get_dataloader(
        X_train,
        y_train,
        fc_edge_index,
        batch_size,
        n_workers,
        cfg,
        is_train=True,
        shuffle=shuffle,
    )
    val_loader = get_dataloader(
        X_val,
        y_val,
        fc_edge_index,
        batch_size,
        n_workers,
        cfg,
        is_train=False,
        shuffle=False,
    )
    test_loader = get_dataloader(
        X_test,
        y_test,
        fc_edge_index,
        batch_size,
        n_workers,
        cfg,
        is_train=False,
        shuffle=False,
    )

    model = GDN(
        [fc_edge_index],
        len(column_names_list),
        dim=model_params["dim"],
        input_dim=model_params["window_size"],
        out_layer_num=model_params["out_layer_num"],
        out_layer_inter_dim=model_params["out_layer_inter_dim"],
        topk=model_params["topk"],
    ).to(device)

    optimizer = torch.optim.Adam(  # type: ignore
        model.parameters(),
        lr=init_lr,
        weight_decay=weight_decay,
    )

    train_loss_list = []

    acu_loss = 0
    min_loss = 1e8

    i = 0
    early_stop_win = 15

    model.train()

    stop_improve_count = 0

    val_result = ([], [], [])

    for i_epoch in range(n_epochs):
        acu_loss = 0
        model.train()
        for x, labels, attack_labels, edge_index in train_loader:
            x, labels, edge_index = [
                item.float().to(device) for item in [x, labels, edge_index]
            ]

            optimizer.zero_grad()
            out = model(x).float().to(device)
            loss = F.mse_loss(out, labels, reduction="mean")

            loss.backward()
            optimizer.step()

            train_loss_list.append(loss.item())
            acu_loss += loss.item()

            i += 1

        # each epoch
        print(
            "epoch ({} / {}) (Loss:{:.8f}, ACU_loss:{:.8f})".format(
                i_epoch, n_epochs, acu_loss / len(train_loader), acu_loss
            ),
            flush=True,
        )

        # use val dataset to judge
        val_loss, val_result = test(model, val_loader, torch.device(device))

        if val_loss < min_loss:
            min_loss = val_loss
            stop_improve_count = 0
        else:
            stop_improve_count += 1

        if stop_improve_count >= early_stop_win:
            break

    _, test_result = test(model, test_loader, torch.device(device))

    print_score(test_result, val_result, params["env_params"]["report"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--params_file", type=str, default="models/gdn/params.yaml")
    args = parser.parse_args()

    params = load_params(args.params_file, file_type=ParamFileTypes.YAML)

    set_seeds(params["env_params"]["random_seed"])

    main(
        dataset_name=params["dataset"],
        **params["train_params"],
        model_params=params["model_params"],
        params=params,
    )
