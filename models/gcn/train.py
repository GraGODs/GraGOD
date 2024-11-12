import argparse

import torch
from torch.utils.data import DataLoader

from datasets.config import get_dataset_config
from datasets.dataset import SlidingWindowDataset
from gragod import CleanMethods, InterPolationMethods, ParamFileTypes, cast_dataset
from gragod.training import load_params, load_training_data, set_seeds
from models.gcn.model import GCN


def main(
    dataset_name: str,
    model_params: dict,
    test_size: float = 0.2,
    val_size: float = 0.2,
    clean: CleanMethods = CleanMethods.NONE,
    interpolate_method: InterPolationMethods | None = None,
    shuffle: bool = True,
    batch_size: int = 64,
    n_workers: int = 0,
    init_lr: float = 0.001,
    weight_decay: float = 0.0,
    n_epochs: int = 30,
    device: str = "mps",
    params: dict = {},
):
    """
    Main function to train and evaluate the GCN model.

    Args:
        dataset_name: Name of the dataset to use.
        model_params: Parameters for the GCN model.
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
    X_train, X_val, _, y_train, y_val, _ = load_training_data(
        dataset=dataset,
        test_size=test_size,
        val_size=val_size,
        normalize=dataset_config.normalize,
        clean=clean == CleanMethods.INTERPOLATE.value,
        interpolate_method=interpolate_method,
    )

    # TODO: we should load the edge_index (graph connectivity) from each dataset
    edge_index = (
        torch.tensor(
            [[i, j] for i in range(X_train.shape[1]) for j in range(X_train.shape[1])],
            dtype=torch.long,  # edge_index must be long type
        )
        .t()
        .to(device)
    )

    data_train = SlidingWindowDataset(
        X_train,
        edge_index=edge_index,
        window_size=model_params["window_size"],
        labels=y_train,
        drop=clean == CleanMethods.DROP.value,
    )
    data_val = SlidingWindowDataset(
        X_val,
        edge_index=edge_index,
        window_size=model_params["window_size"],
        labels=y_val,
        drop=clean == CleanMethods.DROP.value,
    )

    train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(data_val, batch_size=batch_size, shuffle=False)

    model = GCN(
        window_size=model_params["window_size"],
        n_layers=model_params["n_layers"],
        hidden_dim=model_params["hidden_dim"],
        K=model_params["K"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=init_lr, weight_decay=weight_decay
    )

    criterion = torch.nn.MSELoss()

    train_loss_list = []

    for i_epoch in range(n_epochs):
        optimizer.zero_grad()
        model.train()
        total_train_loss = torch.tensor(0.0, device=device)
        hidden_states = []
        outs = []
        for window, target, _, edge_index in train_loader:
            window = window.to(device)
            target = target.to(device)

            out, h = model(window.squeeze(0), edge_index)

            loss = criterion(out, target.squeeze(1))
            total_train_loss += loss

            hidden_states.append(h)
            outs.append(out)
            train_loss_list.append(loss.item())

        total_train_loss = total_train_loss / len(train_loader)
        total_train_loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        total_val_loss = torch.tensor(0.0, device=device)
        with torch.no_grad():
            for window, target, _, edge_index in val_loader:
                window = window.to(device)
                target = target.to(device)

                out, _ = model(window.squeeze(0), edge_index)
                loss = criterion(out, target.squeeze(1))
                total_val_loss += loss

        total_val_loss = total_val_loss / len(val_loader)
        print(
            f"Epoch {i_epoch},",
            f"Train Loss: {total_train_loss.item()},",
            f"Val Loss: {total_val_loss.item()}",
        )

    # TODO: add evaluation


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--params_file", type=str, default="models/gcn/params.yaml")
    args = parser.parse_args()

    params = load_params(args.params_file, file_type=ParamFileTypes.YAML)

    set_seeds(params["env_params"]["random_seed"])

    main(
        dataset_name=params["dataset"],
        **params["train_params"],
        model_params=params["model_params"],
        params=params,
    )
