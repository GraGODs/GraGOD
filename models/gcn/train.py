import argparse

import torch
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from datasets.config import get_dataset_config
from datasets.dataset import SlidingWindowDataset
from gragod import CleanMethods, InterPolationMethods, ParamFileTypes, cast_dataset
from gragod.training import load_params, load_training_data, set_seeds
from gragod.training.callbacks import get_training_callbacks
from gragod.training.trainer import TrainerPL
from models.gcn.model import GCN, GCN_PLModule


def main(
    dataset_name: str,
    model_name: str,
    model_params: dict,
    params: dict,
    batch_size: int,
    n_epochs: int,
    init_lr: float,
    test_size: float,
    val_size: float,
    clean: CleanMethods,
    interpolate_method: InterPolationMethods | None,
    shuffle: bool,
    device: str,
    n_workers: int,
    log_dir: str,
    log_every_n_steps: int,
    ckpt_path: str | None,
    weight_decay: float,
    eps: float,
    betas: tuple[float, float],
    down_len: int | None = None,
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
        down_len=down_len,
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

    args_summary = {
        "dataset": dataset,
        "model_params": model_params,
        "train_params": params["train_params"],
    }

    logger = TensorBoardLogger(
        save_dir=log_dir, name=model_name, default_hp_metric=False
    )

    # Define callbacks
    callback_dict = get_training_callbacks(
        log_dir=logger.log_dir,
        model_name=model_name,
        monitor="Loss/val",
    )
    callbacks = list(callback_dict.values())
    trainer = TrainerPL(
        model=model,
        model_pl=GCN_PLModule,
        model_params=model_params,
        criterion=torch.nn.MSELoss(),
        batch_size=batch_size,
        n_epochs=n_epochs,
        init_lr=init_lr,
        device=device,
        log_dir=log_dir,
        callbacks=callbacks,
        checkpoint_cb=callback_dict["checkpoint"],
        logger=logger,
        log_every_n_steps=log_every_n_steps,
        weight_decay=weight_decay,
        eps=eps,
        betas=betas,
    )
    if ckpt_path:
        trainer.load(ckpt_path)

    trainer.fit(train_loader, val_loader, args_summary=args_summary)

    return trainer


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
