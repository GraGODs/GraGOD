import argparse

import torch
import torch.nn as nn
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from datasets.config import get_dataset_config
from datasets.dataset import SlidingWindowDataset
from gragod import CleanMethods, InterPolationMethods, ParamFileTypes
from gragod.training import load_params, load_training_data, set_seeds
from gragod.training.trainer import TrainerPL
from gragod.types import cast_dataset
from models.gdn.model import GDN, GDN_PLModule


def _get_attack_or_not_attack(labels: torch.Tensor) -> torch.Tensor:
    """Gets the attack or not attack tensor.

    It's a tensor with 1 if the attack is present in any of the columns and 0 otherwise.

    Args:
        labels: The labels tensor.

    Returns:
        The attack or not attack tensor.
    """
    return (labels.sum(dim=1) > 0).float()


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
        clean=clean == CleanMethods.INTERPOLATE.value,
        interpolate_method=interpolate_method,
    )
    y_train = _get_attack_or_not_attack(y_train)
    y_val = _get_attack_or_not_attack(y_val)
    y_test = _get_attack_or_not_attack(y_test)

    # TODO: load this from each dataset
    # Create a fully connected graph
    edge_index = (
        torch.tensor(
            [[i, j] for i in range(X_train.shape[1]) for j in range(X_train.shape[1])],
            dtype=torch.long,  # edge_index must be long type
        )
        .t()
        .to(device)
    )

    train_dataset = SlidingWindowDataset(
        data=X_train,
        labels=y_train,
        edge_index=edge_index,
        window_size=model_params["window_size"],
        drop=clean == CleanMethods.DROP.value,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=n_workers, shuffle=shuffle
    )

    val_dataset = SlidingWindowDataset(
        data=X_val,
        labels=y_val,
        edge_index=edge_index,
        window_size=model_params["window_size"],
        drop=clean == CleanMethods.DROP.value,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, num_workers=n_workers, shuffle=False
    )

    model = GDN(
        [edge_index],
        X_train.shape[1],
        embed_dim=model_params["embed_dim"],
        input_dim=model_params["window_size"],
        out_layer_num=model_params["out_layer_num"],
        out_layer_inter_dim=model_params["out_layer_inter_dim"],
        topk=model_params["topk"],
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
    early_stop = EarlyStopping(
        monitor="Loss/val",
        min_delta=0.0001,
        patience=2,
        verbose=True,
        mode="min",
    )
    checkpoint = ModelCheckpoint(
        monitor="Loss/val",
        dirpath=logger.log_dir,
        filename=model_name,
        save_top_k=1,
        mode="min",
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    callbacks = [early_stop, checkpoint, lr_monitor]

    trainer = TrainerPL(
        model=model,
        model_pl=GDN_PLModule,
        model_params=model_params,
        criterion=nn.MSELoss(),
        batch_size=batch_size,
        n_epochs=n_epochs,
        init_lr=init_lr,
        device=device,
        log_dir=log_dir,
        callbacks=callbacks,
        checkpoint_cb=checkpoint,
        logger=logger,
        log_every_n_steps=log_every_n_steps,
    )
    if ckpt_path:
        trainer.load(ckpt_path)

    trainer.fit(train_loader, val_loader, args_summary=args_summary)

    # TODO: Run prediction with the best model, not the last one
    # _, test_result = test(model, test_loader, torch.device(device))
    # print_score(test_result, val_result, params["env_params"]["report"])


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
