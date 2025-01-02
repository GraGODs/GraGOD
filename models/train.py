import argparse
import json
import os
from typing import Any, Dict, Literal, Optional, Tuple, cast

import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn
from torch.nn import Module

from datasets.config import get_dataset_config
from datasets.dataset import get_data_loader, get_edge_index
from gragod import CleanMethods, InterPolationMethods, ParamFileTypes
from gragod.models import get_model_and_module
from gragod.training import load_params, load_training_data, set_seeds
from gragod.training.callbacks import get_training_callbacks
from gragod.training.trainer import TrainerPL
from gragod.types import Datasets, Models, cast_dataset

RANDOM_SEED = 42


def train(
    model: Models,
    dataset: Datasets,
    model_name: str,
    model_params: Dict[str, Any],
    params: Dict[str, Any],
    batch_size: int,
    n_epochs: int,
    init_lr: float,
    test_size: float,
    val_size: float,
    clean: CleanMethods,
    interpolate_method: Optional[InterPolationMethods],
    shuffle: bool,
    device: str,
    n_workers: int,
    log_dir: str,
    log_every_n_steps: int,
    weight_decay: float,
    eps: float,
    betas: Tuple[float, float],
    monitor: str = "Loss/val",
    monitor_mode: Literal["min", "max"] = "min",
    early_stop_patience: int = 20,
    early_stop_delta: float = 0.0001,
    save_top_k: int = 1,
    ckpt_path_resume: Optional[str] = None,
    down_len: Optional[int] = None,
    target_dims: Optional[int] = None,
    horizon: int = 1,
) -> TrainerPL:
    """Train a model on a dataset.

    Args:
        model: Model type to train
        dataset_name: Name of dataset to train on
        model_name: Name to give the model
        model_params: Model hyperparameters
        params: Full parameter dictionary
        batch_size: Training batch size
        n_epochs: Number of training epochs
        init_lr: Initial learning rate
        test_size: Fraction of data to use for testing
        val_size: Fraction of data to use for validation
        clean: Data cleaning method
        interpolate_method: Method for interpolating missing values
        shuffle: Whether to shuffle training data
        device: Device to train on
        n_workers: Number of data loading workers
        log_dir: Directory for logs
        log_every_n_steps: How often to log
        weight_decay: Weight decay factor
        eps: Small constant for numerical stability
        betas: Adam optimizer betas
        monitor: Metric to monitor
        monitor_mode: Whether to minimize or maximize monitored metric
        early_stop_patience: Patience for early stopping
        early_stop_delta: Minimum change for early stopping
        save_top_k: Number of best models to save
        ckpt_path_resume: Path to checkpoint to load
        down_len: Length to downsample to
        target_dims: Target dimensions to predict
        horizon: Prediction horizon

    Returns:
        Trained model trainer
    """
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

    print(f"Initial data shapes: Train: {X_train.shape}, Val: {X_val.shape}")

    edge_index = get_edge_index(X_train, device)

    train_loader = get_data_loader(
        X=X_train,
        edge_index=edge_index,
        y=y_train,
        window_size=model_params["window_size"],
        clean=clean,
        batch_size=batch_size,
        n_workers=n_workers,
        shuffle=shuffle,
    )

    val_loader = get_data_loader(
        X=X_val,
        edge_index=edge_index,
        y=y_val,
        window_size=model_params["window_size"],
        clean=clean,
        batch_size=batch_size,
        n_workers=n_workers,
        shuffle=shuffle,
    )

    model_class, model_pl_module = get_model_and_module(model)

    model_params["edge_index"] = [edge_index]
    model_params["n_features"] = X_train.shape[1]
    model_params["out_dim"] = X_train.shape[1]

    logger = TensorBoardLogger(
        save_dir=log_dir, name=model_name, default_hp_metric=False
    )

    callback_dict = get_training_callbacks(
        log_dir=logger.log_dir,
        model_name=model_name,
        monitor=monitor,
        monitor_mode=monitor_mode,
        early_stop_patience=early_stop_patience,
        early_stop_delta=early_stop_delta,
        save_top_k=save_top_k,
    )
    callbacks = list(callback_dict.values())

    criterion = (
        {
            "forecast_criterion": cast(Module, nn.MSELoss()),
            "recon_criterion": cast(Module, nn.MSELoss()),
        }
        if model_name.lower() in ["gru", "mtad_gat"]
        else cast(Module, nn.MSELoss())
    )

    model_instance = model_class(
        **model_params,
    ).to(device)

    trainer = TrainerPL(
        model=model_instance,
        model_pl=model_pl_module,
        model_params=model_params,
        criterion=criterion,
        batch_size=batch_size,
        n_epochs=n_epochs,
        init_lr=init_lr,
        device=device,
        log_dir=log_dir,
        callbacks=callbacks,
        checkpoint_cb=cast(ModelCheckpoint, callback_dict["checkpoint"]),
        logger=logger,
        target_dims=target_dims,
        log_every_n_steps=log_every_n_steps,
        weight_decay=weight_decay,
        eps=eps,
        betas=betas,
    )

    if ckpt_path_resume:
        trainer.load(ckpt_path_resume)
    model_params["edge_index"] = edge_index.tolist()

    args_summary = {
        "dataset": dataset.value,
        "model_params": model_params,
        "train_params": params["train_params"],
        "predictor_params": params["predictor_params"],
    }

    # Save args_summary to a file
    with open(os.path.join(log_dir, "args_summary.json"), "w") as f:
        json.dump(args_summary, f)

    model_params["edge_index"] = torch.tensor(model_params["edge_index"]).to(device)
    trainer.fit(train_loader, val_loader, args_summary=args_summary)

    return trainer


def main(model: Models, params_file: str) -> None:
    """Main training function.

    Args:
        model_name: Name of model to train
        params_file: Path to parameter file
    """
    params = load_params(params_file, file_type=ParamFileTypes.YAML)
    set_seeds(RANDOM_SEED)

    train(
        model=model,
        dataset=cast_dataset(params["dataset"]),
        **params["train_params"],
        model_params=params["model_params"],
        params=params,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Models,
        help=f"Model to train ({', '.join(model.value for model in Models)})",
    )
    parser.add_argument(
        "--params_file",
        type=str,
        default=None,
    )
    args = parser.parse_args()

    if args.params_file is None:
        args.params_file = f"models/{args.model.value}/params.yaml"

    main(args.model, args.params_file)
