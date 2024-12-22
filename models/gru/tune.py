import argparse
import json
import os
from pathlib import Path
from time import time

import optuna
import torch
import yaml
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn
from torch.utils.data import DataLoader

from datasets.config import get_dataset_config
from datasets.dataset import SlidingWindowDataset
from gragod import CleanMethods, ParamFileTypes
from gragod.metrics import get_metrics, print_all_metrics
from gragod.predictions.prediction import get_threshold
from gragod.training import load_params, load_training_data, set_seeds
from gragod.training.callbacks import get_training_callbacks
from gragod.training.trainer import TrainerPL
from gragod.types import Datasets, cast_dataset
from models.gru.model import GRU_PLModule, GRUModel
from models.gru.predict import generate_scores, run_model

RANDOM_SEED = 42
set_seeds(RANDOM_SEED)

# TODO:
# - Add batch finder


def get_experiment_metrics(
    trainer: TrainerPL,
    val_loader: DataLoader,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    model_params: dict,
    params: dict,
    dataset: Datasets,
):
    # Get metrics
    forecasts = run_model(
        model=trainer.lightning_module,  # type: ignore
        loader=val_loader,
        device=params["train_params"]["device"],
    )
    scores = generate_scores(
        forecasts=forecasts,
        data=X_val,
        window_size=model_params["window_size"],
    )

    threshold = get_threshold(
        dataset=dataset,
        scores=scores,
        labels=y_val[model_params["window_size"] :],
        n_thresholds=params["predictor_params"]["n_thresholds"],
    )
    pred = scores > threshold
    metrics = get_metrics(
        predictions=pred,
        labels=y_val[model_params["window_size"] :],
        scores=scores,
        dataset=dataset,
    )
    return metrics


def create_datasets(
    X: torch.Tensor,
    y: torch.Tensor,
    window_size: int,
    horizon: int,
    batch_size: int,
    shuffle: bool = True,
    drop: bool = False,
    n_workers: int = 0,
):
    """Create dataset and dataloader with specific window size."""
    dataset = SlidingWindowDataset(
        X,
        window_size=window_size,
        horizon=horizon,
        labels=y,
        drop=drop,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=n_workers,
        persistent_workers=n_workers > 0,
        # pin_memory=True,
    )

    return dataloader


def objective(
    trial: optuna.Trial,
    params: dict,
    X_train: torch.Tensor,
    X_val: torch.Tensor,
    y_train: torch.Tensor,
    y_val: torch.Tensor,
    log_dir: Path,
    dataset: Datasets,
):
    """Optuna objective function."""
    start_time = time()
    print(f"Trial number: {trial.number}")

    # Get trial hyperparameters
    model_params = {
        "hidden_size": trial.suggest_categorical("hidden_size", [16, 32, 64, 128, 256]),
        "n_layers": trial.suggest_categorical("n_layers", [1, 3, 5, 7]),
        "bidirectional": True,
        "rnn_dropout": trial.suggest_categorical("rnn_dropout", [0.1, 0.2, 0.3]),
        "fc_dropout": trial.suggest_categorical("fc_dropout", [0.1, 0.2, 0.3]),
    }

    train_params_search = {
        "window_size": trial.suggest_categorical("window_size", [64, 128, 256, 512]),
        "init_lr": trial.suggest_categorical("init_lr", [1e-4, 1e-3]),
        "weight_decay": params["train_params"]["weight_decay"],
        "eps": params["train_params"]["eps"],
        "betas": params["train_params"]["betas"],
    }

    # Create datasets
    train_loader = create_datasets(
        X_train,
        y_train,
        window_size=train_params_search["window_size"],
        horizon=params["train_params"]["horizon"],
        batch_size=params["train_params"]["batch_size"],
        shuffle=params["train_params"]["shuffle"],
        drop=params["train_params"]["clean"] == CleanMethods.DROP.value,
        n_workers=params["train_params"]["n_workers"],
    )
    val_loader = create_datasets(
        X_val,
        y_val,
        window_size=train_params_search["window_size"],
        horizon=params["train_params"]["horizon"],
        batch_size=params["train_params"]["batch_size"],
        shuffle=False,
        drop=params["train_params"]["clean"] == CleanMethods.DROP.value,
        n_workers=params["train_params"]["n_workers"],
    )
    inference_train_loader = create_datasets(
        X_train,
        y_train,
        window_size=train_params_search["window_size"],
        horizon=params["train_params"]["horizon"],
        batch_size=params["train_params"]["batch_size"],
        shuffle=False,
        drop=False,
        n_workers=params["train_params"]["n_workers"],
    )

    # Create model
    model = GRUModel(
        n_features=X_train.shape[1],
        out_dim=X_train.shape[1],
        **model_params,
    )

    # Define callbacks and logger
    logger = TensorBoardLogger(
        save_dir=log_dir, name=f"trial_{trial.number}", default_hp_metric=False
    )

    callbacks = get_training_callbacks(
        log_dir=logger.log_dir,
        model_name="best",
        monitor="Loss/val",
        monitor_mode="min",
        early_stop_patience=20,
    )

    # Create trainer
    trainer = TrainerPL(
        model=model,
        model_pl=GRU_PLModule,
        model_params=model_params,
        criterion={"forecast_criterion": nn.MSELoss()},
        batch_size=params["train_params"]["batch_size"],
        n_epochs=params["train_params"]["n_epochs"],
        init_lr=train_params_search["init_lr"],
        device=params["train_params"]["device"],
        log_dir=str(log_dir),
        logger=logger,
        callbacks=list(callbacks.values()),
        checkpoint_cb=callbacks["checkpoint"],  # type: ignore
        target_dims=params["train_params"].get("target_dims"),
        log_every_n_steps=params["train_params"]["log_every_n_steps"],
        weight_decay=train_params_search["weight_decay"],
        eps=train_params_search["eps"],
        betas=train_params_search["betas"],
    )

    # Train the model
    trainer.fit(train_loader, val_loader)

    # Load the best model
    model = GRU_PLModule.load_from_checkpoint(
        checkpoint_path=os.path.join(logger.log_dir, "best.ckpt"),
        model=model,
        model_params=model_params,
        **params["train_params"],
    )

    train_metrics = get_experiment_metrics(
        trainer=trainer,
        val_loader=inference_train_loader,
        X_val=X_train,
        model_params=model_params,
        y_val=y_train,
        params=params,
        dataset=dataset,
        # epsilon=predictor_params["epsilon"],
    )
    print_all_metrics(train_metrics, "------- Train -------")

    # save
    json.dump(
        train_metrics,
        open(
            os.path.join(
                logger.log_dir,
                "metrics.json",
            ),
            "w",
        ),
    )
    # Log metrics to tensorboard
    for metric_name, metric_value in train_metrics.items():
        if isinstance(metric_value, (int, float)):
            trainer.logger.experiment.add_scalar(
                f"val_metrics/{metric_name}", metric_value, trial.number
            )
    # Deallocate memory
    del model
    del trainer
    torch.cuda.empty_cache()

    end_time = time()
    print(f"Trial {trial.number} completed in {end_time - start_time:.2f} seconds")

    return train_metrics["vus_roc_mean"]


def main(params_file: str, n_trials: int, study_name: str):
    torch.set_float32_matmul_precision("medium")
    # Load parameters
    params = load_params(params_file, file_type=ParamFileTypes.YAML)

    # Setup logging
    log_dir = Path(params["train_params"]["log_dir"]) / study_name
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create new study if none exists
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )

    # Setup dataset
    dataset = cast_dataset(params["dataset"])
    dataset_config = get_dataset_config(dataset=dataset)

    # Load data
    X_train, X_val, _, y_train, y_val, _ = load_training_data(
        dataset=dataset,
        test_size=params["train_params"]["test_size"],
        val_size=params["train_params"]["val_size"],
        normalize=dataset_config.normalize,
        clean=params["train_params"]["clean"] == CleanMethods.INTERPOLATE.value,
        interpolate_method=params["train_params"]["interpolate_method"],
    )
    # Run optimization
    study.optimize(
        lambda trial: objective(
            trial=trial,
            params=params,
            X_train=X_train,
            X_val=X_val,
            y_train=y_train,
            y_val=y_val,
            log_dir=log_dir,
            dataset=dataset,
        ),
        n_trials=n_trials,
    )
    # Save results
    best_params = study.best_params
    best_value = study.best_value

    # Save the best parameters
    output_file = log_dir / "best_params.yaml"
    with open(output_file, "w") as f:
        yaml.dump({"best_params": best_params, "best_value": best_value}, f)

    print(f"Best parameters: {best_params}")
    print(f"Best value: {best_value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--params-file", type=str, default="models/gru/params.yaml")
    parser.add_argument("--n-trials", type=int, default=500, help="Number of trials")
    parser.add_argument("--study-name", type=str)
    args = parser.parse_args()

    main(args.params_file, args.n_trials, args.study_name)
