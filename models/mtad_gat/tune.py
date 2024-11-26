import argparse
import os
from pathlib import Path

import optuna
import torch
import yaml
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn
from torch.utils.data import DataLoader

from datasets.config import get_dataset_config
from datasets.dataset import SlidingWindowDataset
from gragod import CleanMethods, ParamFileTypes
from gragod.metrics import get_metrics, print_all_metrics
from gragod.training import load_params, load_training_data, set_seeds
from gragod.training.trainer import TrainerPL
from gragod.types import cast_dataset
from models.mtad_gat.model import MTAD_GAT, MTAD_GAT_PLModule
from models.mtad_gat.predict import EPSILON, generate_scores, get_predictions

RANDOM_SEED = 42
set_seeds(RANDOM_SEED)


def get_experiment_metrics(
    trainer, val_loader, train_loader, X_train, X_val, y_val, model_params
):
    # Get metrics
    output_val = trainer.predict(val_loader)
    forecasts, reconstructions = zip(*output_val)
    forecasts = torch.cat(forecasts)
    reconstructions = torch.cat(reconstructions)[:, -1, :]
    val_scores = generate_scores(
        forecasts=forecasts,
        reconstructions=reconstructions,
        data=X_val,
        window_size=model_params["window_size"],
        EPSILON=EPSILON,
    )

    output_train = trainer.predict(train_loader)
    forecasts, reconstructions = zip(*output_train)
    forecasts = torch.cat(forecasts)
    reconstructions = torch.cat(reconstructions)[:, -1, :]
    train_scores = generate_scores(
        forecasts=forecasts,
        reconstructions=reconstructions,
        data=X_train,
        window_size=model_params["window_size"],
        EPSILON=EPSILON,
    )
    val_pred, val_thresholds = get_predictions(
        train_score=train_scores, test_score=val_scores
    )
    val_metrics = get_metrics(
        predictions=val_pred,
        labels=y_val[model_params["window_size"] :],
        scores=val_scores,
    )
    return val_metrics


def create_datasets(X, y, window_size, horizon, batch_size, shuffle=True):
    """Create dataset and dataloader with specific window size."""
    dataset = SlidingWindowDataset(
        X,
        window_size=window_size,
        horizon=horizon,
        labels=y,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )

    return dataloader


def objective(trial, params, X_train, X_val, y_train, y_val, log_dir):
    """Optuna objective function."""
    # Get trial hyperparameters
    print(f"Trial number: {trial.number}")
    model_params = {
        "window_size": trial.suggest_int("window_size", 10, 20),
        "kernel_size": trial.suggest_int("kernel_size", 3, 11, step=2),
        "use_gatv2": trial.suggest_categorical("use_gatv2", [True, False]),
        "feat_gat_embed_dim": trial.suggest_int(
            "feat_gat_embed_dim", 100, 300, step=100
        ),
        "time_gat_embed_dim": trial.suggest_int(
            "time_gat_embed_dim", 100, 300, step=100
        ),
        "gru_n_layers": trial.suggest_int("gru_n_layers", 1, 3),
        "gru_hid_dim": trial.suggest_int("gru_hid_dim", 100, 900, step=100),
        "forecast_n_layers": trial.suggest_int("forecast_n_layers", 1, 6),
        "forecast_hid_dim": trial.suggest_int("forecast_hid_dim", 100, 900, step=100),
        "recon_n_layers": trial.suggest_int("recon_n_layers", 1, 5),
        "recon_hid_dim": trial.suggest_int("recon_hid_dim", 100, 900, step=100),
        "dropout": trial.suggest_float("dropout", 0.1, 0.5),
        "alpha": trial.suggest_float("alpha", 0.1, 0.3),
    }

    train_params_search = {
        "init_lr": trial.suggest_float("init_lr", 1e-5, 1e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
        "eps": trial.suggest_float("eps", 1e-5, 1e-3, log=True),
        "betas": (
            trial.suggest_float("betas", 0.9, 0.999),
            trial.suggest_float("betas", 0.9, 0.999),
        ),
    }
    # Create dataloaders with the trial's window size
    train_loader = create_datasets(
        X_train,
        y_train,
        window_size=model_params["window_size"],
        horizon=params["train_params"]["horizon"],
        batch_size=params["train_params"]["batch_size"],
        shuffle=params["train_params"]["shuffle"],
    )

    val_loader = create_datasets(
        X_val,
        y_val,
        window_size=model_params["window_size"],
        horizon=params["train_params"]["horizon"],
        batch_size=params["train_params"]["batch_size"],
        shuffle=False,
    )

    # Create model
    model = MTAD_GAT(
        n_features=X_train.shape[1],
        out_dim=X_train.shape[1],
        **model_params,
    )

    # Define callbacks and logger
    logger = TensorBoardLogger(
        save_dir=log_dir, name=f"mtad_gat_trial_{trial.number}", default_hp_metric=False
    )

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
        filename="best",
        save_top_k=1,
        mode="min",
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [early_stop, checkpoint, lr_monitor]

    # Create trainer
    trainer = TrainerPL(
        model=model,
        model_pl=MTAD_GAT_PLModule,
        model_params=model_params,
        criterion={"forecast_criterion": nn.MSELoss(), "recon_criterion": nn.MSELoss()},
        batch_size=params["train_params"]["batch_size"],
        n_epochs=params["train_params"]["n_epochs"],
        init_lr=train_params_search["init_lr"],
        device=params["train_params"]["device"],
        log_dir=log_dir,
        logger=logger,
        callbacks=callbacks,
        checkpoint_cb=checkpoint,
        target_dims=params["train_params"].get("target_dims"),
        log_every_n_steps=params["train_params"]["log_every_n_steps"],
        weight_decay=train_params_search["weight_decay"],
        eps=train_params_search["eps"],
        betas=train_params_search["betas"],
    )

    # Train the model
    trainer.fit(train_loader, val_loader)

    # Load the best model
    model = MTAD_GAT_PLModule.load_from_checkpoint(
        checkpoint_path=os.path.join(logger.log_dir, "best.ckpt"),
        model=model,
        model_params=model_params,
        **params["train_params"],
    )

    val_metrics = get_experiment_metrics(
        trainer=trainer,
        val_loader=val_loader,
        train_loader=train_loader,
        X_val=X_val,
        model_params=model_params,
        X_train=X_train,
        y_val=y_val,
    )
    print_all_metrics(val_metrics, "------- Val -------")

    return val_metrics["vus_roc_mean"]


def main(params_file: str, n_trials: int):
    # Load parameters
    params = load_params(params_file, file_type=ParamFileTypes.YAML)

    # Setup logging
    log_dir = Path(params["train_params"]["log_dir"]) / "mtad_gat_optuna_generic_search"
    log_dir.mkdir(parents=True, exist_ok=True)

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

    # Create study
    study = optuna.create_study(
        direction="maximize",
        study_name="mtad_gat_optimization",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )

    # Run optimization
    study.optimize(
        lambda trial: objective(trial, params, X_train, X_val, y_train, y_val, log_dir),
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
    parser.add_argument(
        "--params_file", type=str, default="models/mtad_gat/params.yaml"
    )
    parser.add_argument(
        "--n_trials", type=int, default=2, help="Number of optimization trials"
    )

    args = parser.parse_args()
    main(args.params_file, args.n_trials)
