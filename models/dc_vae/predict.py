import argparse
import json
import os

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from datasets.config import get_dataset_config
from datasets.dataset import SlidingWindowDataset
from gragod import InterPolationMethods, ParamFileTypes
from gragod.metrics.calculator import get_metrics
from gragod.metrics.visualization import print_all_metrics
from gragod.predictions.prediction import (
    get_threshold,
    smooth_scores,
    standarize_error_scores,
)
from gragod.training import load_params, load_training_data, set_seeds
from gragod.types import CleanMethods, cast_dataset
from models.dc_vae.model import DCVAE, DCVAE_PLModule

RANDOM_SEED = 42
EPSILON = 0.8

set_seeds(RANDOM_SEED)


def run_model(
    model: DCVAE_PLModule,
    loader: DataLoader,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate predictions using the VAE model.
    Returns the reconstructed means and log variances.
    """
    trainer = pl.Trainer(accelerator=device)
    output = trainer.predict(model, loader)
    x_means, x_log_vars = zip(*output)

    # Get only the last prediction for each sequence
    x_means = torch.cat([x[:, -1:, :] for x in x_means]).squeeze()
    x_log_vars = torch.cat([x[:, -1:, :] for x in x_log_vars]).squeeze()

    return x_means, x_log_vars


def generate_scores(
    x_means: torch.Tensor,
    x_log_vars: torch.Tensor,
    data: torch.Tensor,
    window_size: int,
) -> torch.Tensor:
    """
    Calculate anomaly scores based on the normal-operation region defined by μ(t)
    and σ(t).
    For each time-series m, an anomaly is detected at time t if:
    |x_m(t) - μ_m(t)| > α_m × σ_m(t)

    Args:
        x_means: Predicted means (μ) from VAE
        x_log_vars: Predicted log variances (log σ²) from VAE
        data: Original input data
        window_size: Size of sliding window

    Returns:
        Anomaly scores for each time point and feature
    """
    # Get the values we're trying to predict (after window_size)
    true_values = data[window_size:]

    # Calculate |x_m(t) - μ_m(t)|
    absolute_deviation = torch.abs(true_values - x_means)

    # Calculate σ_m(t) from log variance
    std_dev = torch.exp(0.5 * x_log_vars)  # Convert log variance to standard deviation

    # Calculate normalized deviation score: |x_m(t) - μ_m(t)| / σ_m(t)
    # This score represents how many standard deviations away from the mean each
    # point is
    scores = absolute_deviation / std_dev

    return scores


def get_mae(
    x_means: torch.Tensor,
    data: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    Calculate the mean absolute error between the predicted means and the original data.
    """
    # Get the indices of the labels that are 0
    normal_indices = labels == 0
    return torch.mean(torch.abs(data[normal_indices] - x_means[normal_indices]))


def get_variance_scores(data: torch.Tensor, x_means: torch.Tensor) -> torch.Tensor:
    """
    Calculate the variance scores (explained variance) based on the formula:
    Var_score = 1 - Var(x(t) - μ_x(t)) / Var(x(t))

    Args:
        data: Original input data x(t)
        x_means: Predicted means μ_x(t)

    Returns:
        Variance scores between [0,1] where 1 represents perfect reconstruction
    """
    # Calculate reconstruction error: x(t) - μ_x(t)
    reconstruction_error = data - x_means

    # Calculate variance of reconstruction error
    error_variance = torch.var(reconstruction_error, dim=0)

    # Calculate variance of input data
    data_variance = torch.var(data, dim=0)

    # Calculate variance score: 1 - Var(error) / Var(data)
    var_scores = 1 - (error_variance / data_variance)

    return torch.mean(var_scores, dim=0)


def main(
    dataset_name: str,
    model_params: dict,
    batch_size: int = 264,
    ckpt_path: str | None = None,
    device: str = "mps",
    n_workers: int = 0,
    target_dims: int | None = None,
    save_dir: str = "output",
    test_size: float = 0.1,
    val_size: float = 0.1,
    clean: str = "interpolate",
    interpolate_method: InterPolationMethods | None = None,
    params: dict = {},
    **kwargs,
) -> dict:
    """
    Main function to load data, model and generate predictions.
    """

    checkpoint_path = (
        os.path.join(params["predictor_params"]["ckpt_folder"], "dcvae.ckpt")
        if ckpt_path is None
        else ckpt_path
    )

    if not os.path.exists(checkpoint_path):
        raise ValueError(f"Checkpoint not found at {checkpoint_path}")

    dataset = cast_dataset(dataset_name)
    dataset_config = get_dataset_config(dataset=dataset)

    # Load data
    (
        X_train,
        X_val,
        X_test,
        X_train_labels,
        X_val_labels,
        X_test_labels,
    ) = load_training_data(
        dataset=dataset,
        test_size=test_size,
        val_size=val_size,
        normalize=dataset_config.normalize,
        clean=clean == CleanMethods.INTERPOLATE,
        interpolate_method=interpolate_method,
    )

    window_size = model_params["window_size"]
    X_train_labels = X_train_labels[window_size:]
    X_val_labels = X_val_labels[window_size:]
    X_test_labels = X_test_labels[window_size:]

    # Create dataloaders
    train_dataset = SlidingWindowDataset(
        X_train,
        window_size,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=n_workers,
    )
    val_dataset = SlidingWindowDataset(X_val, window_size)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=n_workers,
    )
    test_dataset = SlidingWindowDataset(X_test, window_size)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=n_workers,
    )

    # Create and load model
    n_features = X_train.shape[1]
    model = DCVAE(
        input_dim=n_features,
        hidden_dims=model_params["hidden_dims"],
        kernel_size=model_params["kernel_size"],
        dilations=model_params["dilations"],
        latent_dim=model_params["latent_dim"],
    )

    print(f"Loading model from checkpoint: {checkpoint_path}")
    lightning_module = DCVAE_PLModule.load_from_checkpoint(
        checkpoint_path,
        model=model,
        model_params=model_params,
        **params["train_params"],
    )

    model = lightning_module.model.to("cuda" if device == "gpu" else "mps")
    model.eval()

    # Generate predictions and calculate metrics
    x_means_train, x_log_vars_train = run_model(
        model=lightning_module,
        loader=train_loader,
        device=device,
    )
    x_means_val, x_log_vars_val = run_model(
        model=lightning_module,
        loader=val_loader,
        device=device,
    )
    x_means_test, x_log_vars_test = run_model(
        model=lightning_module,
        loader=test_loader,
        device=device,
    )
    print("Obtained predictions")
    train_scores = generate_scores(
        x_means=x_means_train,
        x_log_vars=x_log_vars_train,
        data=X_train,
        window_size=window_size,
    )
    train_scores = standarize_error_scores(train_scores)
    train_scores = smooth_scores(train_scores, 5)
    print("Generating scores for val")
    val_scores = generate_scores(
        x_means=x_means_val,
        x_log_vars=x_log_vars_val,
        data=X_val,
        window_size=window_size,
    )
    val_scores = standarize_error_scores(val_scores)
    val_scores = smooth_scores(val_scores, 5)
    print("Scores for val obtained")
    test_scores = generate_scores(
        x_means=x_means_test,
        x_log_vars=x_log_vars_test,
        data=X_test,
        window_size=window_size,
    )
    test_scores = standarize_error_scores(test_scores)
    test_scores = smooth_scores(test_scores, 5)
    print("Generating thresholds")
    thresholds = get_threshold(
        dataset=dataset,
        scores=train_scores,
        labels=X_train_labels,
        n_thresholds=params["predictor_params"]["n_thresholds"],
        range_based=True,
    )
    print("Thresholds obtained")
    X_train_pred = (train_scores > thresholds).float()
    metrics_train = get_metrics(
        dataset=dataset,
        predictions=X_train_pred,
        labels=X_train_labels,
        scores=train_scores,
    )
    print_all_metrics(metrics_train, "------- Train -------")
    print("Generating predictions for val")
    X_val_pred = (val_scores > thresholds).float()
    print("Predictions for val obtained")
    print("Calculating metrics for val")
    metrics_val = get_metrics(
        dataset=dataset,
        predictions=X_val_pred,
        labels=X_val_labels,
        scores=val_scores,
    )
    print_all_metrics(metrics_val, "------- Val -------")

    X_test_pred = (test_scores > thresholds).float()
    metrics_test = get_metrics(
        dataset=dataset,
        predictions=X_test_pred,
        labels=X_test_labels,
        scores=test_scores,
    )
    print_all_metrics(metrics_test, "------- Test -------")

    # Save metrics
    json.dump(
        metrics_train,
        open(
            os.path.join(
                params["predictor_params"]["ckpt_folder"], "metrics_train.json"
            ),
            "w",
        ),
    )

    # Calculate MAE
    mae_train = get_mae(
        x_means=x_means_train,
        data=X_train[window_size:, :],
        labels=X_train_labels,
    )
    print(f"MAE for train: {mae_train}")
    mae_val = get_mae(
        x_means=x_means_val,
        data=X_val[window_size:, :],
        labels=X_val_labels,
    )
    print(f"MAE for val: {mae_val}")
    mae_test = get_mae(
        x_means=x_means_test,
        data=X_test[window_size:, :],
        labels=X_test_labels,
    )
    print(f"MAE for test: {mae_test}")

    # Calculate variance scores
    var_scores_train = get_variance_scores(
        data=X_train[window_size:, :],
        x_means=x_means_train,
    )
    print(f"Variance scores for train: {var_scores_train}")
    var_scores_val = get_variance_scores(
        data=X_val[window_size:, :],
        x_means=x_means_val,
    )
    print(f"Variance scores for val: {var_scores_val}")
    var_scores_test = get_variance_scores(
        data=X_test[window_size:, :],
        x_means=x_means_test,
    )
    print(f"Variance scores for test: {var_scores_test}")
    json.dump(
        metrics_val,
        open(
            os.path.join(params["predictor_params"]["ckpt_folder"], "metrics_val.json"),
            "w",
        ),
    )
    return {
        "predictions": X_train_pred,
        "labels": X_train_labels,
        "scores": train_scores,
        "data": X_train[:, window_size:],
        "thresholds": thresholds,
        "x_means": x_means_train,
        "x_log_vars": x_log_vars_train,
        "metrics": metrics_train,
        "mae": mae_train,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--params_file", type=str, default="models/dc_vae/params.yaml")
    parser.add_argument("--ckpt_path", type=str, default=None)
    args = parser.parse_args()
    params = load_params(args.params_file, file_type=ParamFileTypes.YAML)

    main(
        dataset_name=params["dataset"],
        ckpt_path=args.ckpt_path,
        **params["train_params"],
        model_params=params["model_params"],
        params=params,
    )
