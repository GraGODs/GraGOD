import argparse
import json
import os

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from datasets.config import get_dataset_config
from datasets.dataset import SlidingWindowDataset
from gragod import CleanMethods, InterPolationMethods, ParamFileTypes
from gragod.metrics import get_metrics, print_all_metrics
from gragod.predictions.prediction import get_threshold
from gragod.training import load_params, load_training_data
from gragod.types import cast_dataset
from models.gcn.model import GCN, GCN_PLModule


def run_model(
    model: GCN_PLModule,
    loader: DataLoader,
    device: str,
    edge_index: torch.Tensor,
) -> torch.Tensor:
    """
    Generate predictions and calculate anomaly scores.
    Returns the anomaly predictions and evaluation metrics.
    """
    trainer = pl.Trainer(accelerator=device)
    output = trainer.predict(model, loader, return_predictions=True)
    if output is None:
        raise ValueError("Model predictions returned None")

    forecasts = torch.cat([torch.tensor(x) for x in output])

    return forecasts


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
    clean: CleanMethods = CleanMethods.NONE,
    interpolate_method: InterPolationMethods | None = None,
    params: dict = {},
    **kwargs,
):
    """
    Main function to load data, model and generate predictions.
    Returns a dictionary containing evaluation metrics.
    """
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
        clean=False,
        interpolate_method=interpolate_method,
    )

    print(f"Initial data shapes: {X_train.shape}, {X_test.shape}")

    window_size = model_params["window_size"]
    X_train_labels = X_train_labels[window_size:]
    X_val_labels = X_val_labels[window_size:]
    X_test_labels = X_test_labels[window_size:]

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

    # Create test dataloader
    train_dataset = SlidingWindowDataset(X_train, window_size, edge_index=edge_index)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=n_workers,
    )

    val_dataset = SlidingWindowDataset(X_val, window_size, edge_index=edge_index)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=n_workers,
    )

    test_dataset = SlidingWindowDataset(X_test, window_size, edge_index=edge_index)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=n_workers,
    )

    # Create and load model

    model = GCN(
        window_size=model_params["window_size"],
        n_layers=model_params["n_layers"],
        hidden_dim=model_params["hidden_dim"],
        K=model_params["K"],
    ).to(device)

    checkpoint_path = (
        os.path.join(params["predictor_params"]["ckpt_folder"], "gcn.ckpt")
        if ckpt_path is None
        else ckpt_path
    )

    if not os.path.exists(checkpoint_path):
        raise ValueError(f"Checkpoint not found at {checkpoint_path}")

    print(f"Loading model from checkpoint: {checkpoint_path}")
    lightning_module = GCN_PLModule.load_from_checkpoint(
        checkpoint_path,
        model=model,
        model_params=model_params,
        **params["train_params"],
    )

    model = lightning_module.model.to(device)
    model.eval()

    # Generate predictions and calculate metrics
    forecasts_val = run_model(
        model=lightning_module,
        loader=val_loader,
        device=device,
        edge_index=edge_index,
    )
    forecasts_test = run_model(
        model=lightning_module,
        loader=test_loader,
        device=device,
        edge_index=edge_index,
    )

    val_scores = torch.abs(forecasts_val - X_val[window_size:])
    test_scores = torch.abs(forecasts_test - X_test[window_size:])

    threshold = get_threshold(
        val_scores, X_val_labels, params["predictor_params"]["n_thresholds"]
    )
    val_pred = (val_scores > threshold).float()
    test_pred = (test_scores > threshold).float()

    # we only calculate train if there's at least one anomaly
    if torch.any(X_train_labels == 1):
        forecasts_train = run_model(
            model=lightning_module,
            loader=train_loader,
            device=device,
            edge_index=edge_index,
        )
        train_scores = torch.abs(forecasts_train - X_train[window_size:])
        train_pred = (train_scores > threshold).float()
        train_metrics = get_metrics(train_pred, X_train_labels, train_scores)
        print_all_metrics(train_metrics, "------- Train -------")
        json.dump(
            train_metrics,
            open(
                os.path.join(
                    params["predictor_params"]["ckpt_folder"], "train_metrics.json"
                ),
                "w",
            ),
        )

    val_metrics = get_metrics(val_pred, X_val_labels, val_scores)
    test_metrics = get_metrics(test_pred, X_test_labels, test_scores)
    print_all_metrics(val_metrics, "------- Validation -------")
    print_all_metrics(test_metrics, "------- Test -------")

    json.dump(
        val_metrics,
        open(
            os.path.join(params["predictor_params"]["ckpt_folder"], "val_metrics.json"),
            "w",
        ),
    )

    json.dump(
        test_metrics,
        open(
            os.path.join(
                params["predictor_params"]["ckpt_folder"], "test_metrics.json"
            ),
            "w",
        ),
    )

    return {
        "predictions": test_pred,
        "labels": X_test_labels,
        "scores": test_scores,
        "data": X_test,
        "thresholds": threshold,
        "forecasts": forecasts_test,
        "metrics": test_metrics,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--params_file", type=str, default="models/gcn/params.yaml")
    args = parser.parse_args()
    params = load_params(args.params_file, file_type=ParamFileTypes.YAML)

    main(
        dataset_name=params["dataset"],
        **params["train_params"],
        model_params=params["model_params"],
        params=params,
    )
