import argparse
import json
import os

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from datasets.config import get_dataset_config
from datasets.dataset import SlidingWindowDataset
from gragod import InterPolationMethods, ParamFileTypes
from gragod.metrics import (
    generate_metrics_per_class_table,
    generate_metrics_table,
    get_metrics,
)
from gragod.training import load_params, load_training_data, set_seeds
from gragod.types import cast_dataset
from models.mtad_gat.model import MTAD_GAT, MTAD_GAT_PLModule
from models.mtad_gat.spot import SPOT

RANDOM_SEED = 42
EPSILON = 0.8

set_seeds(RANDOM_SEED)


def run_model(
    model: MTAD_GAT_PLModule,
    loader: DataLoader,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate predictions and calculate anomaly scores.
    Returns the anomaly predictions and evaluation metrics.
    """
    trainer = pl.Trainer(accelerator=device)
    output = trainer.predict(model, loader)
    forecasts, reconstructions = zip(*output)
    forecasts = torch.cat(forecasts)
    reconstructions = torch.cat(reconstructions)[:, -1, :]

    return forecasts, reconstructions


def generate_scores(
    forecasts: torch.Tensor,
    reconstructions: torch.Tensor,
    data: torch.Tensor,
    window_size: int,
    EPSILON: float,
) -> torch.Tensor:
    true_values = data[window_size:]

    score = torch.sqrt((forecasts - true_values) ** 2) + EPSILON * torch.sqrt(
        (reconstructions - true_values) ** 2
    )
    score = score / (1 + EPSILON)
    return score


def get_predictions(
    train_score: torch.Tensor, test_score: torch.Tensor
) -> torch.Tensor:
    """
    Get threshold for anomaly detection.
    """
    thresholds = []
    for i in range(train_score.shape[1]):
        s = SPOT(q=1e-3)
        s.fit(train_score[:, i].numpy(), test_score[:, i].numpy())
        s.initialize(level=0.95)
        ret = s.run(dynamic=False, with_alarm=False)
        threshold = torch.Tensor(ret["thresholds"]).mean()
        thresholds.append(threshold)
    thresholds = torch.stack(thresholds)
    predictions = test_score > thresholds
    predictions = predictions.int()
    return predictions


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
    clean: bool = True,
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

    window_size = model_params["window_size"]
    X_train_labels = X_train_labels[window_size:]
    X_val_labels = X_val_labels[window_size:]
    X_test_labels = X_test_labels[window_size:]

    # Create test dataloader
    train_dataset = SlidingWindowDataset(X_train, window_size)
    train_loader = DataLoader(
        train_dataset,
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
    model = MTAD_GAT(
        n_features=n_features,
        out_dim=n_features,
        **model_params,
    )

    checkpoint_path = (
        os.path.join(params["predictor_params"]["ckpt_folder"], "mtad_gat.ckpt")
        if ckpt_path is None
        else ckpt_path
    )

    if not os.path.exists(checkpoint_path):
        raise ValueError(f"Checkpoint not found at {checkpoint_path}")

    print(f"Loading model from checkpoint: {checkpoint_path}")
    lightning_module = MTAD_GAT_PLModule.load_from_checkpoint(
        checkpoint_path,
        model=model,
        model_params=model_params,
        **params["train_params"],
    )

    model = lightning_module.model.to(device)
    model.eval()

    # Generate predictions and calculate metrics
    forecasts_train, reconstructions_train = run_model(
        model=lightning_module,
        loader=train_loader,
        device=device,
    )
    forecasts_test, reconstructions_test = run_model(
        model=lightning_module,
        loader=test_loader,
        device=device,
    )

    train_scores = generate_scores(
        forecasts=forecasts_train,
        reconstructions=reconstructions_train,
        data=X_train,
        window_size=window_size,
        EPSILON=EPSILON,
    )
    test_scores = generate_scores(
        forecasts=forecasts_test,
        reconstructions=reconstructions_test,
        data=X_test,
        window_size=window_size,
        EPSILON=EPSILON,
    )
    X_test_pred = get_predictions(train_scores, test_scores)

    metrics = get_metrics(X_test_pred, X_test_labels)
    metrics_table = generate_metrics_table(metrics)
    metrics_per_class_table = generate_metrics_per_class_table(metrics)
    print(metrics_table)
    print(metrics_per_class_table)

    # save
    json.dump(
        metrics,
        open(
            os.path.join(params["predictor_params"]["ckpt_folder"], "metrics.json"), "w"
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--params_file", type=str, default="models/mtad_gat/params.yaml"
    )
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
