import argparse
import json
import os
from pathlib import Path
from typing import Literal, TypedDict

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import _PREDICT_OUTPUT
from torch.utils.data import DataLoader

from datasets.config import get_dataset_config
from datasets.dataset import get_data_loader, get_edge_index
from gragod import CleanMethods, Datasets, Models, ParamFileTypes
from gragod.metrics.calculator import get_metrics
from gragod.metrics.visualization import print_all_metrics
from gragod.models import get_model_and_module
from gragod.predictions.prediction import get_threshold, post_process_scores
from gragod.training import load_params, load_training_data, set_seeds
from gragod.utils import load_checkpoint_path


class DatasetPredictOutput(TypedDict):
    output: _PREDICT_OUTPUT
    predictions: torch.Tensor | None
    labels: torch.Tensor
    scores: torch.Tensor
    data: torch.Tensor
    thresholds: torch.Tensor
    metrics: dict | None


class PredictOutput(TypedDict):
    train: DatasetPredictOutput
    val: DatasetPredictOutput
    test: DatasetPredictOutput


RANDOM_SEED = 42


def run_model(
    model: pl.LightningModule,
    loader: DataLoader,
    device: str,
    X_true: torch.Tensor,
    post_process: bool = True,
    window_size_smooth: int = 5,
    **kwargs,
) -> tuple[torch.Tensor, _PREDICT_OUTPUT]:
    """
    Generate predictions and calculate anomaly scores.
    Returns the anomaly predictions and evaluation metrics.
    """
    trainer = pl.Trainer(accelerator=device)
    output = trainer.predict(model, loader)
    if output is None:
        raise ValueError("Model predictions returned None")

    scores = model.calculate_anomaly_score(
        predict_output=output, X_true=X_true, **kwargs
    )

    output = model.postprocess_predict(output)

    if post_process:
        print(f"Post processing scores with window size {window_size_smooth}")
        scores = post_process_scores(scores, window_size=window_size_smooth)

    return scores, output


def calculate_metrics(
    scores: torch.Tensor,
    threshold: torch.Tensor,
    y: torch.Tensor,
    dataset: Datasets,
    dataset_split: Literal["train", "val", "test"],
    save_dir: Path,
):
    y_pred = (scores > threshold).float()
    metrics = get_metrics(
        dataset=dataset,
        predictions=y_pred,
        labels=y,
        scores=scores,
    )
    print_all_metrics(metrics, f"------- {dataset_split} -------")
    json.dump(
        metrics,
        open(
            os.path.join(
                save_dir,
                f"{dataset_split}_metrics.json",
            ),
            "w",
        ),
    )
    return metrics, y_pred


def predict(
    model: Models,
    dataset: Datasets,
    model_params: dict,
    batch_size: int = 264,
    ckpt_path: str | None = None,
    device: str = "mps",
    n_workers: int = 0,
    test_size: float = 0.1,
    val_size: float = 0.1,
    params: dict = {},
    down_len: int | None = None,
    **kwargs,
) -> PredictOutput:
    """
    Main function to load data, model and generate predictions.
    Returns a dictionary containing evaluation metrics.
    """
    dataset_config = get_dataset_config(dataset=dataset)

    # Load data
    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = load_training_data(
        dataset=dataset,
        test_size=test_size,
        val_size=val_size,
        normalize=dataset_config.normalize,
        clean=False,
        down_len=down_len,
    )

    print(
        f"Initial data shapes: "
        f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}"
    )

    edge_index = get_edge_index(X_train, device)

    window_size = model_params["window_size"]

    # Create test dataloader
    train_loader = get_data_loader(
        X=X_train,
        edge_index=edge_index,
        y=y_train,
        window_size=model_params["window_size"],
        clean=CleanMethods.NONE,
        batch_size=batch_size,
        n_workers=n_workers,
        shuffle=False,
    )

    val_loader = get_data_loader(
        X=X_val,
        edge_index=edge_index,
        y=y_val,
        window_size=model_params["window_size"],
        clean=CleanMethods.NONE,
        batch_size=batch_size,
        n_workers=n_workers,
        shuffle=False,
    )
    test_loader = get_data_loader(
        X=X_test,
        edge_index=edge_index,
        y=y_test,
        window_size=window_size,
        clean=CleanMethods.NONE,
        batch_size=batch_size,
        n_workers=n_workers,
        shuffle=False,
    )
    X_train = X_train[window_size:]
    X_val = X_val[window_size:]
    X_test = X_test[window_size:]
    y_train = y_train[window_size:]
    y_val = y_val[window_size:]
    y_test = y_test[window_size:]

    # Create and load model
    _, model_pl_module = get_model_and_module(model)

    model_params["edge_index"] = [edge_index]
    model_params["n_features"] = X_train.shape[1]
    model_params["out_dim"] = X_train.shape[1]

    checkpoint_path = (
        load_checkpoint_path(
            checkpoint_folder=params["predictor_params"]["ckpt_folder"],
            experiment_name=params["train_params"]["model_name"],
        )
        if ckpt_path is None
        else Path(ckpt_path)
    )

    print(f"Loading model from checkpoint: {checkpoint_path}")
    lightning_module = model_pl_module.load_from_checkpoint(
        checkpoint_path,
        map_location=device,
    )

    lightning_module.eval()

    # Generate predictions and calculate metrics
    train_scores, train_output = run_model(
        model=lightning_module,
        loader=train_loader,
        device=device,
        X_true=X_train,
        post_process=params["predictor_params"]["post_process_scores"],
        window_size_smooth=params["predictor_params"]["window_size_smooth"],
    )
    val_scores, val_output = run_model(
        model=lightning_module,
        loader=val_loader,
        device=device,
        X_true=X_val,
        post_process=params["predictor_params"]["post_process_scores"],
        window_size_smooth=params["predictor_params"]["window_size_smooth"],
    )
    test_scores, test_output = run_model(
        model=lightning_module,
        loader=test_loader,
        device=device,
        X_true=X_test,
        post_process=params["predictor_params"]["post_process_scores"],
        window_size_smooth=params["predictor_params"]["window_size_smooth"],
    )

    # If there's no anomalies in the train set, use the val set instead
    if (
        not torch.any(y_train == 1)
        and params["predictor_params"]["dataset_for_threshold"] == "train"
    ):
        print(
            "No anomalies in train set, cannot calculate threshold. "
            "Using val set instead."
        )
        params["predictor_params"]["dataset_for_threshold"] = "val"

    threshold = get_threshold(
        dataset=dataset,
        scores=(
            val_scores
            if params["predictor_params"]["dataset_for_threshold"] == "val"
            else train_scores  # type: ignore
        ),
        labels=(
            y_val
            if params["predictor_params"]["dataset_for_threshold"] == "val"
            else y_train
        ),
        n_thresholds=params["predictor_params"]["n_thresholds"],
    )

    if torch.any(y_train == 1):
        train_metrics, y_train_pred = calculate_metrics(
            scores=train_scores,
            threshold=threshold,
            y=y_train,
            dataset=dataset,
            dataset_split="train",
            save_dir=checkpoint_path.parent,
        )
    else:
        train_metrics = None
        y_train_pred = None

    val_metrics, y_val_pred = calculate_metrics(
        scores=val_scores,
        threshold=threshold,
        y=y_val,
        dataset=dataset,
        dataset_split="val",
        save_dir=checkpoint_path.parent,
    )
    test_metrics, y_test_pred = calculate_metrics(
        scores=test_scores,
        threshold=threshold,
        y=y_test,
        dataset=dataset,
        dataset_split="test",
        save_dir=checkpoint_path.parent,
    )

    train_output_dict: DatasetPredictOutput = {
        "output": train_output,
        "predictions": y_train_pred,
        "labels": y_train,
        "scores": train_scores,
        "data": X_train,
        "thresholds": threshold,
        "metrics": train_metrics,
    }
    val_output_dict: DatasetPredictOutput = {
        "output": val_output,
        "predictions": y_val_pred,
        "labels": y_val,
        "scores": val_scores,
        "data": X_val,
        "thresholds": threshold,
        "metrics": val_metrics,
    }
    test_output_dict: DatasetPredictOutput = {
        "output": test_output,
        "predictions": y_test_pred,
        "labels": y_test,
        "scores": test_scores,
        "data": X_test,
        "thresholds": threshold,
        "metrics": test_metrics,
    }

    return_dict: PredictOutput = {
        "train": train_output_dict,
        "val": val_output_dict,
        "test": test_output_dict,
    }

    return return_dict


def main(
    model: Models,
    dataset: Datasets,
    ckpt_path: str | None = None,
    params_file: str = "models/mtad_gat/params.yaml",
) -> PredictOutput:
    """
    Main function to load data, model and generate predictions.

    Args:
        model: Name of model to predict
        params_file: Path to parameter file
    """
    params = load_params(params_file, file_type=ParamFileTypes.YAML)
    set_seeds(RANDOM_SEED)

    return predict(
        model=model,
        dataset=dataset,
        **params["train_params"],
        model_params=params["model_params"],
        params=params,
        ckpt_path=ckpt_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Models,
        help="Model to train (gru, gcn, gdn, mtad_gat)",
    )
    parser.add_argument(
        "--dataset",
        type=Datasets,
        help="Dataset to predict",
    )
    parser.add_argument(
        "--params_file",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
    )
    args = parser.parse_args()

    if args.params_file is None:
        args.params_file = f"models/{args.model.value}/params.yaml"

    params = load_params(args.params_file, file_type=ParamFileTypes.YAML)

    main(
        model=args.model,
        dataset=args.dataset,
        params_file=args.params_file,
        ckpt_path=args.ckpt_path,
    )
