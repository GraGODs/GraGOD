import argparse
import os
from pathlib import Path
from typing import Any, cast

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from datasets.config import get_dataset_config
from datasets.dataset import get_data_loader
from datasets.graph import get_edge_index
from gragod import CleanMethods, Datasets, Models, ParamFileTypes
from gragod.metrics.calculator import get_metrics_and_save
from gragod.metrics.visualization import (
    plot_score_histograms_grid_telco,
    plot_single_score_histogram,
)
from gragod.models import get_model_and_module
from gragod.predictions.prediction import get_threshold, post_process_scores
from gragod.training import load_params, load_training_data, set_seeds
from gragod.utils import load_checkpoint_path, set_device
from models.schemas import DatasetPredictOutput, PredictOutput

RANDOM_SEED = 42


def run_model(
    model: pl.LightningModule,
    loader: DataLoader,
    device: str,
    X_true: torch.Tensor,
    post_process: bool = True,
    window_size_smooth: int = 5,
    **kwargs,
) -> tuple[torch.Tensor, Any]:
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

    output = model.post_process_predictions(output)

    if post_process:
        print(f"Post processing scores with window size {window_size_smooth}")
        scores = post_process_scores(scores, window_size=window_size_smooth)

    return scores, output


def calculate_metrics(
    scores: torch.Tensor,
    threshold: torch.Tensor,
    y: torch.Tensor,
    dataset: Datasets,
    dataset_split: str,
    save_dir: Path,
):
    y_pred = (scores > threshold).float()

    metrics = get_metrics_and_save(
        dataset=dataset,
        predictions=y_pred,
        labels=y,
        scores=scores,
        save_dir=save_dir,
        dataset_split=dataset_split,
    )
    return metrics, y_pred


def process_dataset(
    model: pl.LightningModule,
    X_true: torch.Tensor,
    y: torch.Tensor,
    thresholds: torch.Tensor | None,
    device: str,
    dataset: Datasets,
    model_name: str,
    edge_index: torch.Tensor,
    save_metrics_dir: Path,
    dataset_split: str,
    window_size: int = 5,
    batch_size: int = 264,
    n_workers: int = 0,
    predict_params: dict = {},
):
    # Create test dataloader
    loader = get_data_loader(
        X=X_true,
        edge_index=edge_index,
        y=y,
        window_size=window_size,
        clean=CleanMethods.NONE,
        batch_size=batch_size,
        n_workers=n_workers,
        shuffle=False,
    )

    # First `window_size` samples are not used for prediction
    X_true = X_true[window_size:]
    y = y[window_size:]

    # Run model
    scores, output = run_model(
        model=model,
        loader=loader,
        device=device,
        X_true=X_true,
        **predict_params,
    )

    # Discard last datapoint since it can't be used on recon
    y = y[:-1]

    if thresholds is None:
        thresholds = get_threshold(
            dataset=dataset,
            scores=scores,
            labels=y,
            n_thresholds=predict_params["n_thresholds"],
            range_based=predict_params["range_based"],
        )

    # Calculate metrics
    if torch.any(y == 1):
        metrics, y_pred = calculate_metrics(
            scores=scores,
            threshold=thresholds,
            y=y,
            dataset=dataset,
            dataset_split=dataset_split,
            save_dir=save_metrics_dir,
        )
    else:
        metrics = None
        y_pred = None

    if save_metrics_dir:
        save_predictions_dir = os.path.join(save_metrics_dir, "predictions")
        os.makedirs(save_predictions_dir, exist_ok=True)
        save_path = os.path.join(
            save_predictions_dir,
            f"{dataset_split}_{model_name.lower()}_{dataset.value.lower()}",
        )
        torch.save(output, save_path + "_output.pt")
        torch.save(y_pred, save_path + "_predictions.pt")
        torch.save(y, save_path + "_labels.pt")
        torch.save(scores, save_path + "_scores.pt")
        torch.save(X_true, save_path + "_data.pt")
        torch.save(thresholds, save_path + "_thresholds.pt")

    # Plots
    save_plots_dir = os.path.join(save_metrics_dir, "plots")
    os.makedirs(save_plots_dir, exist_ok=True)
    if dataset == Datasets.TELCO:
        fig_1 = plot_score_histograms_grid_telco(
            scores=scores,
            labels=y,
            thresholds=thresholds,
        )
        fig_2 = plot_score_histograms_grid_telco(
            scores=scores,
            labels=y,
            thresholds=thresholds,
            use_ranged_anomalies=True,
        )
        fig_1.savefig(
            os.path.join(
                save_plots_dir,
                f"{dataset_split}_{model_name.lower()}_{dataset.value.lower()}"
                + "_score_histograms.png",
            )
        )
        fig_2.savefig(
            os.path.join(
                save_plots_dir,
                f"{dataset_split}_{model_name.lower()}_{dataset.value.lower()}"
                + "_score_histograms_with_ranges.png",
            )
        )

    fig_1 = plot_single_score_histogram(
        scores=scores.flatten(),
        labels=y.flatten(),
        use_ranged_anomalies=False,
        model_name=model_name,
        dataset_name=dataset.value,
    )
    fig_2 = plot_single_score_histogram(
        scores=scores.flatten(),
        labels=y.flatten(),
        use_ranged_anomalies=True,
        model_name=model_name,
        dataset_name=dataset.value,
    )

    fig_1.savefig(
        os.path.join(
            save_plots_dir,
            f"{dataset_split}_{model_name.lower()}_{dataset.value.lower()}"
            + "_score_histogram_single.png",
        )
    )
    fig_2.savefig(
        os.path.join(
            save_plots_dir,
            f"{dataset_split}_{model_name.lower()}_{dataset.value.lower()}"
            + "_score_histogram_single_with_ranges.png",
        )
    )
    output_dict: DatasetPredictOutput = {
        "output": output,
        "predictions": y_pred,
        "labels": y,
        "scores": scores,
        "data": X_true,
        "thresholds": thresholds,
        "metrics": metrics,
    }
    return output_dict


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
    max_std: float | None = None,
    labels_widening: bool = False,
    cutoff_value: float | None = None,
    **kwargs,
) -> PredictOutput:
    """
    Main function to load data, model and generate predictions.
    Returns a dictionary containing evaluation metrics.
    """
    torch.set_float32_matmul_precision("high")
    device = set_device()
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
        max_std=max_std,
        labels_widening=labels_widening,
        cutoff_value=cutoff_value,
    )
    edge_index = get_edge_index(
        X_train, device, model_params.get("edge_index_path", None)
    )

    window_size = model_params["window_size"]

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

    # Create and load model
    _, model_pl_module = get_model_and_module(model)
    model_params["edge_index"] = [edge_index]
    model_params["n_features"] = X_train.shape[1]
    model_params["out_dim"] = X_train.shape[1]

    checkpoint_path = (
        load_checkpoint_path(
            checkpoint_path=params["predictor_params"]["ckpt_folder"],
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

    # Process each dataset split
    dataset_arguments = {
        "train": {"X_true": X_train, "y": y_train},
        "val": {"X_true": X_val, "y": y_val},
        "test": {"X_true": X_test, "y": y_test},
    }

    if params["predictor_params"]["dataset_for_threshold"] == "train":
        datasets_to_process = ["train", "val", "test"]
    elif params["predictor_params"]["dataset_for_threshold"] == "test":
        datasets_to_process = ["test", "train", "val"]
    else:
        datasets_to_process = ["val", "train", "test"]
    thresholds = None
    return_dict = {}

    for dataset_split in datasets_to_process:
        output_dict = process_dataset(
            model=lightning_module,
            X_true=dataset_arguments[dataset_split]["X_true"],
            y=dataset_arguments[dataset_split]["y"],
            thresholds=thresholds,
            device=device,
            dataset=dataset,
            model_name=params["train_params"]["model_name"],
            save_metrics_dir=checkpoint_path.parent,
            dataset_split=dataset_split,
            edge_index=edge_index,
            window_size=window_size,
            batch_size=batch_size,
            n_workers=n_workers,
            predict_params=params["predictor_params"],
        )
        if thresholds is None:
            thresholds = output_dict["thresholds"]
        return_dict[dataset_split] = output_dict

    return_dict = cast(PredictOutput, return_dict)
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
        help=f"Model to train [{', '.join(model.value for model in Models)}]",
    )
    parser.add_argument(
        "--dataset",
        type=Datasets,
        help=f"Dataset to predict [{', '.join(dataset.value for dataset in Datasets)}]",
    )
    parser.add_argument(
        "--params_file",
        type=str,
        default=None,
        help="Path to parameter file",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="Path to checkpoint file",
    )
    args = parser.parse_args()

    if args.params_file is None:
        args.params_file = f"models/{args.model.value}/params.yaml"

    if args.ckpt_path is not None and not args.ckpt_path.endswith(".ckpt"):
        raise ValueError(
            "Checkpoint path must end with .ckpt, got "
            f"{args.ckpt_path} with extension {Path(args.ckpt_path).suffix}"
        )

    params = load_params(args.params_file, file_type=ParamFileTypes.YAML)

    main(
        model=args.model,
        dataset=args.dataset,
        params_file=args.params_file,
        ckpt_path=args.ckpt_path,
    )
