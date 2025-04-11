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
from gragod.metrics.visualization import save_histograms
from gragod.models import get_model_and_module
from gragod.predictions.prediction import get_system_scores, post_process_scores
from gragod.predictions.threshold_calculator import get_thresholds
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

    Args:
        model: PyTorch Lightning module to run predictions with
        loader: DataLoader containing the input data
        device: Device to run the model on (e.g., 'cpu', 'cuda', 'mps')
        X_true: Ground truth data tensor
        post_process: Whether to apply post-processing to the anomaly scores
        window_size_smooth: Window size for smoothing the anomaly scores
        **kwargs: Additional keyword arguments

    Returns:
        A tuple containing:
            - Anomaly scores tensor
            - Model output (forecast and/or reconstruction)

    Raises:
        ValueError: If model predictions return None
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
    """
    Process a dataset split to generate predictions, scores, and metrics.

    Args:
        model: PyTorch Lightning module to run predictions with
        X_true: Ground truth data tensor
        y: Ground truth labels tensor
        thresholds: Anomaly detection thresholds, or None to calculate them
        device: Device to run the model on
        dataset: Dataset enum value
        model_name: Name of the model being used
        edge_index: Edge index tensor for graph-based models
        save_metrics_dir: Directory to save metrics and visualizations
        dataset_split: Split name ('train', 'val', or 'test')
        window_size: Window size for the model
        batch_size: Batch size for data loading
        n_workers: Number of workers for data loading
        predict_params: Dictionary of prediction parameters

    Returns:
        Dictionary containing:
            - output: Model output (forecast and/or reconstruction)
            - predictions: Anomaly predictions
            - labels: Ground truth labels
            - scores: Anomaly scores
            - data: Input data
            - thresholds: Anomaly detection thresholds
            - metrics: Evaluation metrics

    Raises:
        AssertionError: If there's a shape mismatch between outputs
    """
    start_index = predict_params["start_index"]
    # First `start_index` samples are not predicted
    X_true = X_true[start_index - window_size :]
    y = y[start_index - window_size :]

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

    # Drop everything until the predicted samples
    X_true = X_true[window_size:-1, :]
    y = y[window_size:]
    # Discard last datapoint since it can't be used on recon
    y = y[:-1].int()

    # Run model
    scores, output = run_model(
        model=model,
        loader=loader,
        device=device,
        X_true=X_true,
        post_process=predict_params["post_process_scores"],
        **predict_params,
    )

    forecast, reconstruction = output if isinstance(output, tuple) else (output, None)

    if (
        y.shape[0] != forecast.shape[0]
        or y.shape[0] != scores.shape[0]
        or y.shape[0] != X_true.shape[0]
    ):
        print(
            f"Shape mismatch: y.shape={y.shape},\
            forecast.shape={forecast.shape},\
            scores.shape={scores.shape},\
            X_true.shape={X_true.shape}"
        )
        raise AssertionError("Shape mismatch between y, forecast, and scores")

    # Check reconstruction shape if it exists
    if reconstruction is not None and y.shape[0] != reconstruction.shape[0]:
        print(
            f"Shape mismatch: y.shape={y.shape},\
            reconstruction.shape={reconstruction.shape}"
        )
        raise AssertionError("Shape mismatch between y and reconstruction")

    calculate_threshold = (
        thresholds is None or predict_params["threshold_method"] == "mse_dynamic"
    )
    if calculate_threshold:
        thresholds = get_thresholds(
            dataset=dataset,
            scores=scores,
            labels=y,
            method=predict_params["threshold_method"],
            n_thresholds=predict_params["n_thresholds"],
            range_based=predict_params["range_based"],
            system_output_mode=predict_params.get("system_output_mode", None),
        )

    if y.ndim == 1 or y.shape[1] == 1:
        # We only calculate system metrics if there's only system anomalies
        system_scores = get_system_scores(
            scores=scores,
            mode=predict_params["system_output_mode"],
        )
        system_predictions = (system_scores > thresholds).int()
        system_labels = y
        per_class_y_pred = None

    else:
        system_scores, system_predictions, system_labels = None, None, None
        per_class_y_pred = (scores > thresholds).float()
        system_predictions = None

    # Calculate metrics
    if torch.any(y == 1):
        metrics = get_metrics_and_save(
            dataset=dataset,
            predictions=per_class_y_pred,
            labels=y,
            scores=scores,
            save_dir=save_metrics_dir,
            dataset_split=dataset_split,
            range_metrics_alpha=predict_params["range_metrics_alpha"],
            system_predictions=system_predictions,
            system_labels=system_labels,
            system_scores=system_scores,
        )
    else:
        metrics = None

    if save_metrics_dir:
        save_predictions_dir = os.path.join(save_metrics_dir, "predictions")
        os.makedirs(save_predictions_dir, exist_ok=True)
        save_path = os.path.join(
            save_predictions_dir,
            f"{dataset_split}_{model_name.lower()}_{dataset.value.lower()}",
        )
        torch.save(output, save_path + "_output.pt")
        torch.save(
            per_class_y_pred if per_class_y_pred is not None else system_predictions,
            save_path + "_predictions.pt",
        )
        torch.save(y, save_path + "_labels.pt")
        torch.save(scores, save_path + "_scores.pt")
        torch.save(X_true, save_path + "_data.pt")
        torch.save(thresholds, save_path + "_thresholds.pt")

        save_histograms(
            scores=system_scores if system_scores is not None else scores,
            y=y,
            thresholds=thresholds,
            dataset=dataset,
            dataset_split=dataset_split,
            model_name=model_name,
            save_metrics_dir=save_metrics_dir,
        )
    output_dict: DatasetPredictOutput = {
        "output": output,
        "predictions": (
            per_class_y_pred if per_class_y_pred is not None else system_predictions
        ),
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

    Args:
        model: Model enum value to use for prediction
        dataset: Dataset enum value to predict on
        model_params: Dictionary of model parameters
        batch_size: Batch size for data loading
        ckpt_path: Path to checkpoint file, or None to use default path
        device: Device to run the model on
        n_workers: Number of workers for data loading
        test_size: Fraction of data to use for testing
        val_size: Fraction of data to use for validation
        params: Dictionary of additional parameters
        down_len: Downsampling length, or None for no downsampling
        max_std: Maximum standard deviation for outlier removal
        labels_widening: Whether to widen anomaly labels
        cutoff_value: Cutoff value for data preprocessing
        **kwargs: Additional keyword arguments

    Returns:
        Dictionary containing prediction outputs for train, validation, and test sets
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

        if not torch.any(y_val == 1):
            print(
                "No anomalies in val set either, cannot calculate threshold. "
                "Using test set instead."
            )
            params["predictor_params"]["dataset_for_threshold"] = "test"

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
    elif params["predictor_params"]["dataset_for_threshold"] == "val":
        datasets_to_process = ["val", "train", "test"]
    elif params["predictor_params"]["dataset_for_threshold"] == "test":
        datasets_to_process = ["test", "train", "val"]
    else:
        raise ValueError(
            f"Invalid dataset for threshold: "
            f"{params['predictor_params']['dataset_for_threshold']}"
        )
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
    start_index: int | None = None,
) -> PredictOutput:
    """
    Main function to load data, model and generate predictions.

    Args:
        model: Name of model to predict
        dataset: Dataset to predict on
        ckpt_path: Path to checkpoint file, or None to use default path
        params_file: Path to parameter file
        start_index: Starting index for predictions, or None to use default

    Returns:
        Dictionary containing prediction outputs for train, validation, and test sets

    Raises:
        AssertionError: If start_index is less than the model's window size
    """
    params = load_params(params_file, file_type=ParamFileTypes.YAML)
    set_seeds(RANDOM_SEED)

    if start_index is not None:
        assert (
            start_index >= params["model_params"]["window_size"]
        ), "The start index should be greater than or equal to the model's window size"

        params["predictor_params"]["start_index"] = start_index

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
    parser.add_argument(
        "--start_index",
        "-si",
        type=int,
        default=None,
        help="If a variety of models is being tested, the maximum window size of the"
        "first model is used to drop the initial points from the other models",
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
        start_index=args.start_index,
    )
