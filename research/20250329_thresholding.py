# %%
from gragod.start_research import *
from datasets.swat import load_swat_df
from datasets.telco import load_telco_df
from gragod.metrics.visualization import generate_metrics_table

dataset_name = "swat"

df_train, *_ = load_swat_df() if dataset_name == "swat" else load_telco_df()
column_names = df_train.columns.tolist()

# %%
import torch
import os
import math
import itertools
import matplotlib.pyplot as plt
import numpy as np
from gragod.types import cast_dataset
from gragod.metrics.visualization import (
    plot_single_score_histogram,
    plot_score_histograms_grid_telco,
)
from gragod.metrics.calculator import get_metrics
import importlib
import sys
from gragod.predictions.prediction import get_system_scores

if "gragod.predictions.prediction" in sys.modules:
    importlib.reload(sys.modules["gragod.predictions.prediction"])
from gragod.predictions.threshold_calculator import get_thresholds
import tabulate


N_THRESHOLDS = 1000
models = ["gdn", "gcn", "mtad_gat", "gru"]
version = "version_0"
plot_legend = True
for model in models:
    path_to_model = (
        f"/home/fbello/GraGOD/output/benchmark/{dataset_name}/{model}/{version}"
    )

    predictions_path = path_to_model + f"/predictions"
    test_predictions = torch.load(
        predictions_path + f"/test_{model}_{dataset_name}_predictions.pt"
    )
    test_data = torch.load(predictions_path + f"/test_{model}_{dataset_name}_data.pt")
    test_labels = torch.load(
        predictions_path + f"/test_{model}_{dataset_name}_labels.pt"
    ).int()
    test_scores = torch.load(
        predictions_path + f"/test_{model}_{dataset_name}_scores.pt"
    )
    test_outputs = torch.load(
        predictions_path + f"/test_{model}_{dataset_name}_output.pt"
    )
    val_predictions = torch.load(
        predictions_path + f"/val_{model}_{dataset_name}_predictions.pt"
    )
    val_data = torch.load(predictions_path + f"/val_{model}_{dataset_name}_data.pt")
    val_labels = torch.load(
        predictions_path + f"/val_{model}_{dataset_name}_labels.pt"
    ).int()
    val_scores = torch.load(predictions_path + f"/val_{model}_{dataset_name}_scores.pt")
    val_outputs = torch.load(
        predictions_path + f"/val_{model}_{dataset_name}_output.pt"
    )

    train_scores = torch.load(
        predictions_path + f"/train_{model}_{dataset_name}_scores.pt"
    )
    train_labels = torch.load(
        predictions_path + f"/train_{model}_{dataset_name}_labels.pt"
    ).int()

    val_system_scores = get_system_scores(val_scores, "mean").squeeze()
    test_system_scores = get_system_scores(test_scores, "mean").squeeze()

    # Define threshold methods and their parameters
    threshold_methods = {
        "Dynamic Threshold": {
            "method": "mse_dynamic",
            "scores": test_scores,
            "labels": test_labels,
            "window_size": 100 if dataset_name == "telco" else 30,
            "k": 2 if dataset_name == "telco" else 1,
        },
        "F1 Optimized": {
            "method": "f1_optimize",
            "scores": val_scores,
            "labels": val_labels,
        },
        "Otsu": {
            "method": "otsu",
            "scores": val_scores,
            "labels": val_labels,
        },
        "GMM": {
            "method": "gmm",
            "scores": val_scores,
            "labels": val_labels,
        },
    }

    # Calculate all thresholds
    all_thresholds = {}
    for name, params in threshold_methods.items():
        # Set default parameters
        method_params = {
            "dataset": cast_dataset(dataset_name),
            "n_thresholds": N_THRESHOLDS,
            "range_based": True,
            "range_metrics_alpha": 0.5,
            "system_output_mode": "mean",
        }
        # Add method-specific parameters
        method_params.update(params)

        # Calculate threshold
        all_thresholds[name] = get_thresholds(**method_params)

    all_metrics = []

    for name, threshold in all_thresholds.items():
        if not dataset_name == "telco":
            test_predictions = (test_system_scores > threshold.squeeze()).int()
        metrics = (
            get_metrics(
                dataset=cast_dataset(dataset_name),
                range_metrics_alpha=0.5,
                predictions=(test_scores > threshold).int(),
                labels=test_labels,
                scores=test_scores,
            )
            if dataset_name == "telco"
            else get_metrics(
                dataset=cast_dataset(dataset_name),
                range_metrics_alpha=0.5,
                system_predictions=test_predictions,
                system_labels=test_labels.squeeze(),
                system_scores=test_system_scores,
            )
        )
        # Remove all VUS metrics
        for key in list(metrics.keys()):
            if key.startswith("vus") or key.startswith("custom"):
                metrics.pop(key)
        metrics_table = generate_metrics_table(metrics)
        print(f"Method: {name}, model: {model}")
        print(metrics_table)

    all_thresholds.pop("Dynamic Threshold")
    if dataset_name == "swat":
        fig_test = plot_single_score_histogram(
            scores=test_system_scores,
            labels=test_labels.squeeze(),
            thresholds=all_thresholds,
            model_name=model,
            dataset_name=dataset_name,
            plot_legend=plot_legend,
        )
        fig_val = plot_single_score_histogram(
            scores=val_system_scores,
            labels=val_labels.squeeze(),
            thresholds=all_thresholds,
            model_name=model,
            dataset_name=dataset_name,
            plot_legend=plot_legend,
        )
        fig_test.savefig(
            f"/home/fbello/GraGOD/output/benchmark/figures/test_histogram_{model}_{dataset_name}_test_threshold.pdf",
            dpi=1200,
            bbox_inches="tight",
        )
        fig_val.savefig(
            f"/home/fbello/GraGOD/output/benchmark/figures/val_histogram_{model}_{dataset_name}.pdf",
            dpi=1200,
            bbox_inches="tight",
        )
        plot_legend = False
    else:
        plot_score_histograms_grid_telco(
            scores=torch.cat([val_scores, train_scores]),
            labels=torch.cat([val_labels, train_labels]),
            thresholds=all_thresholds,
        )
# %%
