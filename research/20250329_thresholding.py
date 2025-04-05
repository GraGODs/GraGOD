# %%
from gragod.start_research import *
from datasets.swat import load_swat_df
from datasets.telco import load_telco_df

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


N_THRESHOLDS = 10
# models = ["gdn", "gcn", "mtad_gat", "gru"]
models = ["gcn"]
version = "version_0"
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
    )
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

    val_system_scores = torch.mean(val_scores, dim=1)
    test_system_scores = torch.mean(test_scores, dim=1)

    dynamic_threshold_val = get_thresholds(
        dataset=cast_dataset(dataset_name),
        scores=val_scores,
        labels=val_labels,
        method="mse_dynamic",
        n_thresholds=N_THRESHOLDS,
        range_based=False,
        range_metrics_alpha=0.5,
        window_size=100,
        system_output_mode="mean",
    )

    print(f"Shape of dynamic_threshold_val: {dynamic_threshold_val.shape}")

    f1_optimized_threshold_val = get_thresholds(
        dataset=cast_dataset(dataset_name),
        scores=val_scores,
        labels=val_labels,
        method="f1_optimize",
        n_thresholds=N_THRESHOLDS,
        range_based=False,
        range_metrics_alpha=0.5,
        system_output_mode="mean",
    )

    print(f"Shape of f1_optimized_threshold_val: {f1_optimized_threshold_val.shape}")

    otsu_threshold_val = get_thresholds(
        dataset=cast_dataset(dataset_name),
        scores=val_scores,
        labels=val_labels,
        method="otsu",
        n_thresholds=N_THRESHOLDS,
        range_based=False,
        range_metrics_alpha=0.5,
        system_output_mode="mean",
    )
    print(f"Values of otsu_threshold_val: {otsu_threshold_val}")
    print(f"Shape of otsu_threshold_val: {otsu_threshold_val.shape}")

    gmm_threshold_val = get_thresholds(
        dataset=cast_dataset(dataset_name),
        scores=val_scores,
        labels=val_labels,
        method="gmm",
        n_thresholds=N_THRESHOLDS,
        range_based=False,
        range_metrics_alpha=0.5,
        system_output_mode="mean",
    )

    print(f"Values of gmm_threshold_val: {gmm_threshold_val}")
    print(f"Shape of gmm_threshold_val: {gmm_threshold_val.shape}")

    # plot_single_score_histogram(
    #     scores=get_system_scores(val_scores, "mean"),
    #     labels=val_labels,
    # )
