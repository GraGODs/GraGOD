from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tabulate
import torch

from gragod.utils import count_anomaly_ranges


def generate_metrics_table(metrics: dict, only_system: bool = False) -> str:
    """Generate a table of metrics as a string."""

    # Define metric groups and their display names
    metric_groups = {}
    for metric in metrics.keys():
        if metric.endswith("_system"):
            metric_name = metric.replace("_system", "")
            metric_groups[metric_name] = metric_name.title()

    # Create headers
    if only_system:
        metrics_table = [["System"]]
    else:
        metrics_table = [["Metric", "Global", "Mean", "System"]]

    # Build table rows dynamically
    for metric_key, metric_name in metric_groups.items():
        if only_system:
            row = [
                f"{metrics.get(f'{metric_key}_system', '')}",
            ]
        else:
            row = [
                metric_name,
                f"{metrics.get(f'{metric_key}_global', '')}",
                f"{metrics.get(f'{metric_key}_mean', '')}",
                f"{metrics.get(f'{metric_key}_system', '')}",
            ]
        metrics_table.append(row)

    return tabulate.tabulate(metrics_table, headers="firstrow", tablefmt="grid")


def generate_metrics_per_class_table(metrics: dict) -> str:
    """Generate a table of per-class metrics as a string."""

    n_classes = 0
    metrics_per_class = {}
    for metric in metrics.keys():
        if metric.endswith("_per_class"):
            metrics_per_class[metric] = metrics[metric]
            n_classes = len(metrics_per_class[metric])

    if n_classes == 0:
        raise ValueError("No per-class metrics found")

    metrics_per_class_table = []
    for i in range(n_classes):
        table_i = []
        table_i.append(i)
        for metric in metrics_per_class.keys():
            table_i.append(metrics_per_class[metric][i])
        metrics_per_class_table.append(table_i)

    headers = ["Class"] + [
        key.replace("_per_class", "").title() for key in metrics_per_class.keys()
    ]

    return tabulate.tabulate(
        metrics_per_class_table,
        headers=headers,
        tablefmt="grid",
    )


def print_all_metrics(metrics: dict, message: str):
    print(message)
    if "precision_per_class" in metrics:
        metrics_table = generate_metrics_table(metrics)
        print(metrics_table)
        metrics_per_class_table = generate_metrics_per_class_table(metrics)
        print(metrics_per_class_table)
    else:
        metrics_table = generate_metrics_table(metrics, only_system=True)
        print(metrics_table)


def get_metrics_per_class(metrics: dict, n_classes: int):
    metrics_per_class = defaultdict(dict)
    for metric_name, metric_value in metrics.items():
        if "per_class" in metric_name:
            for i in range(n_classes):
                metrics_per_class[i][metric_name.split("_per_class")[0]] = metric_value[
                    i
                ]
    return metrics_per_class


def get_metrics_mean(metrics: dict):
    metrics_mean = {}
    for metric_name, metric_value in metrics.items():
        if "mean" in metric_name:
            metrics_mean[metric_name.split("_mean")[0]] = metric_value
    return metrics_mean


def plot_single_score_histogram(
    scores: torch.Tensor,
    labels: torch.Tensor,
    metrics: dict | None = None,
    use_ranged_anomalies=False,
    model_name: str = "GRU",
    dataset_name: str = "Telco",
):
    fig, ax = plt.subplots(figsize=(10, 5))

    # Convert to numpy arrays
    scores_np = scores.clone().numpy()
    labels_np = labels.clone().numpy()

    # Handle negative scores
    min_score = np.min(scores_np)
    if min_score < 0:
        offset = np.abs(min_score) + 0.0001
        scores_np += offset

    # Get updated score range after adjustment
    min_score = np.min(scores_np)
    max_score = np.max(scores_np)

    # Split scores
    anomalous_scores = scores_np[labels_np == 1]
    normal_scores = scores_np[labels_np == 0]

    if use_ranged_anomalies:
        max_anomalous_scores = []
        anomaly_ranges = count_anomaly_ranges(pd.DataFrame(labels_np))
        for anomaly_range in anomaly_ranges:
            for start, end in zip(
                anomaly_range["start_times"], anomaly_range["end_times"]
            ):
                max_anomalous_scores.extend(
                    [np.max(scores_np[start:end])] * len(scores_np[start:end])
                )
        anomalous_scores = max_anomalous_scores

    # Create bins based on adjusted scores
    bin_edges = np.logspace(np.log10(min_score), np.log10(max_score), num=30)

    ax.hist(
        normal_scores,
        bins=bin_edges,
        alpha=0.9,
        label="Normal",
        color="springgreen",
        log=True,
        edgecolor="black",
    )
    ax.hist(
        anomalous_scores,
        bins=bin_edges,
        alpha=0.7,
        label="Anomalous",
        color="salmon",
        log=True,
        edgecolor="black",
    )
    ax.set_xscale("log")
    ax.set_title(f"{model_name} - {dataset_name} - Histogram of Anomaly Scores")
    ax.set_xlabel("Anomaly Score (log scale)", fontsize=16)
    ax.set_ylabel("Number of Samples (log scale)", fontsize=16)
    ax.legend(title="Label", fontsize=15, title_fontsize=15, loc="upper left")

    # Add metrics if provided
    if metrics:
        metrics_mean = get_metrics_mean(metrics)
        metrics_text = "\n".join([f"{k}: {v:.2f}" for k, v in metrics_mean.items()])
        ax.text(
            0.02,
            0.77,
            metrics_text,
            transform=ax.transAxes,
            fontsize=8,
            bbox=dict(facecolor="white", alpha=0.5),
        )

    return fig


def plot_score_histograms_grid_telco(
    scores: torch.Tensor,
    labels: torch.Tensor,
    thresholds: torch.Tensor,
    metrics: dict | None = None,
    use_ranged_anomalies: bool = False,
    log_axis: bool = True,
):
    scores
    metrics_per_class = (
        get_metrics_per_class(metrics=metrics, n_classes=scores.shape[1])
        if metrics is not None
        else None
    )
    num_classes = scores.shape[1]
    num_rows = 4
    num_cols = 3
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 15))
    axes = axes.flatten()

    for i in range(num_classes):
        ax = axes[i]
        scores_class = scores[:, i].clone().numpy()
        labels_class = labels[:, i].clone().numpy()
        threshold_class = thresholds[i].clone()

        min_score = np.min(scores_class)
        if min_score:
            scores_class = scores_class + np.abs(min_score) + 0.0001
            threshold_class = threshold_class + np.abs(min_score) + 0.0001

        anomalous_scores = scores_class[labels_class == 1]
        normal_scores = scores_class[labels_class == 0]
        if anomalous_scores.size == 0:
            continue

        if use_ranged_anomalies:
            max_anomalous_scores = []
            anomaly_ranges = count_anomaly_ranges(pd.DataFrame(labels_class))
            for anomaly_range in anomaly_ranges:
                for start, end in zip(
                    anomaly_range["start_times"], anomaly_range["end_times"]
                ):
                    max_anomalous_scores.extend(
                        [np.max(scores_class[start:end])] * len(scores_class[start:end])
                    )
            anomalous_scores = max_anomalous_scores
            min_val = min(np.min(normal_scores), np.min(max_anomalous_scores))
            max_val = max(np.max(normal_scores), np.max(max_anomalous_scores))

        else:
            min_val = min(np.min(normal_scores), np.min(anomalous_scores))
            max_val = max(np.max(normal_scores), np.max(anomalous_scores))

        # Handle negative values by using linear spacing when needed
        if log_axis:
            bin_edges = np.logspace(np.log10(min_val), np.log10(max_val), num=30)
        else:
            bin_edges = np.linspace(min_val, max_val, num=30)

        ax.hist(
            normal_scores,
            bins=bin_edges,
            alpha=0.9,
            label="Normal",
            color="springgreen",
            log=True,
            edgecolor="black",
        )
        ax.hist(
            anomalous_scores,
            bins=bin_edges,
            alpha=0.7,
            label="Anomalous",
            color="salmon",
            log=True,
            edgecolor="black",
        )

        # ax.set_xlabel("Anomaly Score (log scale)", fontsize=10)
        # ax.set_ylabel("Number of Samples (log scale)", fontsize=10)
        ax.set_title(f"TS{i+1}", fontsize=20)
        ax.axvline(threshold_class, color="blue", linestyle="--", label="Threshold")

        # Display metrics on the plot
        if metrics_per_class:
            metrics_text = "\n".join(
                [f"{k}: {v:.2f}" for k, v in metrics_per_class[i].items()]
            )
            ax.text(
                0.02,
                0.77,
                metrics_text,
                transform=ax.transAxes,
                fontsize=8,
                verticalalignment="top",
                horizontalalignment="left",
                bbox=dict(facecolor="white", alpha=0.5),
            )

        ax.legend(title="Label", fontsize=15, title_fontsize=15, loc="upper left")
        ax.set_xscale("log" if log_axis else "linear")

    # Hide unused axes
    for j in range(num_classes, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    # plt.show()
    return fig
