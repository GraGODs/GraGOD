from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tabulate
import torch
from matplotlib.axes import Axes

from gragod.predictions.prediction import get_system_scores
from gragod.utils import count_anomaly_ranges


def generate_metrics_table(metrics: dict) -> str:
    """
    Generate a table of metrics as a string.

    Args:
        metrics: Dictionary containing metric names as keys and their values

    Returns:
        A formatted string table of metrics
    """
    # Define metric groups and their display names
    metric_groups = {}
    metric_types = set()
    for metric in metrics.keys():
        if not metric.endswith("_per_class"):
            metric_types.add(metric.split("_")[-1])
            metric_name = "_".join(metric.split("_")[:-1])
            metric_groups[metric_name] = metric_name.title()

    # Create headers
    headers = [str(metric_type).capitalize() for metric_type in metric_types]
    metrics_table = [["Metric"] + headers]

    # Build table rows dynamically
    for metric_key, metric_name in metric_groups.items():
        row = []
        row.append(metric_name)
        for metric_type in metric_types:
            row.append(f"{metrics.get(f'{metric_key}_{metric_type}', '')}")
        metrics_table.append(row)

    return tabulate.tabulate(metrics_table, headers="firstrow", tablefmt="grid")


def generate_metrics_per_class_table(metrics: dict) -> str:
    """
    Generate a table of per-class metrics as a string.

    Args:
        metrics: Dictionary containing metric names as keys and their values

    Returns:
        A formatted string table of per-class metrics

    Raises:
        ValueError: If no per-class metrics are found in the input dictionary
    """
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
    """
    Print all metrics with a message header.

    Args:
        metrics: Dictionary containing metric names as keys and their values
        message: Message to display before printing the metrics
    """
    print(message)
    if "precision_per_class" in metrics:
        metrics_table = generate_metrics_table(metrics)
        print(metrics_table)
        metrics_per_class_table = generate_metrics_per_class_table(metrics)
        print(metrics_per_class_table)
    else:
        metrics_table = generate_metrics_table(metrics)
        print(metrics_table)


def get_metrics_per_class(metrics: dict, n_classes: int):
    """
    Extract per-class metrics from the metrics dictionary.

    Args:
        metrics: Dictionary containing metric names as keys and their values
        n_classes: Number of classes

    Returns:
        Dictionary with class indices as keys and their metrics as values
    """
    metrics_per_class = defaultdict(dict)
    for metric_name, metric_value in metrics.items():
        if "per_class" in metric_name:
            for i in range(n_classes):
                metrics_per_class[i][metric_name.split("_per_class")[0]] = metric_value[
                    i
                ]
    return metrics_per_class


def get_metrics_mean(metrics: dict):
    """
    Extract mean metrics from the metrics dictionary.

    Args:
        metrics: Dictionary containing metric names as keys and their values

    Returns:
        Dictionary with metric names as keys and their mean values
    """
    metrics_mean = {}
    for metric_name, metric_value in metrics.items():
        if "mean" in metric_name:
            metrics_mean[metric_name.split("_mean")[0]] = metric_value
    return metrics_mean


def plot_score_histograms(
    scores: torch.Tensor,
    labels: torch.Tensor,
    thresholds: dict[str, torch.Tensor] | None = None,
    plot_legend: bool = True,
    use_ranged_anomalies: bool = False,
    title: str | None = None,
    save_path: str | None = None,
    grid_title: str | None = None,
    titles: list[str] | None = None,
    num_cols: int = 3,
):
    """
    Plot a histogram of anomaly scores for a single feature.

    It plots a single histogram or a grid depending on the shape of the labels tensor

    Args:
        scores: Tensor of anomaly scores
        labels: Tensor of ground truth labels (0 for normal, 1 for anomalous)
        thresholds: Dictionary with threshold name as key and threshold value as tensor
        plot_legend: Whether to plot the legend
        use_ranged_anomalies: Whether to use the maximum score within each anomaly range
        title: Title of the plot
        save_path: Path to save the figure
        grid_title: Title of the grid
        titles: List of titles for each subplot
        num_cols: Number of columns in the grid
    """
    if labels.ndim == 1 or labels.shape[1] == 1:
        return plot_single_score_histogram(
            scores=get_system_scores(scores, "mean"),
            labels=labels.squeeze(),
            thresholds=thresholds,
            plot_legend=plot_legend,
            use_ranged_anomalies=use_ranged_anomalies,
            title=title,
            save_path=save_path,
        )
    else:
        return plot_grid_score_histograms(
            scores=scores,
            labels=labels,
            thresholds=thresholds,
            plot_legend=plot_legend,
            use_ranged_anomalies=use_ranged_anomalies,
            grid_title=grid_title,
            titles=titles,
            num_cols=num_cols,
            save_path=save_path,
        )


def plot_single_score_histogram(
    scores: torch.Tensor,
    labels: torch.Tensor,
    thresholds: dict[str, torch.Tensor] | None = None,
    plot_legend: bool = True,
    use_ranged_anomalies: bool = False,
    title: str | None = None,
    save_path: str | None = None,
):
    """
    Plot a histogram of anomaly scores for a single feature.

    Args:
        scores: Tensor of anomaly scores
        labels: Tensor of ground truth labels (0 for normal, 1 for anomalous)
        thresholds: Dictionary with threshold name as key and threshold value as tensor
        plot_legend: Whether to plot the legend
        use_ranged_anomalies: Whether to use the maximum score within each anomaly range
        title: Title of the plot
        save_path: Path to save the figure

    Returns:
        Matplotlib figure object containing the histogram
    """
    scores = scores.squeeze()
    if scores.ndim not in [0, 1]:
        raise ValueError(f"Scores must be 1D, got shape: {scores.shape}")

    fig, ax = plt.subplots(figsize=(10, 8))
    _plot_histogram_on_axis(
        ax=ax,
        scores=scores.clone().numpy(),
        labels=labels.clone().numpy(),
        use_ranged_anomalies=use_ranged_anomalies,
        plot_legend=plot_legend,
        thresholds=thresholds,
        feature_idx=None,
    )

    if title:
        ax.set_title(f"{title}")

    ax.set_xlabel("Score", fontsize=20)
    ax.set_ylabel("Number of Samples (log scale)", fontsize=20)
    ax.tick_params(axis="both", labelsize=20)

    if save_path:
        plt.savefig(save_path, dpi=1200, bbox_inches="tight")
    return fig


def plot_grid_score_histograms(
    scores: torch.Tensor,
    labels: torch.Tensor,
    thresholds: dict[str, torch.Tensor] | None = None,
    plot_legend: bool = True,
    use_ranged_anomalies: bool = False,
    grid_title: str | None = None,
    titles: list[str] | None = None,
    num_cols: int = 3,
    save_path: str | None = None,
):
    """
    Plot a grid of histograms for Telco dataset features.

    Args:
        scores: Tensor of anomaly scores with shape (n_samples, n_features)
        labels: Tensor of ground truth labels with shape (n_samples, n_features)
        thresholds: Tensor of thresholds for each feature
        plot_legend: Whether to plot the legend
        use_ranged_anomalies: Whether to use the maximum score within each anomaly range
        grid_title: Title for the entire grid
        titles: List of titles for each subplot
        num_cols: Number of columns in the grid
        save_path: Path to save the figure

    Returns:
        Matplotlib figure object containing the grid of histograms
    """
    if thresholds:
        for threshold_name, threshold_values in thresholds.items():
            if threshold_values.shape[0] != scores.shape[1]:
                raise ValueError(
                    "Threshold length must have the same as the feature number"
                    f"\nGot {threshold_values.shape[0]} and {scores.shape[1]}"
                )

    if titles:
        if len(titles) != scores.shape[1]:
            raise ValueError(
                f"""Titles must have the same number of features as the scores\n\
                Got {len(titles)} and {scores.shape[1]}"""
            )

    num_classes = scores.shape[1]
    num_rows = (num_classes + num_cols - 1) // num_cols
    subplot_size = (8, 6)
    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(subplot_size[0] * num_cols, subplot_size[1] * num_rows),
    )
    axes = axes.flatten()

    for i in range(num_classes):
        ax = axes[i]
        scores_class = scores[:, i].clone().numpy()
        labels_class = labels[:, i].clone().numpy()

        _plot_histogram_on_axis(
            ax=ax,
            scores=scores_class,
            labels=labels_class,
            use_ranged_anomalies=use_ranged_anomalies,
            plot_legend=plot_legend,
            thresholds=thresholds,
            feature_idx=i,
        )

        ax.tick_params(axis="both", labelsize=20)

        # Only add x-axis label to bottom row
        if i >= num_classes - num_cols:
            ax.set_xlabel("Score", fontsize=20)

        # Only add y-axis label to leftmost column
        if i % num_cols == 0:
            ax.set_ylabel("Number of Samples (log scale)", fontsize=20)

        if titles:
            ax.set_title(titles[i], fontsize=20)

    # Hide unused axes
    for j in range(num_classes, len(axes)):
        axes[j].set_visible(False)

    if grid_title:
        fig.suptitle(grid_title, fontsize=20)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=1200, bbox_inches="tight")
    return fig


def _plot_histogram_on_axis(
    ax: Axes,
    scores: np.ndarray,
    labels: np.ndarray,
    use_ranged_anomalies: bool = False,
    plot_legend: bool = True,
    thresholds: dict[str, torch.Tensor] | None = None,
    feature_idx: int | None = None,
):
    """
    Helper function to plot histogram on a given axis.

    Args:
        ax: Matplotlib axis to plot on
        scores: Numpy array of scores
        labels: Numpy array of labels
        use_ranged_anomalies: Whether to use the maximum score within each anomaly range
        plot_legend: Whether to plot the legend
        thresholds: Dictionary with threshold name as key and threshold value as tensor
        feature_idx: Index of the feature being plotted (for multi-feature thresholds)
    """
    anomalous_scores = scores[labels == 1]
    normal_scores = scores[labels == 0]

    if use_ranged_anomalies and len(anomalous_scores) > 0:
        max_anomalous_scores = []
        anomaly_ranges = count_anomaly_ranges(pd.DataFrame(labels))
        for anomaly_range in anomaly_ranges:
            for start, end in zip(
                anomaly_range["start_times"], anomaly_range["end_times"]
            ):
                max_anomalous_scores.extend(
                    [np.max(scores[start:end])] * len(scores[start:end])
                )
        anomalous_scores = max_anomalous_scores

    min_score = np.min(scores)
    max_score = np.max(scores)

    bin_edges = np.linspace(min_score, max_score, num=30).tolist()

    ax.hist(
        normal_scores,
        bins=bin_edges,
        alpha=0.9,
        label="Normal",
        color="springgreen",
        edgecolor="black",
        log=True,
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

    # Define a list of colors for different thresholds
    colors = [
        "blue",
        "red",
        "green",
        "purple",
        "orange",
        "brown",
        "magenta",
        "cyan",
    ]

    if thresholds:
        for i, (threshold_name, threshold_value) in enumerate(thresholds.items()):
            color = colors[i % len(colors)]
            if feature_idx is not None:
                threshold = threshold_value[feature_idx].item()
            else:
                threshold = threshold_value.item()
            ax.axvline(
                threshold,
                color=color,
                linestyle="--",
                label=threshold_name,
                linewidth=3,
            )

    if plot_legend:
        ax.legend(fontsize=20, loc="upper right")
