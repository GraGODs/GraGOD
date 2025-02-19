# %%
from gragod import start_research
import torch
import matplotlib.pyplot as plt
import numpy as np
from gragod.types import Datasets
from gragod.metrics.calculator import MetricsCalculator
from gragod.metrics.visualization import print_all_metrics, generate_metrics_table
from collections import defaultdict
from gragod.predictions.prediction import get_threshold_per_class
from pprint import pprint
import os
import pandas as pd
# %%
data_path = "test_gru_v6_telco_data.pt"
labels_path = "test_gru_v6_telco_labels.pt"
output_path = "test_gru_v6_telco_output.pt"
predictions_path = "test_gru_v6_telco_predictions.pt"
scores_path = "test_gru_v6_telco_scores.pt"

data = torch.load(data_path)
labels = torch.load(labels_path)
output = torch.load(output_path)
predictions = torch.load(predictions_path)
scores = torch.load(scores_path)

threshold = get_threshold_per_class(
    dataset=Datasets.TELCO,
    scores=scores,
    labels=labels,
    n_thresholds=100,
    range_based=True
)
# %%
calculator = MetricsCalculator(
    dataset=Datasets.TELCO,
    labels=labels,
    predictions=predictions,
    scores=scores
)

metrics = calculator.get_all_metrics(alpha=1.0)
print_all_metrics(metrics, "Metrics for all classes")
# %%
def plot_scores_and_predictions(scores, labels, predictions, data,output, start_index=0, end_index=-1):
    """
    Plots scores and predictions for a given column index.

    Parameters:
    - scores: numpy array of scores
    - labels: numpy array of labels
    - predictions: numpy array of predictions
    - data: numpy array of data
    - output: numpy array of output
    - start_index: index of the start of the plot
    - end_index: index of the end of the plot
    """

    # Create a figure and axis for real values and time series
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot real values
    ax.plot(data[start_index:end_index], label='Real Values', color='blue')

    # Plot time series
    ax.plot(output[start_index:end_index], label='Predicted Values', color='green')

    # Set title and labels for the plot
    ax.set_title(f"Real Values and Predicted Values")
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.grid(True)

    # Add legend
    ax.legend()

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()
    # Create a figure and axis for scores and data
    fig1, ax1 = plt.subplots(figsize=(12, 6))

    # Plot scores
    clipped_scores = np.clip(scores[start_index:end_index], None, 10)
    ax1.plot(clipped_scores, label='Scores', color='blue')

    # Plot data
    ax1.plot(labels[start_index:end_index]*10, label='Labels', color='green')

    # Set title and labels for the first plot
    ax1.set_title(f"Scores and Data")
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Value')
    ax1.grid(True)

    # Add legend
    ax1.legend()

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()

    # Create a figure and axis for predictions and labels
    fig2, ax2 = plt.subplots(figsize=(12, 6))

    # Plot predictions
    ax2.plot(predictions[start_index:end_index], label='Predictions', color='orange')

    # Plot labels
    ax2.plot(labels[start_index:end_index], label='Labels', color='red')

    # Set title and labels for the second plot
    ax2.set_title(f"Predictions and Labels")
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Value')
    ax2.grid(True)

    # Add legend
    ax2.legend()

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()

# %%
def plot_score_histogram(scores, labels, threshold, metrics=None):
    if metrics is not None:
        pprint(metrics)
    
    scores = scores.numpy()
    labels = labels.numpy()
    min_score = np.min(scores)
    if min_score < 0:
        scores = scores + np.abs(min_score) + 0.0001
        threshold = threshold + np.abs(min_score) + 0.0001
    # Separate scores based on labels
    anomalous_scores = scores[labels == 1]
    normal_scores = scores[labels == 0]

    # Define bin edges on a logarithmic scale
    min_score = min(np.min(normal_scores), np.min(anomalous_scores))
    max_score = max(np.max(normal_scores), np.max(anomalous_scores))
    bin_edges = np.logspace(np.log10(min_score), np.log10(max_score), num=30)
    
    # Plot histogram
    plt.figure(figsize=(8, 8))
    plt.hist(normal_scores, bins=bin_edges, alpha=0.7, label='normal', color='palegreen', log=True, edgecolor='black')
    plt.hist(anomalous_scores, bins=bin_edges, alpha=0.7, label='anomalous', color='salmon', log=True, edgecolor='black')

    # Add labels and legend
    plt.xlabel('Anomaly Score (log scale)', fontsize=12)
    plt.ylabel('Number of Samples (log scale)', fontsize=12)
    plt.title('Anomaly Score Distribution', fontsize=14)
    plt.legend(title='Ground Truth', fontsize=10)
    plt.xscale('log')
    plt.axvline(threshold, color='black', linestyle='--', label='Threshold')
    # Show plot
    plt.tight_layout()
    plt.show()

metrics_per_class = defaultdict(dict)
for metric_name, metric_value in metrics.items():
    if "per_class" in metric_name:
        for i in range(scores.shape[1]):
            metrics_per_class[i][metric_name.split("_per_class")[0]] = metric_value[i]

for i in range(scores.shape[1]):    
    plot_score_histogram(scores[:, i], labels[:, i], threshold[i], metrics_per_class[i])
    plot_scores_and_predictions(scores[:, i], labels[:, i], predictions[:, i], data[:, i], output[:, i], start_index=22800, end_index=22900)
    # break


# %%
def count_anomaly_ranges(labels: pd.DataFrame):
    """Count contiguous sequences of 1s in each time series column"""
    results = []

    # Convert tensor to numpy for easier manipulation
    labels_np = labels.to_numpy()
    if labels.ndim == 1:
        labels_np = labels_np.reshape(-1, 1)

    for col in range(labels_np.shape[1]):
        column_data = labels_np[:, col]

        # Find where the values change (0->1 or 1->0)
        diffs = np.diff(column_data, prepend=0, append=0)
        run_starts = np.where(diffs == 1)[0]
        run_ends = np.where(diffs == -1)[0]

        # Calculate lengths of each anomaly range
        lengths = run_ends - run_starts
        total_ranges = len(lengths)
        total_anomalies = np.sum(lengths)

        results.append(
            {
                "column": col,
                "total_ranges": total_ranges,
                "total_anomalies": total_anomalies,
                "range_lengths": lengths.tolist(),
                "start_times": run_starts.tolist(),
                "end_times": run_ends.tolist(),
            }
        )

    return results
def plot_score_histogram_with_ranges(scores, labels, threshold, metrics=None, range_size=10):
    if metrics is not None:
        pprint(metrics)
    
    scores = scores.numpy()
    labels = labels.numpy()
    min_score = np.min(scores)
    if min_score < 0:
        scores = scores + np.abs(min_score) + 0.0001
        threshold = threshold + np.abs(min_score) + 0.0001
    # Separate scores based on labels
    anomalous_scores = scores[labels == 1]
    normal_scores = scores[labels == 0]

    # Separate anomalous scores into ranges and keep the highest value of each range
    max_anomalous_scores = []
    anomaly_ranges = count_anomaly_ranges(pd.DataFrame(labels))

    for anomaly_range in anomaly_ranges:
        for start, end in zip(anomaly_range['start_times'], anomaly_range['end_times']):
            max_anomalous_scores.extend([np.max(scores[start:end])]*len(scores[start:end]))

    # Define bin edges on a logarithmic scale
    min_score = min(np.min(normal_scores), np.min(max_anomalous_scores))
    max_score = max(np.max(normal_scores), np.max(max_anomalous_scores))
    bin_edges = np.logspace(np.log10(min_score), np.log10(max_score), num=30)
    
    # Plot histogram
    plt.figure(figsize=(8, 8))
    plt.hist(normal_scores, bins=bin_edges, alpha=0.7, label='normal', color='palegreen', log=True, edgecolor='black')
    plt.hist(max_anomalous_scores, bins=bin_edges, alpha=0.7, label='anomalous', color='salmon', log=True, edgecolor='black')

    # Add labels and legend
    plt.xlabel('Anomaly Score (log scale)', fontsize=12)
    plt.ylabel('Number of Samples (log scale)', fontsize=12)
    plt.title('Anomaly Score Distribution with Ranges', fontsize=14)
    plt.legend(title='Ground Truth', fontsize=10)
    plt.xscale('log')
    plt.axvline(threshold, color='black', linestyle='--', label='Threshold')
    # Show plot
    plt.tight_layout()
    plt.show()

plot_score_histogram_with_ranges(scores[:, 11], labels[:, 11], threshold[11], metrics_per_class[11])

# %%
for i in range(scores.shape[1]):
    plot_score_histogram_with_ranges(scores[:, i], labels[:, i], threshold[i], metrics_per_class[i])
    plot_scores_and_predictions(scores[:, i], labels[:, i], predictions[:, i], data[:, i], output[:, i], start_index=22800, end_index=22900)
# %%
def plot_score_histograms_grid(scores, labels, thresholds, metrics_per_class, model_name = "GRU"):
    num_classes = scores.shape[1]
    num_rows = 4
    num_cols = 3
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 20))
    axes = axes.flatten()

    for i in range(num_classes):
        ax = axes[i]
        scores_class = scores[:, i].numpy()
        labels_class = labels[:, i].numpy()
        threshold_class = thresholds[i]

        min_score = np.min(scores_class)
        if min_score < 0:
            scores_class = scores_class + np.abs(min_score) + 0.0001
            threshold_class = threshold_class + np.abs(min_score) + 0.0001

        anomalous_scores = scores_class[labels_class == 1]
        normal_scores = scores_class[labels_class == 0]
        if anomalous_scores.size == 0:
            continue

        min_score = min(np.min(normal_scores), np.min(anomalous_scores))
        max_score = max(np.max(normal_scores), np.max(anomalous_scores))
        bin_edges = np.logspace(np.log10(min_score), np.log10(max_score), num=30)

        ax.hist(normal_scores, bins=bin_edges, alpha=0.7, label='normal', color='palegreen', log=True, edgecolor='black')
        ax.hist(anomalous_scores, bins=bin_edges, alpha=0.7, label='anomalous', color='salmon', log=True, edgecolor='black')

        ax.set_xlabel('Anomaly Score (log scale)', fontsize=10)
        ax.set_ylabel('Number of Samples (log scale)', fontsize=10)
        ax.set_title(f'TS{i+1} Anomaly Score Histogram in {model_name}', fontsize=12)
        ax.axvline(threshold_class, color='black', linestyle='--', label='Threshold')

        # Display metrics on the plot
        if metrics_per_class:
            metrics_text = "\n".join([f"{k}: {v:.2f}" for k, v in metrics_per_class[i].items()])
            ax.text(0.02, 0.77, metrics_text, transform=ax.transAxes, fontsize=8,
                    verticalalignment='top', horizontalalignment='left', bbox=dict(facecolor='white', alpha=0.5))

        ax.legend(title='Ground Truth', fontsize=8)
        ax.set_xscale('log')

    plt.tight_layout()
    plt.show()
    return fig

# Call the function to plot the grid
plot_score_histograms_grid(scores, labels, threshold, metrics_per_class, model_name="GRU")

# %%
def plot_score_histograms_grid_with_ranges(scores, labels, thresholds, metrics_per_class, model_name = "GRU"):
    num_classes = scores.shape[1]
    num_rows = 4
    num_cols = 3
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 20))
    axes = axes.flatten()

    for i in range(num_classes):
        ax = axes[i]
        scores_class = scores[:, i].numpy()
        labels_class = labels[:, i].numpy()
        threshold_class = thresholds[i]

        min_score = np.min(scores_class)
        if min_score < 0:
            scores_class = scores_class + np.abs(min_score) + 0.0001
            threshold_class = threshold_class + np.abs(min_score) + 0.0001

        anomalous_scores = scores_class[labels_class == 1]
        normal_scores = scores_class[labels_class == 0]
        if anomalous_scores.size == 0:
            continue
        # Separate anomalous scores into ranges and keep the highest value of each range
        max_anomalous_scores = []
        anomaly_ranges = count_anomaly_ranges(pd.DataFrame(labels_class))

        for anomaly_range in anomaly_ranges:
            for start, end in zip(anomaly_range['start_times'], anomaly_range['end_times']):
                max_anomalous_scores.extend([np.max(scores_class[start:end])] * len(scores_class[start:end]))

        min_score = min(np.min(normal_scores), np.min(max_anomalous_scores))
        max_score = max(np.max(normal_scores), np.max(max_anomalous_scores))
        bin_edges = np.logspace(np.log10(min_score), np.log10(max_score), num=30)

        ax.hist(normal_scores, bins=bin_edges, alpha=0.7, label='normal', color='palegreen', log=True, edgecolor='black')
        ax.hist(max_anomalous_scores, bins=bin_edges, alpha=0.7, label='anomalous', color='salmon', log=True, edgecolor='black')

        ax.set_xlabel('Anomaly Score (log scale)', fontsize=10)
        ax.set_ylabel('Number of Samples (log scale)', fontsize=10)
        ax.set_title(f'TS{i+1} Anomaly Score Ranged Histogram in {model_name}', fontsize=12)
        ax.axvline(threshold_class, color='black', linestyle='--', label='Threshold')

        # Display metrics on the plot
        if metrics_per_class:
            metrics_text = "\n".join([f"{k}: {v:.2f}" for k, v in metrics_per_class[i].items()])
            ax.text(0.02, 0.77, metrics_text, transform=ax.transAxes, fontsize=8,
                    verticalalignment='top', horizontalalignment='left', bbox=dict(facecolor='white', alpha=0.5))

        ax.legend(title='Ground Truth', fontsize=8)
        ax.set_xscale('log')

    # Hide any unused axes
    for j in range(num_classes, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()
    return fig

# plot_score_histograms_grid_with_ranges(scores, labels, threshold, metrics_per_class)
# %%
def print_telco_metrics(base_path, model_name, save_path=None):
    for split in ["test", "train", "val"]:
        data_path = os.path.join(base_path, f"{split}_{model_name.lower()}_v6_telco_data.pt")
        labels_path = os.path.join(base_path, f"{split}_{model_name.lower()}_v6_telco_labels.pt")
        output_path = os.path.join(base_path, f"{split}_{model_name.lower()}_v6_telco_output.pt")
        predictions_path = os.path.join(base_path, f"{split}_{model_name.lower()}_v6_telco_predictions.pt")
        scores_path = os.path.join(base_path, f"{split}_{model_name.lower()}_v6_telco_scores.pt")

        data = torch.load(data_path)
        labels = torch.load(labels_path)
        output = torch.load(output_path)
        predictions = torch.load(predictions_path)
        scores = torch.load(scores_path)
        
        calculator = MetricsCalculator(
            dataset=Datasets.TELCO,
            labels=labels,
            predictions=predictions,
            scores=scores
        )
        metrics = calculator.get_all_metrics(alpha=1.0)
        metrics_per_class = defaultdict(dict)
        for metric_name, metric_value in metrics.items():
            if "per_class" in metric_name:
                for i in range(scores.shape[1]):
                    metrics_per_class[i][metric_name.split("_per_class")[0]] = metric_value[i]
        fig_grid = plot_score_histograms_grid(scores, labels, threshold, metrics_per_class, model_name=model_name)
        fig_grid_with_ranges = plot_score_histograms_grid_with_ranges(scores, labels, threshold, metrics_per_class, model_name=model_name)
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            fig_grid.savefig(os.path.join(save_path, f"{split}_{model_name.lower()}_v6_telco_score_histogram.png"))
            fig_grid_with_ranges.savefig(os.path.join(save_path, f"{split}_{model_name.lower()}_v6_telco_score_histogram_with_ranges.png"))

print_telco_metrics(".", "GRU", "scores_histograms/gru")

# %%
