# %%
from gragod import start_research
import pandas as pd
import numpy as np
from datasets.telco import load_telco_df
from matplotlib import pyplot as plt
from datasets.data_processing import preprocess_df
import seaborn as sns
from models.predict import predict
from gragod.types import Models, Datasets

from gragod.training import load_params
from gragod.types import ParamFileTypes
import torch

# %%
params = load_params("models/gru/params_telco.yaml", ParamFileTypes.YAML)
results_v0 = predict(
    model=Models.GRU,
    dataset=Datasets.TELCO,
    ckpt_path="benchmarks_telco/gru/version_6/best.ckpt",
    params_file="models/gru/params_telco.yaml",
    params=params,
    model_params=params["model_params"],
    # **params["predictor_params"],
    **params["train_params"],
)
train_results_v0 = results_v0["train"]
val_results_v0 = results_v0["val"]
test_results_v0 = results_v0["test"]
# %%
# params["predictor_params"]["max_std"] = 0.0
# results_v1 = predict(
#     model=Models.GRU,
#     dataset=Datasets.SWAT,
#     ckpt_path="gru_swat/gru_swat/best.ckpt",
#     params_file="models/gru/params_swat.yaml",
#     params=params,
#     model_params=params["model_params"],
#     **params["predictor_params"],
# )
# %%
# train_results_v1 = results_v1["train"]
# val_results_v1 = results_v1["val"]
# test_results_v1 = results_v1["test"]
# %%
# Print data and predictions for v1
column = 5
start_time = 0
end_time = 5000
start_time, end_time = 0, -1


def plot_results(results, start_time, end_time, column):
    plt.figure(figsize=(15, 5))
    plt.plot(results["labels"][start_time:end_time, column], label="True Labels")
    # plt.plot(results["predictions"][start_time:end_time, column], label="Predictions")
    plt.title(f"Train Results for Column {column}")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot train results
    plt.figure(figsize=(15, 5))
    # plt.plot(results["labels"][start_time:end_time, column], label="True Labels")
    plt.plot(results["data"][start_time:end_time, column], label="Data")
    plt.plot(results["output"][start_time:end_time, column], label="Forecasts")
    plt.title(f"Train Results for Column {column}")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # score = torch.max(results["scores"], dim=1).values
    score = results["scores"][start_time:end_time, column]
    score = torch.clip(score, 0, 10)
    plt.figure(figsize=(15, 5))

    # Plot score on the primary y-axis
    plt.plot(score[start_time:end_time], label="Score", color="b")
    plt.xlabel("Time")
    plt.ylabel("Score", color="b")
    plt.title(f"Score for Column {column}")
    plt.grid(True)

    # Create a secondary y-axis for the labels
    ax2 = plt.gca().twinx()
    ax2.plot(
        results["labels"][start_time:end_time, column], label="True Labels", color="r"
    )
    ax2.set_ylabel("True Labels", color="r")
    plt.axhline(
        y=results["thresholds"][column], color="g", linestyle="--", label="Threshold"
    )

    # Add legends for both y-axes
    plt.legend(loc="upper right")
    ax2.legend(loc="upper left")

    plt.tight_layout()
    plt.show()


# start_time, end_time = 6000, -1
plot_results(train_results_v0, start_time, end_time, column)
# plot_results(val_results_v1, start_time, end_time, column)
# plot_results(test_results_v1, start_time, end_time, column)

# %%
test_results_v0.keys()
# %%
test_results_v0["thresholds"]
# %%
test_results_v1["thresholds"]
# %%
test_results_v1["metrics"]
# %%
test_results_v0["metrics"]
# %%
scores_v1 = test_results_v1["scores"]
score_v1 = torch.max(scores_v1, dim=1).values
# %%
# Print columns with means greater than 100
mean_scores = scores_v1.mean(dim=0)
columns_above_threshold = torch.where(mean_scores > 100)[0]

print("Columns with means greater than 100:")
for col in columns_above_threshold:
    print(f"Column {col.item()}: Mean = {mean_scores[col].item()}")

# %%
from gragod.metrics.calculator import MetricsCalculator


def calculate_metrics(labels, predictions, scores):
    metrics = MetricsCalculator(
        dataset=Datasets.TELCO,
        labels=labels,
        predictions=predictions,
        scores=scores,
    )

    precision = metrics.calculate_precision()
    recall = metrics.calculate_recall()
    f1 = metrics.calculate_f1(precision, recall)
    range_based_precision = metrics.calculate_range_based_precision(alpha=0.0)
    range_based_recall = metrics.calculate_range_based_recall(alpha=1.0)
    range_based_f1 = metrics.calculate_range_based_f1(
        range_based_precision, range_based_recall
    )

    custom_f1 = metrics.calculate_f1(precision, range_based_recall)

    print(f"Precision: {precision.metric_mean}")
    print(f"Recall: {recall.metric_mean}")
    print(f"F1: {f1.metric_mean}")
    print(f"Range-based Precision: {range_based_precision.metric_mean}")
    print(f"Range-based Recall: {range_based_recall.metric_mean}")
    print(f"Range-based F1: {range_based_f1.metric_mean}")
    print(f"Custom F1: {custom_f1.metric_mean}")


print(f"ALL PREDICTIONS:")
calculate_metrics(
    test_results_v0["labels"], test_results_v0["predictions"], test_results_v0["scores"]
)
print(f"ONE PREDICTION:")
calculate_metrics(
    test_results_v0["labels"],
    torch.ones_like(test_results_v0["labels"]),
    test_results_v0["scores"],
)
print(f"ZERO PREDICTION:")
calculate_metrics(
    test_results_v0["labels"],
    torch.zeros_like(test_results_v0["labels"]),
    test_results_v0["scores"],
)


# %%
def count_anomaly_ranges(labels):
    """Count contiguous sequences of 1s in each time series column"""
    results = []

    # Convert tensor to numpy for easier manipulation
    labels_np = labels.cpu().numpy()

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


# Usage
anomaly_stats = count_anomaly_ranges(train_results_v0["labels"])

# Print statistics
for stat in anomaly_stats:
    if stat["total_ranges"] > 0:
        print(f"Column {stat['column']}:")
        print(f"  Total anomaly ranges: {stat['total_ranges']}")
        print(f"  Total anomaly points: {stat['total_anomalies']}")
        print(f"  Range lengths: {stat['range_lengths']}")
        print(f"  Start times: {stat['start_times']}")
        print(f"  End times: {stat['end_times']}")

for stat in anomaly_stats:
    s = 0
    for i in stat["range_lengths"]:
        s += i
    assert s == stat["total_anomalies"]


# %%
# Save train, test, and validation data, labels, predictions, outputs, and scores
def save_results(results, split_name):
    torch.save(results["data"], f"{split_name}_data.pt")
    torch.save(results["labels"], f"{split_name}_labels.pt")
    torch.save(results["predictions"], f"{split_name}_predictions.pt")
    torch.save(results["output"], f"{split_name}_output.pt")
    torch.save(results["scores"], f"{split_name}_scores.pt")


# Save train results
save_results(train_results_v0, "train_gru_v6_telco")

# Save validation results
save_results(val_results_v0, "val_gru_v6_telco")

# Save test results
save_results(test_results_v0, "test_gru_v6_telco")

# %%
(
    df_train_telco,
    df_train_telco_labels,
    df_val_telco,
    df_val_telco_labels,
    df_test_telco,
    df_test_telco_labels,
) = load_telco_df()
df_train_telco.drop(columns=["time"], inplace=True)
df_val_telco.drop(columns=["time"], inplace=True)
df_test_telco.drop(columns=["time"], inplace=True)
df_train_telco_labels.drop(columns=["time"], inplace=True)
df_val_telco_labels.drop(columns=["time"], inplace=True)
df_test_telco_labels.drop(columns=["time"], inplace=True)
# %%
# Plot results for column 1
column = 5


def plot_time_series(df: pd.DataFrame, labels: pd.DataFrame, column: int):
    start_time, end_time = 0, 20
    plt.figure(figsize=(15, 5))

    # Create a secondary y-axis for the labels
    ax2 = plt.gca().twinx()
    ax2.plot(labels.iloc[start_time:end_time, column], label="Labels", color="orange")
    ax2.set_ylabel("Labels")

    # Add legends for both y-axes
    plt.legend(loc="upper left")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.show()


plot_time_series(df_train_telco, df_train_telco_labels, column)
# %%
