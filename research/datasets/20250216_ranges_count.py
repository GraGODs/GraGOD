# %%
from gragod import start_research
import pandas as pd
import numpy as np
from datasets.swat import load_swat_df
from datasets.telco import load_telco_df
from matplotlib import pyplot as plt
from datasets.data_processing import preprocess_df
import seaborn as sns
from IPython.display import display
import torch

import matplotlib.pyplot as plt

# %%
(
    df_train_swat,
    df_train_swat_labels,
    df_val_swat,
    df_val_swat_labels,
    df_test_swat,
    df_test_swat_labels,
) = load_swat_df()
df_train_swat.drop(columns=[" Timestamp"], inplace=True)
df_val_swat.drop(columns=[" Timestamp"], inplace=True)
df_test_swat.drop(columns=[" Timestamp"], inplace=True)
six_hours_in_seconds = 6 * 60 * 60

df_train_swat = df_train_swat.iloc[six_hours_in_seconds:]
df_train_swat_labels = df_train_swat_labels.iloc[six_hours_in_seconds:]
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

# Fill NaN values in the dataframes with 0
df_train_telco.fillna(0, inplace=True)
df_train_telco_labels.fillna(0, inplace=True)
df_val_telco.fillna(0, inplace=True)
df_val_telco_labels.fillna(0, inplace=True)
df_test_telco.fillna(0, inplace=True)
df_test_telco_labels.fillna(0, inplace=True)


# # %%
# def count_up_flanks(data: pd.Series, column: int | None = None):
#     up_flanks = 0
#     if column is None:
#         for i in range(1, len(data)):
#             if (data.iloc[i].values[0] == 1) and (data.iloc[i - 1].values[0] == 0):
#                 up_flanks += 1
#     else:
#         for i in range(1, len(data)):
#             if (data.iloc[i].values[column] == 1) and (
#                 data.iloc[i - 1].values[column] == 0
#             ):
#                 up_flanks += 1
#     return up_flanks


# # %%
# flanks_swat_val = count_up_flanks(df_val_swat_labels)
# flanks_swat_test = count_up_flanks(df_test_swat_labels)

# print(f"SWAT VAL: {flanks_swat_val}")
# print(f"SWAT TEST: {flanks_swat_test}")
# # %%
# for i in range(df_train_telco_labels.shape[1]):
#     flanks_telco_train = count_up_flanks(df_train_telco_labels, i)
#     flanks_telco_val = count_up_flanks(df_val_telco_labels, i)
#     flanks_telco_test = count_up_flanks(df_test_telco_labels, i)
#     print(f"TELCO TRAIN {i}: {flanks_telco_train}")
#     print(f"TELCO VAL {i}: {flanks_telco_val}")
#     print(f"TELCO TEST {i}: {flanks_telco_test}")


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


def print_anomaly_stats(anomaly_stats: list[dict]):
    # Print statistics
    for stat in anomaly_stats:
        if stat["total_ranges"] > 0:
            print(f"Column {stat['column']}:")
            print(f"  Total anomaly ranges: {stat['total_ranges']}")
            print(f"  Total anomaly points: {stat['total_anomalies']}")
            print(f"  Range lengths: {stat['range_lengths']}")
            print(f"  Start times: {stat['start_times']}")
            print(f"  End times: {stat['end_times']}")
        s = 0
        for i in stat["range_lengths"]:
            s += i
        assert s == stat["total_anomalies"]


def generate_ranged_plots(anomaly_stats: list[dict]):
    # Filter out stats with no anomalies
    anomaly_stats = [stat for stat in anomaly_stats if stat["total_ranges"] > 0]

    # Determine the number of subplots needed
    num_plots = len(anomaly_stats)
    if num_plots == 0:
        print("No anomalies to plot.")
        return
    elif num_plots == 1:
        fig, ax = plt.subplots(figsize=(10, 5))  # Single plot
        axs = [ax]
    else:
        fig, axs = plt.subplots(4, 3, figsize=(20, 20))
        axs = axs.flatten()  # Flatten to 1D array for easier iteration

    # Plot for each time series in subplots
    for idx, stat in enumerate(anomaly_stats):
        if idx >= len(axs):  # Only plot up to 12 time series (4x3 grid)
            break
        ax = axs[idx]

        range_lengths = stat["range_lengths"]
        total_anomalies = stat["total_anomalies"]
        ranges = list(range(len(range_lengths)))

        # Find maximum range
        max_idx = np.argmax(range_lengths)
        max_length = range_lengths[max_idx]

        # Plot on the subplot axis
        bars = ax.bar(
            ranges,
            range_lengths,
            color=[
                "red" if (i == max_idx and False) else "skyblue"
                for i in range(len(range_lengths))
            ],
        )

        ax.set_title(f"TS {stat['column']} - {total_anomalies} anomalies", fontsize=10)
        ax.set_xlabel("Range Index", fontsize=8)
        ax.set_ylabel("Range Length", fontsize=8)
        ax.grid(axis="y", alpha=0.4)
        ax.tick_params(axis="both", which="major", labelsize=8)

    plt.tight_layout()
    plt.show()


def run_pipeline(labels: pd.DataFrame):
    anomaly_stats = count_anomaly_ranges(labels)
    # print_anomaly_stats(anomaly_stats)
    generate_ranged_plots(anomaly_stats)


run_pipeline(df_train_telco_labels)
run_pipeline(df_val_telco_labels)
run_pipeline(df_test_telco_labels)

run_pipeline(df_val_swat_labels)
run_pipeline(df_test_swat_labels)


# %%
# Create combined dataframe of all ranges with TS IDs
range_df = pd.DataFrame(
    [
        {"ts_id": stat["column"], "range_length": length}
        for stat in anomaly_stats
        for length in stat["range_lengths"]
    ]
)
# %%
anomaly_stats[0]["range_lengths"]

# %%


# %%
swat_anomaly_stats = count_anomaly_ranges(df_test_swat_labels)
# %%
for stat in swat_anomaly_stats:
    print(f"Column {stat['column']}:")
    print(f"  Total anomaly ranges: {stat['total_ranges']}")
    print(f"  Total anomaly points: {stat['total_anomalies']}")
    print(f"  Range lengths: {stat['range_lengths']}")
    print(f"  Start times: {stat['start_times']}")
    print(f"  End times: {stat['end_times']}")
# %%
