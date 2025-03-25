# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from IPython.display import display
from matplotlib import pyplot as plt

from datasets.data_processing import preprocess_df
from datasets.swat import load_swat_df
from datasets.telco import load_telco_df
from gragod import start_research

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
        fig, axs = plt.subplots(4, 3, figsize=(20, 15))
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

        if num_plots == 1:
            ax.set_xlabel("Range Index", fontsize=16)
            ax.set_ylabel("Range Length", fontsize=16)
        else:
            ax.set_title(f"TS {stat['column'] + 1}", fontsize=20)
            ax.set_xlabel("Range Index", fontsize=20)
            ax.set_ylabel("Range Length", fontsize=20)
        ax.grid(axis="y", alpha=0.4)
        ax.tick_params(axis="both", which="major", labelsize=10)

    plt.tight_layout()
    plt.show()

    return fig


def generate_timeline_plots(anomaly_stats: list[dict], total_length: int):
    """Plot horizontal timeline bars showing normal/anomaly regions."""
    # Filter out stats with no anomalies
    anomaly_stats = [stat for stat in anomaly_stats if stat["total_ranges"] > 0]
    font_size = 20
    # Create figure with larger size and better aspect ratio
    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(20, max(8, len(anomaly_stats) * 0.8)))

    # Set background color
    ax.set_facecolor("white")
    fig.set_facecolor("white")

    for idx, stat in enumerate(anomaly_stats):
        y_pos = idx
        ranges = sorted(zip(stat["start_times"], stat["end_times"]), key=lambda x: x[0])

        # Calculate regions
        current_pos = 0
        normal_regions = []
        anomaly_regions = []

        for start, end in ranges:
            if start > current_pos:
                normal_regions.append((current_pos, start - current_pos))
            anomaly_regions.append((start, end - start))
            current_pos = end

        if current_pos < total_length:
            normal_regions.append((current_pos, total_length - current_pos))

        # Plot normal regions (green)
        if normal_regions:
            ax.broken_barh(
                normal_regions,
                (y_pos - 0.35, 0.7),
                facecolors="#2ecc71",
                edgecolor="none",
                alpha=0.6,
            )

        # Plot anomaly regions (bright red)
        if anomaly_regions:
            ax.broken_barh(
                anomaly_regions,
                (y_pos - 0.35, 0.7),
                facecolors="#ff0000",
                edgecolor="none",
                alpha=1.0,
            )

    # Improved appearance configuration with larger fonts
    ax.set_yticks(range(len(anomaly_stats)))
    ax.set_yticklabels(
        [f"TS{stat['column']+1}" for stat in anomaly_stats], fontsize=font_size
    )

    # Add spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)

    # Much larger labels and title
    ax.set_xlabel(
        "Time Index", fontsize=font_size, fontweight="bold"
    )  # Increased to 40

    # Customize grid
    ax.grid(True, axis="x", alpha=0.2, linestyle="--")
    ax.set_axisbelow(True)

    # Add legend with larger font
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#2ecc71", alpha=0.6, label="Normal"),
        Patch(facecolor="#ff0000", alpha=1.0, label="Anomaly"),
    ]

    # Much larger x-axis tick label size
    ax.tick_params(axis="x", labelsize=font_size)  # Increased to 36

    # Add more space at the bottom for the larger labels
    plt.subplots_adjust(
        bottom=0.2
    )  # Increased bottom margin to accommodate larger labels

    return fig


def generate_timeline_plots_combined(
    anomaly_stats_list: list[list[dict]], total_length: int, labels: list[str]
):
    """Plot horizontal timeline bars showing normal/anomaly regions for multiple sets."""
    # Combine all stats, adding an offset to the column numbers for the second set
    combined_stats = []
    offset = 0
    for stats in anomaly_stats_list:
        for stat in stats:
            if stat["total_ranges"] > 0:  # Only include stats with anomalies
                new_stat = stat.copy()
                new_stat["column"] = stat["column"] + offset
                new_stat["set_index"] = offset  # To track which set it's from
                combined_stats.append(new_stat)
        offset += 100  # Large offset to separate the sets

    font_size = 20
    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(20, max(8, len(combined_stats) * 0.8)))

    ax.set_facecolor("white")
    fig.set_facecolor("white")

    for idx, stat in enumerate(combined_stats):
        y_pos = idx
        ranges = sorted(zip(stat["start_times"], stat["end_times"]), key=lambda x: x[0])

        # Calculate regions
        current_pos = 0
        normal_regions = []
        anomaly_regions = []

        for start, end in ranges:
            if start > current_pos:
                normal_regions.append((current_pos, start - current_pos))
            anomaly_regions.append((start, end - start))
            current_pos = end

        if current_pos < total_length:
            normal_regions.append((current_pos, total_length - current_pos))

        # Plot normal regions (green)
        if normal_regions:
            ax.broken_barh(
                normal_regions,
                (y_pos - 0.35, 0.7),
                facecolors="#2ecc71",
                edgecolor="none",
                alpha=0.6,
            )

        # Plot anomaly regions (bright red)
        if anomaly_regions:
            ax.broken_barh(
                anomaly_regions,
                (y_pos - 0.35, 0.7),
                facecolors="#ff0000",
                edgecolor="none",
                alpha=1.0,
            )

    # Improved appearance configuration with larger fonts
    ax.set_yticks(range(len(combined_stats)))
    # Create labels with set information
    y_labels = []
    for stat in combined_stats:
        set_name = labels[
            stat["set_index"] // 100
        ]  # Use integer division to get set index
        y_labels.append(set_name)  # Use modulo to get original column number

    ax.set_yticklabels(y_labels, fontsize=font_size)

    # Add spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)

    ax.set_xlabel("Time Index", fontsize=font_size, fontweight="bold")

    # Customize grid
    ax.grid(True, axis="x", alpha=0.2, linestyle="--")
    ax.set_axisbelow(True)

    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#2ecc71", alpha=0.6, label="Normal"),
        Patch(facecolor="#ff0000", alpha=1.0, label="Anomaly"),
    ]

    ax.tick_params(axis="x", labelsize=font_size)

    plt.subplots_adjust(bottom=0.2)

    return fig


import os
import shutil

if os.path.exists("anomalies_distribution"):
    shutil.rmtree("anomalies_distribution")
os.makedirs("anomalies_distribution")


def run_pipeline_telco(labels: pd.DataFrame, dataset: str, split: str):
    anomaly_stats = count_anomaly_ranges(labels)
    fig = generate_timeline_plots(anomaly_stats, len(labels))
    display(fig)
    fig.savefig(
        f"anomalies_distribution/telco_labels_temporal_distribution_{split}.pdf"
    )
    plt.close("all")


def run_pipeline_swat_combined(val_labels: pd.DataFrame, test_labels: pd.DataFrame):
    val_stats = count_anomaly_ranges(val_labels)
    test_stats = count_anomaly_ranges(test_labels)

    fig = generate_timeline_plots_combined(
        [val_stats, test_stats],
        len(val_labels),  # Assuming val and test have same length
        labels=["Val", "Test"],
    )
    display(fig)
    fig.savefig(
        f"anomalies_distribution/swat_labels_temporal_distribution_combined.pdf"
    )
    plt.close("all")


# Run the pipelines
run_pipeline_telco(df_train_telco_labels, "telco", "train")
run_pipeline_telco(df_val_telco_labels, "telco", "val")
run_pipeline_telco(df_test_telco_labels, "telco", "test")

# Combined SWAT plot
run_pipeline_swat_combined(df_val_swat_labels, df_test_swat_labels)

# %%

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
