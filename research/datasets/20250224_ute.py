# %%
import torch
from gragod import start_research
from datasets.ute import load_ute_df, load_ute_training_data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import contextily as ctx
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import ConnectionPatch


# %%
df_train, *_ = load_ute_df()
columns = df_train.columns
X_train, X_val, X_test, X_train_labels, X_val_labels, X_test_labels = (
    load_ute_training_data(normalize=True)
)


# %%
# Convert X_test_labels to numpy array if it's a tensor
def plot_anomalies(counts: np.ndarray, name: str):
    plt.figure(figsize=(14, 10))
    plt.bar(range(len(counts)), counts, color="skyblue", edgecolor="black")
    plt.xlabel("Time Series Index", fontsize=32, labelpad=20)
    plt.ylabel("Number of Anomalies", fontsize=32)
    plt.tick_params(axis="both", which="major", labelsize=20)
    plt.xticks(size=32)
    plt.yticks(size=32)
    plt.yscale("log")
    plt.grid(True, alpha=0.7, axis="y")
    plt.tight_layout()
    # Adjust the margins to decrease space between plot box and bars
    plt.margins(x=0.02)  # Reduce horizontal margins
    plt.savefig(f"research/datasets/plots/{name}.pdf", bbox_inches="tight", dpi=1000)
    plt.show()


amount_of_anomalies = np.sum(X_test_labels.numpy() == 1, axis=0)
plot_anomalies(amount_of_anomalies, name="point-anomalies-count")


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


range_counts = count_anomaly_ranges(pd.DataFrame(X_test_labels.numpy()))
range_counts = [r["total_ranges"] for r in range_counts]

plot_anomalies(np.array(range_counts), name="range-anomalies-count")


# %%
# %%
def plot_system_anomalies(labels: pd.DataFrame, name: str):
    plt.figure(figsize=(14, 10))
    plt.plot(np.max(labels.to_numpy(), axis=1), label="System Anomalies")
    plt.xlabel("Time Step", fontsize=24)
    plt.ylabel("Value", fontsize=24)
    plt.legend(fontsize=24)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.tick_params(axis="both", which="major", labelsize=20)
    # Save figure with bbox_inches='tight' to ensure the entire figure is saved
    # plt.savefig(f"research/datasets/plots/{name}.pdf", bbox_inches="tight", dpi=1000)
    plt.show()


plot_system_anomalies(pd.DataFrame(X_test_labels.numpy()), name="system-anomalies")


# %%
# Visualize raw time series with anomaly highlights
def plot_time_series_with_anomalies(
    X, labels, feature_indices=[0, 1, 2, 3], start_idx=0, end_idx=-1
):
    # Convert data to numpy if it's a tensor
    X_np = X.numpy() if isinstance(X, torch.Tensor) else X
    labels_np = labels.numpy() if isinstance(labels, torch.Tensor) else labels
    if end_idx == -1:
        end_idx = X_np.shape[0]

    # Create individual plots for each feature
    for i, feature_idx in enumerate(feature_indices):
        feature_name = columns[feature_idx]

        # Create a new figure for each feature
        plt.figure(figsize=(14, 10))

        # Get values and labels for this feature
        values = X_np[start_idx:end_idx, feature_idx]
        anomaly_mask = labels_np[start_idx:end_idx, feature_idx] == 1

        # Plot the time series
        plt.plot(
            range(start_idx, end_idx), values, color="blue", alpha=0.7, label="Data"
        )

        # Highlight anomalies
        if np.any(anomaly_mask):
            # plt.scatter(
            #     np.where(anomaly_mask)[0] + start_idx,
            #     values[anomaly_mask],
            #     color="red",
            #     alpha=0.7,
            #     label="Anomaly",
            # )

            # Shade anomaly regions for better visibility
            idx_ranges = []
            anomaly_indices = np.where(anomaly_mask)[0]
            for k, g in itertools.groupby(
                enumerate(anomaly_indices), lambda x: x[0] - x[1]
            ):
                group = list(map(lambda x: x[1], g))
                idx_ranges.append((group[0], group[-1]))

            # for start, end in idx_ranges:
            #     plt.axvspan(start + start_idx, end + start_idx, color="red", alpha=0.2)

        print(f"Time series: {feature_name}")
        plt.xlabel("Time Step", fontsize=24)
        plt.ylabel("Value", fontsize=24)
        # plt.axvspan(start_idx, start_idx, color="red", alpha=0.2, label="Anomaly")
        plt.legend(loc="upper right", fontsize=22)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.tick_params(axis="both", which="major", labelsize=22)
        # save as pdf
        plt.savefig(
            f"research/datasets/plots/{feature_name}-station.pdf",
            bbox_inches="tight",
            dpi=1000,
        )
        plt.show()


features_to_process = {
    "RIV": (0, -1),
    "MVJ": (0, -1),
}
for feature, (start_idx, end_idx) in features_to_process.items():
    plot_time_series_with_anomalies(
        X_train,
        X_train_labels,
        # feature_indices=range(X_train.shape[1]),
        feature_indices=[columns.get_loc(feature)],
        start_idx=start_idx,
        end_idx=end_idx,
    )


# %%


# %%
# %%
# %%
# %%


# ute_stations = pd.read_csv("datasets_files/ute/estaciones_ute.csv")
# ute_stations.head()
# all_data = pd.read_csv("datasets_files/ute/train_val_data_full.csv")
# real_columns = all_data.columns
# # %%

# # ute stations have LAT and LON, we can use this to plot the stations on a map

# # Define Montevideo area coordinates
# montevideo_bounds = {
#     "min_lon": -56.4,  # Include parts of San José
#     "max_lon": -56,  # Include parts of Canelones
#     "min_lat": -34.95,  # Southern boundary
#     "max_lat": -34.75,  # Northern boundary
# }

# # Filter stations to only include those whose CODIGO is in real_columns
# valid_stations = ute_stations[ute_stations["CODIGO"].isin(real_columns)]

# # Identify which stations are in Montevideo area
# in_montevideo = (
#     (valid_stations["LON"] >= montevideo_bounds["min_lon"])
#     & (valid_stations["LON"] <= montevideo_bounds["max_lon"])
#     & (valid_stations["LAT"] >= montevideo_bounds["min_lat"])
#     & (valid_stations["LAT"] <= montevideo_bounds["max_lat"])
# )

# # FIGURE: URUGUAY MAP WITH MONTEVIDEO INSET
# fig = plt.figure(figsize=(14, 14))

# # MAIN MAP: FULL URUGUAY
# # Create main axes for Uruguay
# ax_main = plt.gca()
# ax_main.set_axis_off()

# # Create a scatter plot of all valid stations
# scatter = plt.scatter(
#     valid_stations["LON"],
#     valid_stations["LAT"],
#     color="black",
#     s=150,  # Larger markers
#     alpha=0.9,
#     edgecolor="white",
#     linewidth=1.5,
#     zorder=3,
# )

# # Set map extent to include all of Uruguay
# ax_main.set_xlim(-58.5, -53.5)  # Wide longitude range
# ax_main.set_ylim(-35.5, -30)  # Wide latitude range

# # Sort stations by latitude
# outside_montevideo = valid_stations[~in_montevideo].sort_values(
#     by="LAT", ascending=False
# )

# # Create a scatter plot just for non-Montevideo stations to ensure they appear
# plt.scatter(
#     outside_montevideo["LON"],
#     outside_montevideo["LAT"],
#     color="black",
#     s=150,  # Larger markers
#     alpha=0.9,
#     edgecolor="white",
#     linewidth=1.5,
#     zorder=3,
# )

# # Add basemap for main map
# ctx.add_basemap(
#     ax_main,
#     crs="EPSG:4326",
#     source="OpenStreetMap.Mapnik",
#     attribution=False,
#     zoom=10,
# )

# # Highlight Montevideo area with a rectangle
# rect = Rectangle(
#     (montevideo_bounds["min_lon"], montevideo_bounds["min_lat"]),
#     montevideo_bounds["max_lon"] - montevideo_bounds["min_lon"],
#     montevideo_bounds["max_lat"] - montevideo_bounds["min_lat"],
#     linewidth=2,
#     edgecolor="red",
#     facecolor="none",
#     zorder=4,
# )
# ax_main.add_patch(rect)

# # INSET MAP: MONTEVIDEO AREA
# # Create the inset map over water area in the main map
# # Position in lower-left corner (over Río de la Plata)

# # Use a different method to create the inset - move lower and to the right
# axins = fig.add_axes(
#     (0.2, 0.1, 0.15, 0.15)
# )  # [left, bottom, width, height] in figure coordinates - moved right from 0.15 to 0.25
# axins.set_axis_off()

# # Get only Montevideo stations
# montevideo_stations = valid_stations[in_montevideo]

# # Create a scatter plot of Montevideo stations in the inset
# axins.scatter(
#     montevideo_stations["LON"],
#     montevideo_stations["LAT"],
#     color="black",
#     s=100,  # Slightly smaller markers for inset
#     alpha=0.9,
#     edgecolor="white",
#     linewidth=1.5,
#     zorder=3,
# )

# # Set inset extent to Montevideo area with slight padding
# axins.set_xlim(montevideo_bounds["min_lon"], montevideo_bounds["max_lon"])
# axins.set_ylim(montevideo_bounds["min_lat"], montevideo_bounds["max_lat"])
# padding = 0.005
# # Add basemap to inset
# ctx.add_basemap(
#     axins,
#     crs="EPSG:4326",
#     source="OpenStreetMap.Mapnik",
#     attribution=False,
#     zoom=12,
# )

# # Add a border and background to the inset
# axins.set_facecolor("white")  # White background for clarity
# axins.patch.set_alpha(0.7)  # Slightly transparent

# # Make the existing spines visible
# for spine in axins.spines.values():
#     spine.set_visible(True)
#     spine.set_color("black")
#     spine.set_linewidth(0.75)  # Thinner line for the spines

# # Add a thin square box around the inset
# inset_box = Rectangle(
#     (0, 0),  # Lower-left corner in axes coordinates
#     1,
#     1,  # Width and height covering the full axes
#     transform=axins.transAxes,  # Use axes coordinates
#     fill=False,  # No fill, just the edge
#     edgecolor="black",
#     linewidth=1.5,  # Slightly thicker than the spines
#     zorder=5,  # Make sure it's on top
# )
# axins.add_patch(inset_box)

# # Connect the inset to the Montevideo area with lines
# # Get inset position in data coordinates
# inset_pos = axins.get_position()
# fig_size = fig.get_size_inches()
# aspect_ratio = fig_size[0] / fig_size[1]

# # Create connection lines using ConnectionPatch
# from matplotlib.patches import ConnectionPatch

# # Draw connections from the Montevideo rectangle to the inset
# con1 = ConnectionPatch(
#     xyA=(
#         montevideo_bounds["min_lon"],
#         montevideo_bounds["max_lat"],
#     ),  # Rectangle bottom-left
#     coordsA=ax_main.transData,
#     xyB=(
#         montevideo_bounds["max_lon"],
#         montevideo_bounds["max_lat"],
#     ),  # Inset bottom-left
#     coordsB=axins.transData,
#     linestyle="--",
#     color="black",
#     alpha=0.5,
#     linewidth=1,
# )
# fig.add_artist(con1)

# con2 = ConnectionPatch(
#     xyA=(
#         montevideo_bounds["max_lon"],
#         montevideo_bounds["min_lat"],
#     ),  # Rectangle top-right
#     coordsA=ax_main.transData,
#     xyB=(montevideo_bounds["max_lon"], montevideo_bounds["min_lat"]),  # Inset top-right
#     coordsB=axins.transData,
#     linestyle="--",
#     color="black",
#     alpha=0.5,
#     linewidth=1,
# )
# fig.add_artist(con2)

# plt.savefig(
#     "research/datasets/plots/uruguay_ute_stations_map.pdf",
#     dpi=1000,
#     bbox_inches="tight",
# )
# plt.show()

# # %%

# # %%
