# %%
import pandas as pd
import os
import json
import torch
from pathlib import Path
import tensorflow as tf
import matplotlib.pyplot as plt

# Plot the correlation matrix as a heatmap
import matplotlib.pyplot as plt
import seaborn as sns

base_path = "/dataslow/fbello/output"
model = "gdn"
dataset = "swat"
path_to_versions = f"{base_path}/{dataset}/{model}_optimization_{dataset}/{model}/"
loss_name = "Loss/val" if model == "gru" else "Loss/val_epoch"


# %%
def load_metrics(path_to_versions) -> pd.DataFrame:
    # Load metrics for each version
    metrics_data = []
    for version_dir in os.listdir(path_to_versions):
        version_path = os.path.join(path_to_versions, version_dir)
        if os.path.isdir(version_path):
            val_metrics_json = os.path.join(version_path, "val_metrics.json")
            if os.path.exists(val_metrics_json):
                with open(val_metrics_json, "r") as f:
                    val_metrics = json.load(f)
                    val_metrics["version"] = version_dir
                    metrics_data.append(val_metrics)
    return pd.DataFrame(metrics_data)


def sort_df_by_version(df: pd.DataFrame) -> pd.DataFrame:
    # Set the index to be the version number extracted from the version directory name
    # Extract version numbers and set as index
    df.index = df["version"].str.extract(r"version_(\d+)").astype(int).iloc[:, 0]
    # Sort the DataFrame by the index (version number)
    df = df.sort_index()
    print(f"Loaded metrics for {len(df)} versions")
    return df


metrics_df = load_metrics(path_to_versions)
metrics_df = sort_df_by_version(metrics_df)


# %%
import traceback
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


# Extraction function
def tflog2pandas(path):
    # Initialize an empty dictionary to store metrics
    metrics_dict = {}
    steps = set()

    try:
        event_acc = EventAccumulator(path)
        event_acc.Reload()
        tags = event_acc.Tags()["scalars"]

        # First collect all steps and metrics
        for tag in tags:
            event_list = event_acc.Scalars(tag)
            if tag not in metrics_dict:
                metrics_dict[tag] = {}

            for event in event_list:
                metrics_dict[tag][event.step] = event.value
                steps.add(event.step)

        # Create DataFrame with steps as index
        all_steps = sorted(list(steps))
        df = pd.DataFrame(index=all_steps)

        # Add each metric as a column
        for tag, step_values in metrics_dict.items():
            values = [step_values.get(step, None) for step in all_steps]
            df[tag] = values

        # Ensure the index is named 'step'
        df.index.name = "step"

    except Exception:
        print("Event file possibly corrupt: {}".format(path))
        traceback.print_exc()
        return pd.DataFrame()

    return df


def get_best_val_loss_for_versions(path_to_versions) -> pd.DataFrame:
    df_val_losses = pd.DataFrame()
    for i, version_directory in enumerate(os.listdir(path_to_versions)):
        # Find all event files in the directory
        print(f"Processing version {i+1} of {len(os.listdir(path_to_versions))}")
        event_files = [
            f
            for f in os.listdir(f"{path_to_versions}{version_directory}")
            if f.startswith("events.out.tfevents")
        ]

        if not event_files:
            print(f"No event files found in {version_directory}")
            continue

        for eventId in event_files:
            try:
                path = f"{path_to_versions}{version_directory}/{eventId}"  # folderpath
                df = tflog2pandas(path)
                best_val_loss = (
                    df[~df[loss_name].isna()]
                    .sort_values(by=loss_name)[loss_name]
                    .iloc[0]
                )
                if best_val_loss is not None:
                    id = version_directory.split("_")[-1]
                    df_val_losses.loc[id, "version"] = version_directory
                    df_val_losses.loc[id, "best_val_loss"] = best_val_loss
            except Exception as e:
                continue
    return df_val_losses


df_val_losses = get_best_val_loss_for_versions(path_to_versions)
df_val_losses = sort_df_by_version(df_val_losses)
print(len(df_val_losses))
# %%
metrics_df["best_val_loss"] = df_val_losses["best_val_loss"]

# keep only the versions that have vus_pr_system
metrics_df = metrics_df[metrics_df["vus_pr_system"].notna()]


# %%
import numpy as np


def plot_correlation_matrix(
    metrics_df: pd.DataFrame,
    ignore_columns: list[str] = ["custom_f1_system"],
    save_path: str | None = None,
):
    # Calculate correlation matrix
    metrics_df = metrics_df.drop(columns=ignore_columns)
    correlation_matrix = metrics_df.corr()

    # Format column names for display (capitalize and replace underscores with spaces)
    formatted_labels = [
        col.replace("_", " ").title() for col in correlation_matrix.columns
    ]
    formatted_labels = [label.replace("System", "") for label in formatted_labels]
    formatted_labels = [label.replace("Pr ", "PR ") for label in formatted_labels]
    formatted_labels = [label.replace("Roc", "ROC") for label in formatted_labels]
    formatted_labels = [label.replace("Vus ", "VUS-") for label in formatted_labels]
    formatted_labels = [label.replace("Based ", "") for label in formatted_labels]

    # Create a mask for the upper triangle to show only lower triangle (excluding diagonal)
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
    # Create a better figure with more space
    plt.figure(figsize=(6, 6))

    # Use a more visually appealing colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Plot the heatmap with improved aesthetics
    sns.heatmap(
        correlation_matrix,
        annot=True,  # Show correlation values
        cmap=cmap,  # Better color map
        vmin=-1,
        vmax=1,  # Value range
        square=True,  # Make cells square
        linewidths=0.5,  # Add grid lines
        fmt=".2f",  # Format for annotation (2 decimal places)
        xticklabels=formatted_labels,  # Skip the last label in x-axis
        yticklabels=formatted_labels,  # Skip the first label in y-axis
        mask=mask,  # Apply the mask
        annot_kws={"size": 9},  # Increased annotation text size
        cbar_kws={
            "shrink": 0.6,
        },
    )

    # Improve label readability
    font_size = 10
    plt.xticks(rotation=45, ha="right", rotation_mode="anchor", fontsize=font_size)
    plt.yticks(fontsize=font_size)

    # Adjust layout
    plt.tight_layout()

    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=1000, bbox_inches="tight")

    plt.show()


plot_correlation_matrix(
    metrics_df, save_path=f"correlation_matrix_{model}_{dataset}.pdf"
)

# %%
