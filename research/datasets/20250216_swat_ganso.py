# %%
from gragod import start_research
import pandas as pd
import numpy as np
from datasets.swat import load_swat_df
from matplotlib import pyplot as plt
from datasets.data_processing import preprocess_df
import seaborn as sns
from IPython.display import display
import torch

# %%
df_train, df_train_labels, df_val, df_val_labels, df_test, df_test_labels = (
    load_swat_df()
)
df_train.drop(columns=[" Timestamp"], inplace=True)
# df_train_labels.drop(columns=[" Timestamp"], inplace=True)
df_val.drop(columns=[" Timestamp"], inplace=True)
# df_val_labels.drop(columns=[" Timestamp"], inplace=True)
df_test.drop(columns=[" Timestamp"], inplace=True)
# df_test_labels.drop(columns=[" Timestamp"], inplace=True)
six_hours_in_seconds = 6 * 60 * 60
df_train = df_train.iloc[six_hours_in_seconds:]
df_train_labels = df_train_labels.iloc[six_hours_in_seconds:]
# %%
df_train.head()
# %%
df_train.shape
# %%
columns_train = set(df_train.columns)
columns_val = set(df_val.columns)
# %%
columns_train - columns_val
# %%
columns_val - columns_train
# %%

# %%
for column in range(df_train_labels.shape[1]):
    print(
        f"Column {column} unique values: {np.unique(df_train_labels.iloc[:, column].values)}"
    )
    print(f"Column {column} nan count: {df_train_labels.iloc[:, column].isna().sum()}")
    print(
        f"Column {column} nan value in X count: {np.sum(df_train.iloc[:, column].isna())}"
    )
# %%

# Filter and print column 1 series where the label is NaN
nan_indices = df_train_labels.iloc[:, 0].isna()


print("Indices of NaN values in column 1:", np.where(nan_indices)[0])

# print(df_train.iloc[nan_indices, 1])

# %%

# %%


# %%
def plot_time_series(df: pd.DataFrame, labels: pd.DataFrame, column: int):
    start_time, end_time = 0, -1
    plt.figure(figsize=(15, 5))

    # Plot the data on the primary y-axis
    plt.plot(df.iloc[start_time:end_time, column], label="Values")
    plt.title(f"Time Series Plot for Column {column}")
    plt.xlabel("Time")
    plt.ylabel("Values")

    # Create a secondary y-axis for the labels
    ax2 = plt.gca().twinx()
    ax2.plot(labels.iloc[start_time:end_time], label="Labels", color="orange")
    ax2.set_ylabel("Labels")

    # Add legends for both y-axes
    plt.legend(loc="upper left")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.show()


# plot_time_series(df_train, df_train_labels, 10)
# plot_time_series(df_test, df_test_labels, 10)

for column in range(df_train.shape[1]):
    plot_time_series(df_test, df_test_labels, column)
# %%
data_1, labels_1, scaler_1 = preprocess_df(
    data_df=df_train,
    labels_df=df_train_labels,
    normalize=True,
    clean=False,
    max_std=0.0,
)
data_2, labels_2, scaler_2 = preprocess_df(
    data_df=df_train,
    labels_df=df_train_labels,
    normalize=True,
    clean=False,
    max_std=7.0,
)
columns_varied = []
for column in range(data_1.shape[1]):
    if not np.allclose(data_1[:, column], data_2[:, column]):
        print(f"Column {column} is not the same")
        columns_varied.append(column)

# %%
for column in range(data_1.shape[1]):
    # for column in [10]:
    start_time, end_time = 0, -1
    # start_time, end_time = 9090, 9100
    plt.figure(figsize=(15, 5))
    plt.plot(data_1[start_time:end_time, column], label=f"Value column {column}")
    plt.title(f"Time Series Plot for Column {column}")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(15, 5))
    plt.plot(data_2[start_time:end_time, column], label=f"Value column {column}")
    plt.title(f"Time Series Plot for Column {column}")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# %%
column = 50
plt.figure(figsize=(15, 5))
plt.plot(data_2[start_time:end_time, column], label=f"Value column {column}")
plt.title(f"Time Series Plot for Column {column}")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend(loc="upper right")
plt.grid(True)
plt.tight_layout()
plt.show()

print(f"Mean: {torch.mean(data_2[:, column])}")
print(f"Std: {torch.std(data_2[:, column])}")
print(f"Min: {torch.min(data_2[:, column])}")
print(f"Max: {torch.max(data_2[:, column])}")
# %%
columns_to_remove = []
for column in range(data_2.shape[1]):
    min_value = torch.min(data_2[:, column])
    max_value = torch.max(data_2[:, column])
    if min_value == max_value:
        print(f"Column {column} has the same min and max value: {min_value}")
        columns_to_remove.append(column)
df_train.columns[columns_to_remove]
# %%
plt.figure(figsize=(15, 5))
# plt.plot(df_val.iloc[:, column])
plt.plot(df_test_labels.iloc[:], label="Labels")
plt.legend(loc="upper right")

plt.tight_layout()
plt.show()


# %%
# Function to count the number of up flanks in a time series
def count_up_flanks(data):
    up_flanks = 0
    for i in range(1, len(data)):
        if (data.iloc[i].values[0] == 1) and (data.iloc[i - 1].values[0] == 0):
            up_flanks += 1
    return up_flanks


# Example usage
column = 50
up_flanks_count = count_up_flanks(df_test_labels)
print(f"Number of up flanks: {up_flanks_count}")

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

# %%

params = load_params("models/gru/params_swat.yaml", ParamFileTypes.YAML)
params["predictor_params"]["max_std"] = 0.0
results_v0 = predict(
    model=Models.GRU,
    dataset=Datasets.SWAT,
    ckpt_path="benchmarks_swat/gru/version_1/best.ckpt",
    params_file="models/gru/params_swat.yaml",
    params=params,
    model_params=params["model_params"],
    **params["predictor_params"],
)

# %%
params["predictor_params"]["max_std"] = 0.0
results_v1 = predict(
    model=Models.GRU,
    dataset=Datasets.SWAT,
    ckpt_path="gru_swat/gru_swat/best.ckpt",
    params_file="models/gru/params_swat.yaml",
    params=params,
    model_params=params["model_params"],
    **params["predictor_params"],
)
# %%
train_results_v1 = results_v1["train"]
val_results_v1 = results_v1["val"]
test_results_v1 = results_v1["test"]
train_results_v0 = results_v0["train"]
val_results_v0 = results_v0["val"]
test_results_v0 = results_v0["test"]
# %%
# Print data and predictions for v1
column = 0
start_time = 50000
end_time = 60000
start_time, end_time = 0, -1


def plot_results(results, start_time, end_time, column):
    plt.figure(figsize=(15, 5))
    plt.plot(results["labels"][start_time:end_time], label="True Labels")
    plt.plot(results["predictions"][start_time:end_time, column], label="Predictions")
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
    # plt.plot(results["data"][start_time:end_time, column], label="Data")
    plt.plot(results["output"][start_time:end_time, column], label="Forecasts")
    plt.title(f"Train Results for Column {column}")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    score = torch.max(results["scores"], dim=1).values
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
    ax2.plot(results["labels"][start_time:end_time], label="True Labels", color="r")
    ax2.set_ylabel("True Labels", color="r")

    # Add legends for both y-axes
    plt.legend(loc="upper right")
    ax2.legend(loc="upper left")

    plt.tight_layout()
    plt.show()


# start_time, end_time = 6000, -1
plot_results(test_results_v0, start_time, end_time, column)
plot_results(val_results_v1, start_time, end_time, column)
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
