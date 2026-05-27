# %%
from gragod import start_research
import pandas as pd
import numpy as np
from datasets.telco import load_telco_df
from matplotlib import pyplot as plt
from datasets.data_processing import preprocess_df
import seaborn as sns
from IPython.display import display

# %%
df_train, df_train_labels, df_val, df_val_labels, df_test, df_test_labels = (
    load_telco_df()
)
df_train.drop(columns=["time"], inplace=True)
df_train_labels.drop(columns=["time"], inplace=True)
df_val.drop(columns=["time"], inplace=True)
df_val_labels.drop(columns=["time"], inplace=True)
df_test.drop(columns=["time"], inplace=True)
df_test_labels.drop(columns=["time"], inplace=True)

# %%
df_train.head()
# %%
df_train.shape
# %%
df_train_labels.head()
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
    start_time = 8200
    end_time = -1
    # start_time, end_time = 0, -1
    plt.figure(figsize=(15, 5))
    plt.plot(df.iloc[start_time:end_time, column])
    plt.plot(df_train_labels.iloc[start_time:end_time, column], label="Labels")
    # plt.title(f"Time Series Plot for Column {column}")
    # plt.xlabel("Time")
    # plt.ylabel("Value")
    plt.legend(loc="upper right")
    # plt.grid(True)

    plt.tight_layout()
    plt.show()


plot_time_series(df_train, df_train_labels, 1)
# for column in range(df_train.shape[1]):
#     plot_time_series(df_train, df_train_labels, column)
# %%

# %%

# %%
data_1, labels_1, scaler_1 = preprocess_df(
    data_df=df_train,
    labels_df=df_train_labels,
    normalize=False,
    clean=False,
    max_std=0.0,
)
data_2, labels_2, scaler_2 = preprocess_df(
    data_df=df_train,
    labels_df=df_train_labels,
    normalize=False,
    clean=False,
    max_std=7.0,
)
for column in range(data_1.shape[1]):
    if not np.allclose(data_1[:, column], data_2[:, column]):
        print(f"Column {column} is not the same")
# %%
column = 10
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
def plot_time_series_numpy(ts: np.ndarray, column: int):
    plt.figure(figsize=(15, 5))
    plt.plot(ts[:, column])
    plt.title(f"Time Series Plot for Column {column}")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


for column in range(data_1.shape[1]):
    plot_time_series_numpy(data_2, column)

# %%
# z_scores = (data_1 - data_1.mean()) / data_1.std()
# plt.figure(figsize=(15, 5))
# plt.plot(z_scores[start_time:end_time, column])
# plt.axhline(y=7, color="r", linestyle="--", label="Threshold = 7")
# plt.title(f"Z-scores for Column {column}")
# plt.xlabel("Time")
# plt.ylabel("Z-score")
# plt.legend(loc="upper right")
# plt.grid(True)
# plt.tight_layout()
# plt.show()

z_scores = (data_2 - data_2.mean()) / data_2.std()
plt.figure(figsize=(15, 5))
plt.plot(z_scores[start_time:end_time, column])
plt.axhline(y=7, color="r", linestyle="--", label="Threshold = 7")
plt.title(f"Z-scores for Column {column}")
plt.xlabel("Time")
plt.ylabel("Z-score")
plt.legend(loc="upper right")
plt.grid(True)
plt.tight_layout()
plt.show()
# %%
column = 2
start_time, end_time = 0, -1
z_scores = (df_train - df_train.mean()) / df_train.std()
z_scores = np.abs(z_scores)
max_std = 7.0
cutoff_value = df_train.mean() + max_std * df_train.std()
mask = z_scores > max_std
df_train_copy = df_train.copy()
df_train_copy.mask(mask, inplace=True)
df_train_copy.fillna(cutoff_value, inplace=True)
plt.figure(figsize=(15, 5))
plt.plot(df_train_copy.iloc[start_time:end_time, column], label="Masked")
plt.axhline(
    y=cutoff_value.iloc[column], color="r", linestyle="--", label="Threshold = 7"
)
# plt.plot(z_scores.iloc[start_time:end_time, column], label="Original")
# plt.plot(mask.iloc[start_time:end_time, column] * 5, label="Mask")
# plt.axhline(y=7, color="r", linestyle="--", label="Threshold = 7")
plt.title(f"Z-scores for Column {column}")
plt.xlabel("Time")
plt.ylabel("Z-score")
plt.legend(loc="upper right")
plt.grid(True)
plt.tight_layout()
plt.show()
# %%
display(df_train_copy.iloc[start_time:end_time, column])
display(z_scores.iloc[start_time:end_time, column])
display(mask.iloc[start_time:end_time, column])
# %%
diff = np.abs(data_1 - data_2)
plt.figure(figsize=(15, 5))
plt.plot(diff[:, column])
plt.title("Difference between processed data")
plt.xlabel("Time")
plt.ylabel("Difference")
plt.legend(loc="upper right")
plt.grid(True)
plt.tight_layout()
plt.show()


# %%
# Plot histogram distribution for the data for a column
def plot_histogram(ts: np.ndarray, column: int):
    normalized_ts = (ts - ts.mean()) / ts.std()
    plt.figure(figsize=(10, 6))
    plt.hist(normalized_ts, bins=50, alpha=0.75, edgecolor="black")
    plt.title(f"Histogram Distribution for Column {column}")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


for column in range(df_train.shape[1]):
    plot_histogram(df_train.iloc[:, column].to_numpy(), column)
# %%

# %%
