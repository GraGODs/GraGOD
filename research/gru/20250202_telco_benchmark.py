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

params = load_params("models/gru/params_telco.yaml", ParamFileTypes.YAML)
params["predictor_params"]["max_std"] = 7.0
results_v0 = predict(
    model=Models.GRU,
    dataset=Datasets.TELCO,
    ckpt_path="benchmarks/gru/version_4/best.ckpt",
    params_file="models/gru/params_telco.yaml",
    params=params,
    model_params=params["model_params"],
    **params["predictor_params"],
)
# %%
params["predictor_params"]["max_std"] = 0.0
results_v1 = predict(
    model=Models.GRU,
    dataset=Datasets.TELCO,
    ckpt_path="benchmarks/gru/version_5/best.ckpt",
    params_file="models/gru/params_telco.yaml",
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
train_results_v1.keys
# %%
# Print data and predictions for v1
column = 10
start_time = 10000
end_time = 13000
start_time, end_time = 0, -1


def plot_results(results, start_time, end_time, column):
    plt.figure(figsize=(15, 5))
    plt.plot(results["labels"][start_time:end_time, column], label="True Labels")
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
    plt.plot(results["data"][start_time:end_time, column], label="Data")

    plt.plot(results["output"][start_time:end_time, column], label="Forecasts")
    plt.title(f"Train Results for Column {column}")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


start_time, end_time = 6000, -1
plot_results(train_results_v0, start_time, end_time, column)
plot_results(train_results_v1, start_time, end_time, column)

# %%
column = 10
start_time, end_time = 0, -1
plt.figure(figsize=(15, 5))
plt.plot(
    train_results_v0["predictions"][start_time:end_time, column]
    - train_results_v1["predictions"][start_time:end_time, column],
    label="Predictions v0 - v1",
)
plt.plot(train_results_v0["labels"][start_time:end_time, column], label="Labels v0")
plt.title(f"Train Results for Column {column}")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend(loc="upper right")
plt.grid(True)
plt.tight_layout()
plt.show()
# %%

# %%
for i, (f1_v0, f1_v1) in enumerate(
    zip(
        results_v0["train"]["metrics"]["range_based_f1_per_class"],
        results_v1["train"]["metrics"]["range_based_f1_per_class"],
    )
):
    print(i, f1_v0, f1_v1, f1_v0 - f1_v1)
# %%
print(results_v0["train"]["thresholds"])
print(results_v1["train"]["thresholds"])
print(results_v0["train"]["labels"].sum(axis=0))
# %%
print(results_v0["train"]["metrics"])
print(results_v0["test"]["metrics"])
