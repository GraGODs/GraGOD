# %%
import argparse

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from datasets.dataset import SlidingWindowDataset
from gragod import start_research  # noqa
from gragod import ParamFileTypes
from gragod.training import load_params, load_training_data
from gragod.types import CleanMethods, Datasets
from models.dc_vae.predict import main

# %%
params = load_params("models/dc_vae/params.yaml", file_type=ParamFileTypes.YAML)
# %%
result = main(
    dataset_name=params["dataset"],
    # ckpt_path=args.ckpt_path,
    **params["train_params"],
    model_params=params["model_params"],
    params=params,
)
# %%
result.keys()
# %%
predictions = result["predictions"]
labels = result["labels"]
scores = result["scores"]
data = result["data"][params["model_params"]["window_size"] :]
thresholds = result["thresholds"]
metrics = result["metrics"]
x_means = result["x_means"]
x_log_vars = result["x_log_vars"]
# %%
x_vars = torch.exp(x_log_vars)
x_vars
# %%
data.shape
# %%
labels.shape
# %%
thresholds
# %%
column = 0
start_time = 8500
end_time = 9000
plt.figure(figsize=(15, 5))
plt.plot(labels[start_time:end_time, column])
plt.plot(data[start_time:end_time, column])
plt.plot(scores[start_time:end_time, column])
plt.title(f"Time Series Plot for Column {column}")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend(["True", "Scores"])
plt.grid(True)
plt.show()
# %%
predictions_online = scores > 1
plt.figure(figsize=(15, 5))
plt.plot(labels[start_time:end_time, column])
plt.plot(predictions_online[start_time:end_time, column])
plt.title(f"Time Series Plot for Column {column}")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend(["True", "Pred"])
plt.grid(True)
plt.show()
# %%
predictions_online = scores > 0.7
column = 11
start_time = 10500
end_time = 11000
plt.figure(figsize=(15, 5))
plt.plot(labels[start_time:end_time, column])
plt.plot(data[start_time:end_time, column])
plt.plot(x_means[start_time:end_time, column])
plt.fill_between(
    range(0, end_time - start_time),
    x_means[start_time:end_time, column]
    - torch.sqrt(x_vars[start_time:end_time, column]),
    x_means[start_time:end_time, column]
    + torch.sqrt(x_vars[start_time:end_time, column]),
    alpha=0.2,
)

# plt.plot(predictions[start_time:end_time, column])
# plt.plot(scores[start_time:end_time, column])
# plt.plot(predictions_online[start_time:end_time, column])
plt.title(f"Time Series Plot for Series {column + 1}")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend(["True Label", "Series", "Forecast", "Variance"])
plt.grid(True)
plt.show()
# %%
predictions_online = scores > 0.7
column = 3
start_time = 6000
end_time = start_time + 2000
plt.figure(figsize=(15, 5))
plt.plot(labels[start_time:end_time, column] * 5)
plt.plot(data[start_time:end_time, column])
plt.plot(forecasts[start_time:end_time, column])
# plt.plot(predictions[start_time:end_time, column])
plt.plot(scores[start_time:end_time, column])
# plt.plot(predictions_online[start_time:end_time, column])
plt.title(f"Time Series Plot for Series {column + 1}")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend(["True Label", "Series", "Forecast", "Scores"])
plt.grid(True)
plt.show()
