# %%

import matplotlib.pyplot as plt

from gragod import ParamFileTypes, start_research  # noqa
from gragod.training import load_params
from models.mtad_gat.predict import main

# %%
params = load_params("models/mtad_gat/params.yaml", file_type=ParamFileTypes.YAML)


result = main(
    dataset_name=params["dataset"],
    **params["train_params"],
    model_params=params["model_params"],
    params=params,
)

X_test_pred = result["predictions"]
X_test_true = result["labels"]
X_test_scores = result["scores"]
X_test = result["data"]
thresholds_test = result["thresholds"]
metrics = result["metrics"]
reconstructions = result["reconstructions"]
forecasts = result["forecasts"]
# %%
window_size = len(X_test) - len(forecasts)
X_test_ = X_test[window_size:]
# %%
# Scores and predictions
column = 0
start_time = 8000
end_time = 12000
plt.figure(figsize=(15, 5))
plt.plot(X_test_true[start_time:end_time, column])
plt.plot(X_test_scores[start_time:end_time, column])
# plt.axhline(y=thresholds_test[column], color="r", linestyle="--", label="Threshold")
# plt.plot(X_test_pred[:, column])
plt.title(f"Time Series Plot for Column {column}")
plt.xlabel("Time")
plt.ylabel("Value")
plt.grid(True)
plt.show()

# Original series
plt.figure(figsize=(15, 5))
plt.plot(X_test[start_time:end_time, column])
plt.plot(X_test_true[start_time:end_time, column])
plt.title(f"Original Time Series for Column {column}")
plt.xlabel("Time")
plt.ylabel("Value")
plt.grid(True)
plt.show()

# Forecasts
plt.figure(figsize=(15, 5))
plt.plot(forecasts[start_time:end_time, column])
plt.plot(X_test[start_time:end_time, column])
plt.plot(X_test_true[start_time:end_time, column])
plt.title(f"Forecasts for Column {column}")
plt.xlabel("Time")
plt.ylabel("Value")
plt.grid(True)
plt.show()

# Reconstructions
plt.figure(figsize=(15, 5))
plt.plot(reconstructions[start_time:end_time, column])
plt.plot(X_test[start_time:end_time, column])
plt.plot(X_test_true[start_time:end_time, column])
plt.title(f"Reconstructions for Column {column}")
plt.xlabel("Time")
plt.ylabel("Value")
plt.grid(True)
plt.show()

# %%
