# %%
import os

import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader

from datasets.config import get_dataset_config
from gragod import start_research
from gragod.metrics import get_metrics
from gragod.training import load_params, load_training_data, set_seeds
from gragod.types import ParamFileTypes, cast_dataset
from models.mtad_gat.dataset import Dataset, SlidingWindowDataset
from models.mtad_gat.model import MTAD_GAT
from models.mtad_gat.predict import generate_scores, get_predictions, run_model
from models.mtad_gat.spot import SPOT
from models.mtad_gat.trainer_pl import MTAD_GAT_PLModule

# %%

params = load_params("models/mtad_gat/params.yaml", ParamFileTypes.YAML)
dataset_name = params["predictor_params"]["dataset"]
test_size = 0.1
val_size = 0.1
interpolate_method = None
batch_size = 256
n_workers = 0
model_params = params["model_params"]
ckpt_path = None
device = "mps"

# %%

dataset = cast_dataset(dataset_name)
dataset_config = get_dataset_config(dataset=dataset)

# Load data
X_train, X_val, X_test, X_train_labels, X_val_labels, X_test_labels = (
    load_training_data(
        dataset=dataset,
        test_size=test_size,
        val_size=val_size,
        normalize=dataset_config.normalize,
        clean=False,
        interpolate_method=interpolate_method,
    )
)

window_size = model_params["window_size"]
X_train_labels = X_train_labels[window_size:]
X_val_labels = X_val_labels[window_size:]
X_test_labels = X_test_labels[window_size:]

# Create test dataloader
train_dataset = SlidingWindowDataset(X_train, window_size)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    num_workers=n_workers,
)
test_dataset = SlidingWindowDataset(X_test, window_size)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    num_workers=n_workers,
)

# Create and load model
n_features = X_train.shape[1]
model = MTAD_GAT(
    n_features=n_features,
    out_dim=n_features,
    **model_params,
)

checkpoint_path = (
    os.path.join(params["predictor_params"]["ckpt_folder"], "mtad_gat.ckpt")
    if ckpt_path is None
    else ckpt_path
)

if not os.path.exists(checkpoint_path):
    raise ValueError(f"Checkpoint not found at {checkpoint_path}")

print(f"Loading model from checkpoint: {checkpoint_path}")
lightning_module = MTAD_GAT_PLModule.load_from_checkpoint(
    checkpoint_path,
    model=model,
    model_params=model_params,
    **params["train_params"],
)

model = lightning_module.model.to(device)
model.eval()
# %%
forecasts_train, reconstructions_train = run_model(
    model=lightning_module,
    loader=train_loader,
    device=device,
)
forecasts_test, reconstructions_test = run_model(
    model=lightning_module,
    loader=test_loader,
    device=device,
)
# %%
EPSILON = 0.8
train_scores = generate_scores(
    forecasts=forecasts_train,
    reconstructions=reconstructions_train,
    data=X_train,
    window_size=window_size,
    EPSILON=EPSILON,
)
test_scores = generate_scores(
    forecasts=forecasts_test,
    reconstructions=reconstructions_test,
    data=X_test,
    window_size=window_size,
    EPSILON=EPSILON,
)
X_test_pred = get_predictions(train_scores, test_scores)

# %%
n_samples = 1000
forecasts_test_df = pd.DataFrame(forecasts_test).iloc[:n_samples]
reconstructions_test_df = pd.DataFrame(reconstructions_test).iloc[:n_samples]

test_values_df = pd.DataFrame(X_test[window_size:]).iloc[:n_samples]
test_labels_df = pd.DataFrame(X_test_labels).iloc[:n_samples]

# Create a figure with a reasonable size


def plot_time_series(
    predictions, reconstructions, values, labels, scores, column, n_samples
):
    predictions = pd.DataFrame(predictions).iloc[:n_samples]
    reconstructions = pd.DataFrame(reconstructions).iloc[:n_samples]
    values = pd.DataFrame(values).iloc[:n_samples]
    labels = pd.DataFrame(labels).iloc[:n_samples]
    # Plot for the selected columns

    plt.figure(figsize=(15, 8))
    plt.plot(predictions[column], label=f"Feature {column} (predictions)")
    plt.plot(reconstructions[column], label=f"Feature {column} (reconstructions)")
    plt.plot(values[column], label=f"Feature {column} (values)")
    plt.plot(labels[column], label=f"Feature {column} (labels)")
    plt.plot(scores[:n_samples, column], label=f"Feature {column} (scores)")
    plt.legend()
    plt.show()


plot_time_series(
    predictions=test_predictions,
    reconstructions=test_reconstructions,
    values=X_test[window_size:],
    labels=X_test_labels,
    scores=test_score,
    column=6,
    n_samples=1000,
)
# %%
n_samples = 1000
forecasts_train_df = pd.DataFrame(forecasts_train).iloc[:n_samples]
reconstructions_train_df = pd.DataFrame(reconstructions_train).iloc[:n_samples]

train_values_df = pd.DataFrame(X_train[window_size:]).iloc[:n_samples]
train_labels_df = pd.DataFrame(X_train_labels).iloc[:n_samples]

(
    X_train_clean,
    X_val_clean,
    X_test_clean,
    X_train_labels_clean,
    X_val_labels_clean,
    X_test_labels_clean,
) = load_training_data(
    dataset=dataset,
    test_size=test_size,
    val_size=val_size,
    normalize=dataset_config.normalize,
    clean=True,
    interpolate_method=interpolate_method,
)


# %%
def plot_time_series(
    forecasts, reconstructions, values, labels, scores, column, n_samples
):
    forecasts = pd.DataFrame(forecasts).iloc[:n_samples]
    reconstructions = pd.DataFrame(reconstructions).iloc[:n_samples]
    values = pd.DataFrame(values).iloc[:n_samples]
    labels = pd.DataFrame(labels).iloc[:n_samples]
    # Plot for the selected columns

    plt.figure(figsize=(15, 8))
    plt.plot(forecasts[column], label=f"Feature {column} (predictions)")
    plt.plot(reconstructions[column], label=f"Feature {column} (reconstructions)")
    plt.plot(values[column], label=f"Feature {column} (values)")
    plt.plot(labels[column], label=f"Feature {column} (labels)")
    plt.plot(scores[:n_samples, column], label=f"Feature {column} (scores)")
    plt.legend()
    plt.show()


plot_time_series(
    forecasts=forecasts_train,
    reconstructions=reconstructions_train,
    values=X_train[window_size:],
    labels=X_train_labels,
    scores=train_scores,
    column=6,
    n_samples=1000,
)
# %%
plot_time_series(
    forecasts=forecasts_train,
    reconstructions=reconstructions_train,
    values=X_train_clean[window_size:],
    labels=X_train_labels,
    scores=train_scores,
    column=6,
    n_samples=1000,
)
# %%
