# %%
import os

import torch
from torch import nn
from torch.utils.data import DataLoader

from datasets.dataset import SlidingWindowDataset
from datasets.swat import load_swat_training_data
from gragod.metrics import get_metrics, print_all_metrics
from gragod.predictions.prediction import (
    get_threshold,
    smooth_scores,
    standarize_error_scores,
)
from gragod.training import load_params
from gragod.types import ParamFileTypes, cast_dataset
from models.gdn.model import GDN

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# %%
X_train, X_val, X_test, y_train, y_val, y_test = load_swat_training_data(down_len=10)

params = load_params("models/gdn/params.yaml", file_type=ParamFileTypes.YAML)
# %%
device = params["train_params"]["device"]
model_params = params["model_params"]

edge_index = (
    torch.tensor(
        [[i, j] for i in range(X_train.shape[1]) for j in range(X_train.shape[1])],
        dtype=torch.long,  # edge_index must be long type
    )
    .t()
    .to(device)
)

train_dataset = SlidingWindowDataset(
    data=X_train,
    edge_index=edge_index,
    window_size=model_params["window_size"],
    labels=y_train,
    drop=False,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=params["train_params"]["batch_size"],
    num_workers=0,
    shuffle=True,
)

test_dataset = SlidingWindowDataset(
    data=X_test,
    edge_index=edge_index,
    window_size=model_params["window_size"],
    labels=y_test,
    drop=False,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=params["train_params"]["batch_size"],
    num_workers=0,
    shuffle=False,
)
# %%
model = GDN(
    [edge_index],
    X_train.shape[1],
    **model_params,
).to(device)
# %%
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
n_epoch = 50
loss_val_history = []
vus_roc_auc_history = []
f1_history = []
loss_train_history = []
for epoch in range(n_epoch):
    model.train()
    train_loss = 0
    train_forecasts = []
    for x, y, out_labels, edge_index in train_loader:
        x = x.reshape(-1, x.size(2), x.size(1)).to(device)
        y = y.squeeze(1).to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = nn.MSELoss()(out, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        for i in range(out.shape[0]):
            train_forecasts.append(out[i].detach().cpu().numpy())
    avg_train_loss = train_loss / (
        len(train_loader) * params["train_params"]["batch_size"]
    )
    loss_train_history.append(avg_train_loss)
    print(f"Epoch {epoch+1}/{n_epoch}, Training Loss: {avg_train_loss:.4f}")

    # Validation
    model.eval()
    val_loss = 0
    forecasts = []
    with torch.no_grad():
        for x, y, out_labels, edge_index in test_loader:
            x_shaped = x.reshape(-1, x.size(2), x.size(1)).to(device)
            y = y.squeeze(1).to(device)
            out = model(x_shaped)
            loss = nn.MSELoss()(out, y)
            val_loss += loss.item()
            for i in range(out.shape[0]):
                forecasts.append(out[i].detach().cpu().numpy())

    avg_val_loss = val_loss / (len(test_loader) * params["train_params"]["batch_size"])
    loss_val_history.append(avg_val_loss)
    print(f"Validation Loss: {avg_val_loss:.4f}")
    scores = torch.abs(torch.tensor(forecasts) - X_test[model_params["window_size"] :])
    scores = scores.squeeze(0)
    scores = standarize_error_scores(scores)
    scores = smooth_scores(scores, window_size=10)
    train_scores = torch.tensor(train_forecasts)
    train_scores = train_scores.squeeze(0)
    train_scores = standarize_error_scores(train_scores)
    train_scores = smooth_scores(train_scores, window_size=10)
    threshold = get_threshold(
        cast_dataset("swat"),
        scores[: y_val[model_params["window_size"] :].shape[0]],
        y_val[model_params["window_size"] :],
        1000,
    )
    metrics = get_metrics(
        cast_dataset("swat"),
        predictions=scores > threshold,
        labels=y_test[model_params["window_size"] :],
        scores=scores,
    )
    vus_roc_auc_history.append(metrics["vus_roc_system"])
    f1_history.append(metrics["f1_system"])
    print_all_metrics(metrics, f"Epoch {epoch+1}/{n_epoch}")
# %%
import matplotlib.pyplot as plt

loss_val_history_normalized = [x / max(loss_val_history) for x in loss_val_history]
vus_roc_auc_history_normalized = [
    x / max(vus_roc_auc_history) for x in vus_roc_auc_history
]
f1_history_normalized = [x / max(f1_history) for x in f1_history]
loss_train_history_normalized = [
    x / max(loss_train_history) for x in loss_train_history
]
plt.plot(loss_train_history_normalized, label="Loss Train")
plt.plot(loss_val_history_normalized, label="Loss Val")
plt.plot(vus_roc_auc_history_normalized, label="VUS-ROC AUC")
plt.plot(f1_history_normalized, label="F1")
plt.legend()
plt.show()
# %%
