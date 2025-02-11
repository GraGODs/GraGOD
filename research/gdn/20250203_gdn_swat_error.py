# %%
from pathlib import Path

project_root = Path("..", "..")

import torch
from models.gdn.model import GDN_PLModule, GDN
from gragod.training import load_training_data
from gragod import Datasets, CleanMethods, InterPolationMethods
from datasets.swat import load_swat_training_data
from datasets.dataset import get_data_loader, get_edge_index
from gragod.predictions.prediction import generate_scores

# best_params_ckpt = (
#     "/dataslow/fbello/output/gdn_optimization_swat/gdn/version_87/best.ckpt"
# )
best_params_ckpt = (
    "/home/fbello/GraGOD/output/gdn/gdn_graph_topologies_small/version_0/best.ckpt"
)
X_train, X_val, X_test, y_train, y_val, y_test = load_swat_training_data(
    path_to_dataset=str(Path(project_root, "datasets_files", "swat")),
    normalize=True,
    clean=False,
    interpolate_method=None,
)
print(X_train.shape)
print(y_train.shape)

model_pl = GDN_PLModule.load_from_checkpoint(best_params_ckpt, map_location="cuda")
model: GDN = model_pl.model  # type: ignore
# %%
# window_size = 255
window_size = 5
device = "cuda"
edge_index = get_edge_index(X_train, device)
val_loader = get_data_loader(
    X=X_val,
    edge_index=edge_index,
    y=y_val,
    window_size=window_size,
    clean=CleanMethods.NONE,
    batch_size=256,
    n_workers=0,
    shuffle=True,
)
test_loader = get_data_loader(
    X=X_test,
    edge_index=edge_index,
    y=y_test,
    window_size=window_size,
    clean=CleanMethods.NONE,
    batch_size=256,
    n_workers=0,
    shuffle=False,
)
# %%
predictions = torch.empty(
    (0, X_val.shape[1])
)  # Initialize empty tensor with correct feature dim
model.eval()
with torch.no_grad():
    for x, y, out_labels, edge_index in val_loader:
        x = x.reshape(-1, x.size(2), x.size(1)).to(device)
        y = y.squeeze(1).to(device)
        prediction = torch.tensor(model(x))  # (batch_size, n_features)
        predictions = torch.cat([predictions, prediction.detach().cpu()], dim=0)

test_predictions = torch.empty(
    (0, X_test.shape[1])
)  # Initialize empty tensor with correct feature dim
model.eval()
with torch.no_grad():
    for x, y, out_labels, edge_index in test_loader:
        x = x.reshape(-1, x.size(2), x.size(1)).to(device)
        y = y.squeeze(1).to(device)
        prediction = torch.tensor(model(x))  # (batch_size, n_features)
        test_predictions = torch.cat(
            [test_predictions, prediction.detach().cpu()], dim=0
        )

# %%
val_scores = generate_scores(predictions, X_val[window_size:], score_type="abs")
test_scores = generate_scores(test_predictions, X_test[window_size:], score_type="abs")
# %%
import matplotlib.pyplot as plt

series = 17
start = 50000
end = 60000
plt.plot(test_predictions[start:end, series])
plt.plot(X_test[window_size:][start:end, series])
plt.plot(y_test[window_size:][start:end])
plt.legend(["prediction", "ground truth", "label"])
plt.show()
# %%
import numpy as np

print(np.average(val_scores[y_val[window_size:].squeeze() == 1]))
print(np.average(val_scores[y_val[window_size:].squeeze() == 0]))

print(np.average(test_scores[y_test[window_size:].squeeze() == 1]))
print(np.average(test_scores[y_test[window_size:].squeeze() == 0]))

# %%
test_scores_system = torch.mean(test_scores, dim=1)
print(test_scores_system.shape)
print(np.average(test_scores_system[y_test[window_size:].squeeze() == 1]))
print(np.average(test_scores_system[y_test[window_size:].squeeze() == 0]))
# %%
threshold = 0.25
label_predictions = np.int32((test_scores_system > threshold).squeeze().numpy())
y_pred = label_predictions
print(y_pred.shape)
print(np.unique(y_pred))

# %%
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score

# Ensure both arrays have same shape and type
y_true = np.int32(y_test[window_size:].squeeze().numpy())

precision, recall, fscore, support = precision_recall_fscore_support(y_true, y_pred)

f1 = f1_score(y_true, y_pred)
# %%
print(precision)
print(recall)
print(fscore)
print(support)
print(f1)
# %%
