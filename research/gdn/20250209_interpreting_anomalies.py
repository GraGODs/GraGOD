# %%
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

torch.set_printoptions(threshold=10000)
best_params_ckpt = (
    "/home/fbello/GraGOD/output/gdn/gdn_graph_topologies_small/version_0/best.ckpt"
)
from models.gdn.model import GDN_PLModule, GDN


model_pl = GDN_PLModule.load_from_checkpoint(best_params_ckpt, map_location="cuda")
model: GDN = model_pl.model

data = pd.read_csv("../../datasets_files/swat/SWaT_data_train.csv")
column_names = data.columns[1:-1]
feature_map = dict(zip(range(len(column_names)), column_names))


def tensor_to_networkx(tensor):
    """
    Convert a 2D tensor to a NetworkX graph and return edge indices.

    Args:
        tensor (torch.Tensor): Input 2D tensor representing node connections
                Shape: (num_nodes, n_edges)

    Returns:
        tuple: (nx.DiGraph, torch.Tensor)
            - Directed graph representing the tensor connections
            - Edge indices tensor
    """
    # Convert tensor to numpy if it's a CUDA tensor
    if tensor.is_cuda:
        tensor = tensor.cpu()

    # Convert tensor to numpy array
    adj_matrix = tensor.numpy()

    # Create a directed graph
    G = nx.DiGraph()

    # Prepare edge indices
    edge_index_list = []

    # Add nodes
    num_nodes = adj_matrix.shape[0]
    G.add_nodes_from(range(num_nodes))

    # Add edges based on the tensor values
    for i in range(num_nodes):
        for j in range(adj_matrix.shape[1]):
            # Add an edge from node i to the value at adj_matrix[i, j]
            if 0 <= adj_matrix[i, j] < num_nodes:
                G.add_edge(i, adj_matrix[i, j])
                edge_index_list.append([i, adj_matrix[i, j]])

    # Convert edge indices to a PyTorch tensor
    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t()

    return G, edge_index


# %%
feature_num = 51
coeff_weights = model.gnn_layers[0].gnn.alpha.cpu().detach().numpy()
learned_graph = model.learned_graph
_, edge_index = tensor_to_networkx(learned_graph)
weight_mat = np.zeros((feature_num, feature_num))

print(edge_index.shape)
print(coeff_weights.shape)

for i in range(len(edge_index[0])):
    edge_i, edge_j = edge_index[:, i]
    edge_i, edge_j = edge_i % feature_num, edge_j % feature_num
    weight_mat[edge_i][edge_j] += coeff_weights[i]

# %%
print(weight_mat)
print(weight_mat.shape)
# %%
eps = 1e-4
weight_mat = np.clip(weight_mat, eps, 1.0)
# Create directed graph from weight matrix
G = nx.from_numpy_array(weight_mat, create_using=nx.DiGraph)

# Get edge weights for coloring
edges = G.edges()
weights = [G[u][v]["weight"] for u, v in edges]

# Create spring layout for better visualization
pos = nx.spring_layout(G)

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 8))

# Draw nodes
nx.draw_networkx_nodes(G, pos, node_size=500, ax=ax)
nx.draw_networkx_labels(G, pos, ax=ax)

# Draw edges with colors based on weights
edges = nx.draw_networkx_edges(
    G,
    pos,
    edge_color=weights,
    edge_cmap=plt.cm.Reds,
    width=2,
    edge_vmin=0,
    edge_vmax=1,
    arrows=True,
    arrowsize=20,
    ax=ax,
)

# Add colorbar
sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(vmin=eps, vmax=1))
sm.set_array([])
plt.colorbar(sm, ax=ax)

plt.title("Graph Structure with Edge Weights")
ax.set_axis_off()
plt.show()

# %%
from models.predict import main
from gragod.types import cast_dataset, cast_model

model_name = cast_model("gdn")
dataset_name = cast_dataset("swat")
ckpt_path = (
    "/home/fbello/GraGOD/output/gdn/gdn_graph_topologies_small/version_0/best.ckpt"
)
params_file = "/home/fbello/GraGOD/models/gdn/params.yaml"

predict_output = main(model_name, dataset_name, ckpt_path, params_file)

# %%
system_scores = predict_output["test"]["scores"].max(dim=1)[0]
highest_anomaly_sample_idx = np.argmax(system_scores)
highest_anomaly_sensor_idx = predict_output["test"]["scores"][
    highest_anomaly_sample_idx
].argmax()
assert torch.isclose(
    torch.max(predict_output["test"]["scores"]),
    predict_output["test"]["scores"][highest_anomaly_sample_idx][
        highest_anomaly_sensor_idx
    ],
    rtol=1e-5,
    atol=1e-8,
)

# %%
scores = np.stack(
    [weight_mat[highest_anomaly_sensor_idx], weight_mat[:, highest_anomaly_sensor_idx]],
    axis=1,
)
scores = np.max(scores, axis=1)
print(scores.shape)
red_nodes = list(np.where(scores > 0.1)[0])

# %%
G = nx.from_numpy_array(weight_mat)
G.remove_edges_from(nx.selfloop_edges(G))

# %%
anomaly_node_size = 80
default_node_size = 20

central_node_color = "yellow"
anomaly_node_color = "red"
default_node_color = "black"

anomaly_edge_color = "red"
default_edge_color = (0.35686275, 0.20392157, 0.34901961, 0.1)
edges = [set(edge) for edge in G.edges()]
edge_colors = [default_edge_color for edge in edges]

node_colors = [default_node_color for i in range(feature_num)]
node_sizes = [default_node_size for i in range(feature_num)]

node_colors[highest_anomaly_sensor_idx] = central_node_color
node_sizes[highest_anomaly_sensor_idx] = anomaly_node_size
for node in red_nodes:

    if node == highest_anomaly_sensor_idx:
        continue

    node_colors[node] = anomaly_node_color
    node_sizes[node] = anomaly_node_size

    edge_pos = edges.index(set((node, highest_anomaly_sensor_idx.item())))
    edge_colors[edge_pos] = anomaly_edge_color
# %%
pos = nx.spring_layout(G)

x, y = pos[highest_anomaly_sensor_idx.item()]
plt.text(
    x,
    y + 0.15,
    s=feature_map[highest_anomaly_sensor_idx.item()],
    bbox=dict(facecolor=central_node_color, alpha=0.5),
    horizontalalignment="center",
)
print("Central Node:", feature_map[highest_anomaly_sensor_idx.item()])

for node in red_nodes:
    x, y = pos[node]
    plt.text(
        x,
        y + 0.15,
        s=feature_map[node],
        bbox=dict(facecolor=anomaly_node_color, alpha=0.5),
        horizontalalignment="center",
    )

    print("Red Node:", feature_map[node])
nx.draw(G, pos, edge_color=edge_colors, node_color=node_colors, node_size=node_sizes)
# %%
# plot output and true values around the anomaliest anomaly
start = highest_anomaly_sample_idx - 10000
end = highest_anomaly_sample_idx + 10000

output = predict_output["test"]["output"]  # [start:end]
true = predict_output["test"]["data"]  # [start:end]

for i in range(output.shape[1]):
    plt.plot(output[:, i], label="output")
    plt.plot(true[:, i], label="true")
    plt.plot(predict_output["test"]["labels"], label="labels")
    plt.title(feature_map[i])
    plt.legend()
    plt.show()

# %%
for i in range(51):
    plt.plot(predict_output["test"]["data"][:, i], label="true")
    plt.plot(predict_output["test"]["output"][:, i], label="output")
    plt.plot(predict_output["test"]["scores"][:, i], label="scores")
    plt.plot(predict_output["test"]["labels"], label="labels")
    plt.title(feature_map[i])
    plt.legend()
    plt.show()

# %%
