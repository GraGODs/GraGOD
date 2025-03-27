# %%
from gragod.start_research import *
from datasets.telco import load_telco_training_data
from datasets.swat import load_swat_training_data
from sklearn.preprocessing import normalize
from sklearn import linear_model

# %%
dataset_name = "telco"
if dataset_name == "swat":
    X_train, X_val, X_test, X_train_labels, X_val_labels, X_test_labels = (
        load_swat_training_data(down_len=100, normalize=True)
    )
else:
    X_train, X_val, X_test, X_train_labels, X_val_labels, X_test_labels = (
        load_telco_training_data(normalize=True)
    )

# %%
# infer a graph topology from the training data
import torch
import numpy as np
from scipy.stats import pearsonr
import networkx as nx
import matplotlib.pyplot as plt
from datasets.graph import networkx_to_edge_index


def infer_meinshausen_graph(X_train, alpha=0.01, epsilon=1e-5):
    """
    Infer graph topology using Meinshausen-Buhlmann method.

    Args:
        X_train: Training data tensor
        alpha: Regularization parameter for Lasso
        max_iter: Maximum number of iterations for Lasso
        epsilon: Threshold for considering an edge

    Returns:
        torch.Tensor: Edge index tensor for GNN
    """
    X = X_train.cpu().numpy()
    p = X.shape[1]

    # Normalize the data
    Xn = X
    # Estimate the adjacency matrix using Meinshausen-Buhlmann method
    B = np.zeros((p, p))
    for j in range(p):
        y = Xn[:, j]
        X_temp = np.delete(Xn, j, axis=1)
        reg = linear_model.Lasso(alpha=alpha, max_iter=10000)
        reg.fit(X_temp, y)
        B[:, j] = np.insert(reg.coef_, j, 0)

    # Threshold small values
    B[np.abs(B) < epsilon] = 0

    # Create a graph from the adjacency matrix
    G = nx.Graph()
    for i in range(p):
        G.add_node(i)

    for i in range(p):
        for j in range(i + 1, p):
            if B[i, j] != 0 or B[j, i] != 0:
                G.add_edge(i, j)

    # Convert to PyTorch edge_index format for GNN
    edge_index = networkx_to_edge_index(G)

    print(
        f"Inferred graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges"
    )
    return edge_index


def infer_meinshausen_graph_inverse(X_train, alpha=0.01, epsilon=9.9e4):
    """
    Infer graph topology using Meinshausen-Buhlmann method.

    Args:
        X_train: Training data tensor
        alpha: Regularization parameter for Lasso
        max_iter: Maximum number of iterations for Lasso
        epsilon: Threshold for considering an edge

    Returns:
        torch.Tensor: Edge index tensor for GNN
    """
    X = X_train.cpu().numpy()
    p = X.shape[1]

    # Normalize the data
    Xn = X
    # Estimate the adjacency matrix using Meinshausen-Buhlmann method
    B = np.zeros((p, p))
    for j in range(p):
        y = Xn[:, j]
        X_temp = np.delete(Xn, j, axis=1)
        reg = linear_model.Lasso(alpha=alpha, max_iter=10000)
        reg.fit(X_temp, y)
        B[:, j] = np.insert(reg.coef_, j, 0)

    # Threshold small values
    B = 1.0 / (np.abs(B) + 1e-5)
    B[np.abs(B) < epsilon] = 0

    # Create a graph from the adjacency matrix
    G = nx.Graph()
    for i in range(p):
        G.add_node(i)

    for i in range(p):
        for j in range(i + 1, p):
            if B[i, j] != 0 or B[j, i] != 0:
                G.add_edge(i, j)

    edge_index = networkx_to_edge_index(G)

    print(
        f"Inferred graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges"
    )
    return edge_index


# %%
alpha = 0.0005 if dataset_name == "swat" else 0.00004
edge_index_mb = infer_meinshausen_graph(
    X_train,
    alpha=alpha,
)
alpha_i = 0.0000001 if dataset_name == "swat" else 0.00004
edge_index_mb_inverse = infer_meinshausen_graph_inverse(
    X_train,
    alpha=alpha_i,
)


from datasets.graph import edge_index_to_networkx, networkx_to_edge_index

torch.save(edge_index_mb, f"datasets_files/{dataset_name}/edge_index_meinshausen.pt")
torch.save(
    edge_index_mb_inverse,
    f"datasets_files/{dataset_name}/edge_index_meinshausen_inverse.pt",
)
G_mb = edge_index_to_networkx(edge_index_mb)
G_mb_inverse = edge_index_to_networkx(edge_index_mb_inverse, directed=True)

import networkx as nx

nx.draw(G_mb, with_labels=True)
plt.show()

plt.figure(figsize=(10, 10))
nx.draw(G_mb_inverse, with_labels=True)
plt.show()

# %%
print(torch.load(f"datasets_files/{dataset_name}/edge_index_random.pt").shape)
print(f"edge_index_mb: {edge_index_mb.shape}")
print(f"edge_index_mb_inverse: {edge_index_mb_inverse.shape}")
# %%
import networkx as nx
import matplotlib.pyplot as plt
import torch
from datasets.graph import edge_index_to_networkx
from datasets.telco import load_telco_training_data, load_telco_df
from datasets.swat import load_swat_training_data
from datasets.swat import load_swat_df


def pretty_plot_telco_graph(edge_index, nodes_names, file_name, k=1.3):
    G = edge_index_to_networkx(edge_index)
    plt.figure(figsize=(5, 5))

    # Create a complete set of nodes from 0 to 11 (representing 1 to 12)
    all_nodes = list(range(len(nodes_names)))
    # Add any missing nodes to the graph
    for node in all_nodes:
        if node not in G.nodes():
            G.add_node(node)

    # Set uniform node size and color
    node_color = "skyblue"
    node_size = 800

    # Use a layout that keeps nodes closer together
    # Reduce k parameter to decrease separation between nodes
    pos = nx.spring_layout(G, seed=42, k=k)  # Lower k value brings nodes closer

    # Create a mapping from node indices to node names
    labels = {node: nodes_names[node] for node in G.nodes()}

    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color=node_color,
        node_size=node_size,
        font_weight="bold",
        font_size=12,
        labels=labels,  # Use the node names as labels
    )

    # Add padding by adjusting the axis limits
    plt.margins(0.15)  # Add 20% padding around the plot

    plt.savefig(
        f"datasets_files/telco_v1/{file_name}.pdf", bbox_inches="tight", dpi=1000
    )
    plt.show()


edge_index_mb = torch.load("datasets_files/telco_v1/edge_index_meinshausen.pt")
edge_index_random = torch.load("datasets_files/telco_v1/edge_index_random.pt")

edge_index_mb_inverse = torch.load(
    "datasets_files/telco_v1/edge_index_meinshausen_inverse.pt"
)
# edge_index_swat_mb = torch.load("datasets_files/swat/edge_index_meinshausen.pt")
# edge_index_swat_topology = torch.load("datasets_files/swat/edge_index_topology.pt")
# edge_index_swat_random = torch.load("datasets_files/swat/edge_index_random.pt")
# edge_index_swat_mb_inverse = torch.load(
#     "datasets_files/swat/edge_index_meinshausen_inverse.pt"
# )
nodes_names = load_telco_df()[0].columns[1:].tolist()
pretty_plot_telco_graph(edge_index_mb, nodes_names, "edge_index_meinshausen", k=0.7)

pretty_plot_telco_graph(
    edge_index_mb_inverse, nodes_names, "edge_index_meinshausen_inverse"
)
pretty_plot_telco_graph(edge_index_random, nodes_names, "edge_index_random")


nodes_names_swat = load_swat_df()[0].columns.tolist()
# %%
# torch.save(
#     edge_index_mb_inverse, "datasets_files/telco_v1/edge_index_meinshausen_inverse.pt"
# )


# %%
def pretty_plot_swat_graph(edge_index, nodes_names, file_name):
    G = edge_index_to_networkx(edge_index)
    plt.figure(figsize=(10, 10))

    # Create a complete set of nodes from 0 to 11 (representing 1 to 12)
    node_labels = {i: nodes_names[i] for i in range(len(nodes_names))}

    # Group nodes by their prefix (everything except last 3 characters)
    node_groups = {}
    for i, name in enumerate(nodes_names):
        stripped_name = name.strip()
        # Group by the third-to-last character ([-3])
        group_key = stripped_name[-3]
        if group_key not in node_groups:
            node_groups[group_key] = []
        node_groups[group_key].append(i)

    # Set uniform node size and color
    node_color = "skyblue"
    node_size = 800

    # Use a layout that keeps nodes closer together
    # Reduce k parameter to decrease separation between nodes
    pos = nx.spring_layout(G, seed=42, k=1.3)  # Lower k value brings nodes closer

    # Create a mapping from node indices to node names
    labels = {node: nodes_names[node] for node in G.nodes()}

    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color=node_color,
        node_size=node_size,
        font_weight="bold",
        font_size=12,
        labels=labels,  # Use the node names as labels
    )
    # plt.savefig(
    #     f"datasets_files/swat/{file_name}.pdf", bbox_inches="tight", dpi=1000
    # )
    plt.show()


pretty_plot_swat_graph(
    edge_index_swat_mb_inverse, nodes_names_swat, "edge_index_meinshausen_inverse"
)
