# %%
import torch
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# print all torch tensor
torch.set_printoptions(threshold=10000)

path_to_data = "/dataslow/fbello/gragod-data/swat/SWaT_data_val.csv"

df = pd.read_csv(path_to_data)

# %%
df = df.iloc[:, 1:-1]
df.head()

# %%
# best_params_ckpt = (
#     "/dataslow/fbello/output/gdn_optimization_swat/gdn/version_87/best.ckpt"
# )
best_params_ckpt = (
    "/home/fbello/GraGOD/output/gdn/gdn_graph_topologies_small/version_0/best.ckpt"
)
from models.gdn.model import GDN_PLModule, GDN


model_pl = GDN_PLModule.load_from_checkpoint(best_params_ckpt, map_location="cuda")
model: GDN = model_pl.model  # type: ignore
# %%


def adj_matrix_to_edge_index(adj_matrix):
    """
    Convert an adjacency matrix to edge_index format (COO format).

    Parameters:
    -----------
    adj_matrix : np.ndarray
        A square adjacency matrix where adj_matrix[i,j] indicates an edge from node i to node j

    Returns:
    --------
    edge_index : np.ndarray
        A 2xN array where N is the number of edges. Each column represents an edge [source, target]
    """
    # Get indices where entries are non-zero (indicating edges)
    rows, cols = np.nonzero(adj_matrix)

    # Stack them to create edge_index
    edge_index = np.vstack((rows, cols))

    return edge_index


def edge_index_to_adj_matrix(edge_index, num_nodes=None):
    """
    Convert edge_index format to adjacency matrix.

    Parameters:
    -----------
    edge_index : np.ndarray
        A 2xN array where N is the number of edges. Each column represents an edge [source, target]
    num_nodes : int, optional
        Number of nodes in the graph. If None, it's inferred from edge_index

    Returns:
    --------
    adj_matrix : np.ndarray
        A square adjacency matrix where adj_matrix[i,j] indicates an edge from node i to node j
    """
    if num_nodes is None:
        num_nodes = edge_index.max() + 1

    # Create empty adjacency matrix
    adj_matrix = np.zeros((num_nodes, num_nodes))

    # Set entries to 1 where edges exist
    adj_matrix[edge_index[0], edge_index[1]] = 1

    return adj_matrix


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


graph = model.learned_graph
G, edge_index = tensor_to_networkx(graph)
print(edge_index.shape)
# %%
adj_matrix = edge_index_to_adj_matrix(edge_index)
for i in range(adj_matrix.shape[0]):
    print(f"Node {i}: {np.sum(adj_matrix[:,i])} incoming edges and {np.sum(adj_matrix[i,:])} outgoing edges")

# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))  # Increased figure size to accommodate labels
pos = nx.spring_layout(G, k=0.4, iterations=50)  # Improved layout parameters
nx.draw(
    G,
    pos=pos,
    with_labels=True,
    labels={node: str(node) for node in G.nodes()},  # Add node numbers as labels
    node_size=300,
    arrows=False,  # Disable arrows to show only lines
    edge_color="gray",
    alpha=0.7,
    width=0.5,
    font_size=10,  # Small font size for readability
)
# plt.title("SWAT Dataset Graph Structure")
plt.show()

# %%
nx.draw(G)
# %%
G.nodes()
# %%
embeddings = model.embedding.weight.cpu().detach().numpy()
# %%
node_names = df.columns.tolist()
embeddings.shape
# %%
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)
# pintar por nombre de color
n_clusters = 10
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(embeddings)

labels = kmeans.labels_
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels)

# Create legend text for each cluster
legend_text = []
for i in range(n_clusters):
    nodes_in_cluster = [node_names[j] for j in range(len(labels)) if labels[j] == i]
    legend_text.append(f'Cluster {i}: {", ".join(nodes_in_cluster)}')

# Plot node names
for i, name in enumerate(node_names):
    plt.annotate(name, (embeddings_2d[i, 0], embeddings_2d[i, 1]), fontsize=8)


# Add legend with cluster memberships
plt.tight_layout()
plt.show()


# %%
import networkx as nx


def get_edge_index(X: torch.Tensor, device: str) -> torch.Tensor:
    """
    Get the edge index of the graph.
    """
    # TODO: load this from each dataset
    # Create a fully connected graph
    import pandas as pd

    path_to_data = "/dataslow/fbello/gragod-data/swat/SWaT_data_val.csv"
    df = pd.read_csv(path_to_data)
    df = df.iloc[:, 1:-1]
    node_names = df.columns[1:-1].tolist()
    # Create edges between nodes with similar names
    edges = []
    for i, name1 in enumerate(node_names):
        for j, name2 in enumerate(node_names):
            # Skip self-loops and ensure names are different
            if i != j:
                # Strip any leading/trailing whitespace and compare
                stripped_name1 = name1.strip()
                stripped_name2 = name2.strip()

                # Check if names are identical except for last character
                if (
                    len(stripped_name1) == len(stripped_name2)
                    and stripped_name1[:-1] == stripped_name2[:-1]
                ):
                    print(f"edge: {name1} -> {name2}")
                    edges.append([i, j])

    # Convert to tensor and transpose to match PyTorch Geometric format
    edge_index = torch.tensor(edges, dtype=torch.long).t().to(device)

    return edge_index


edge_index = get_edge_index(torch.tensor([]), "cuda")
import matplotlib.pyplot as plt

# visualize the graph from the sparse indexes
G = nx.DiGraph()
# Add all nodes first (assuming nodes are numbered from 0 to max index in edge_index)
num_nodes = max(edge_index.max().item() + 1, len(df.columns))
print(f"num_nodes: {num_nodes}")

# Add nodes with labels from node_names
for i in range(num_nodes):
    G.add_node(i, label=node_names[i])

# Add edges from the edge_index tensor
edges = (
    edge_index.cpu().t().numpy()
)  # Convert to numpy and transpose back to [num_edges, 2] shape
G.add_edges_from(edges)

# Detect communities for coloring
communities = list(nx.community.greedy_modularity_communities(G.to_undirected()))
# Create color map
color_map = []
for node in G.nodes():
    for i, community in enumerate(communities):
        if node in community:
            color_map.append(plt.cm.Set3(i / len(communities)))
            break

# Draw the graph with improved layout and colors
plt.figure(figsize=(10, 6))  # Increased figure size to accommodate labels
pos = nx.spring_layout(G, k=0.4, iterations=50)  # Improved layout parameters
nx.draw(
    G,
    pos=pos,
    with_labels=True,
    labels=nx.get_node_attributes(G, "label"),
    node_color=color_map,
    node_size=300,
    arrows=False,  # Disable arrows to show only lines
    edge_color="gray",
    alpha=0.7,
    width=0.5,
    font_size=10,  # Small font size for readability
)
# plt.title("SWAT Dataset Graph Structure")
plt.show()

# %%
print(len(G.edges()))
# %%
print(len(G.nodes()))
# %%
