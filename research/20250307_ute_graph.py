# %%
import torch
from gragod import start_research
from datasets.ute import load_ute_df, load_ute_training_data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import contextily as ctx
from matplotlib.patches import Rectangle
from matplotlib.patches import ConnectionPatch
from matplotlib.lines import Line2D


# %%
df_train, *_ = load_ute_df()
columns = df_train.columns
X_train, X_val, X_test, X_train_labels, X_val_labels, X_test_labels = (
    load_ute_training_data(normalize=True)
)
# %%
path_to_connections_data = "datasets_files/ute/Paramentros_red_Uruguay.xlsx"
df_connections = pd.read_excel(path_to_connections_data)
df_connections.head()


# %%
def extract_name(name):
    # Extract all letters until the first number
    try:
        result = ""
        for char in name:
            if char.isdigit():
                break
            if char.isalpha():
                result += char
        return result
    except Exception as e:
        print(f"Error extracting name for {name}: {e}")
        return None


df_connections["Source"] = df_connections["From Bus  Name"]
df_connections["Target"] = df_connections["To Bus  Name"]
df_connections["Source"] = df_connections["Source"].apply(extract_name)
df_connections["Target"] = df_connections["Target"].apply(extract_name)
# drop None values
df_connections = df_connections[
    df_connections["Source"].notna() & df_connections["Target"].notna()
]
print(df_connections["Source"].unique())
print(df_connections["Target"].unique())
print(columns)

# %%
# Create a mapping from connection names to 3-letter codes
source_mapping = {}
target_mapping = {}

for code in df_train.columns:
    for source in df_connections["Source"].unique():
        if code in source:
            source_mapping[source] = code
    for target in df_connections["Target"].unique():
        if code in target:
            target_mapping[target] = code

# Print mappings to check
print(f"Mapped {len(source_mapping)} source stations")
print(f"Mapped {len(target_mapping)} target stations")


# %%
G_complete = nx.from_pandas_edgelist(
    df_connections, source="Source", target="Target", create_using=nx.Graph()
)
plt.figure(figsize=(20, 20))
pos = nx.spring_layout(G_complete, seed=42)  # Position nodes using spring layout
nx.draw_networkx_nodes(G_complete, pos, node_size=200, node_color="skyblue")
nx.draw_networkx_edges(G_complete, pos, width=1, alpha=0.7)
nx.draw_networkx_labels(G_complete, pos, font_size=20)
plt.title("UTE Power Grid Network")
plt.axis("off")
plt.tight_layout()
plt.show()
# %%
# Create a mapping of 3-letter codes to the original station names
code_to_station = {}
for source, code in source_mapping.items():
    code_to_station[code] = source
for target, code in target_mapping.items():
    code_to_station[code] = target

# Get the valid 3-letter codes that are in our training data
valid_codes = [
    col
    for col in columns
    if col in source_mapping.values() or col in target_mapping.values()
]
print(f"Found {len(valid_codes)} valid station codes in training data")

# Create subgraph with only the nodes that are in the training data
G_sub = nx.Graph()
G_sub.add_nodes_from(valid_codes)

# Add edges between nodes if they are directly connected in the original graph
for code1 in valid_codes:
    for code2 in valid_codes:
        if code1 != code2:
            # Get original station names
            station1 = code_to_station.get(code1)
            station2 = code_to_station.get(code2)

            if station1 and station2:
                # Check if there's a direct edge between the stations in the complete graph
                if G_complete.has_edge(station1, station2):
                    G_sub.add_edge(code1, code2)
                else:
                    # Check for paths through stations not in our training data
                    for path in nx.all_simple_paths(
                        G_complete, station1, station2, cutoff=5
                    ):
                        # Check if all intermediate nodes are not in our training data
                        intermediate_nodes = path[1:-1]
                        if all(
                            node not in code_to_station.values()
                            for node in intermediate_nodes
                        ):
                            G_sub.add_edge(code1, code2)
                            break

# Visualize the subgraph
plt.figure(figsize=(15, 15))
pos_sub = nx.spring_layout(G_sub, seed=42)
nx.draw_networkx_nodes(G_sub, pos_sub, node_size=30, node_color="lightgreen")
nx.draw_networkx_edges(G_sub, pos_sub, width=1.5, alpha=0.7)
nx.draw_networkx_labels(G_sub, pos_sub, font_size=30)
plt.title("UTE Power Grid Subgraph (Training Data Stations Only)")
plt.axis("off")
plt.tight_layout()
plt.show()

# %%
# Print some statistics about the subgraph
print(f"Number of nodes in subgraph: {G_sub.number_of_nodes()}")
print(f"Number of edges in subgraph: {G_sub.number_of_edges()}")
print(f"Density of subgraph: {nx.density(G_sub):.4f}")

# Find connected components
connected_components = list(nx.connected_components(G_sub))
print(f"Number of connected components: {len(connected_components)}")
for i, comp in enumerate(connected_components):
    print(f"Component {i+1} has {len(comp)} nodes")
# %%
from datasets.graph import networkx_to_edge_index

print(len(G_sub.nodes()))
print(len(G_sub.edges()))
edge_index = networkx_to_edge_index(G_sub)
print(edge_index.shape)
print(edge_index.max())
torch.save(edge_index, "datasets_files/ute/edge_index.pt")
# %%
# Load station location data
ute_stations = pd.read_csv("datasets_files/ute/estaciones_ute.csv")
ute_stations.head()

# %%
# Create a geographical visualization of the power grid


# Function to map station codes to coordinates
def create_station_coordinates_mapping(stations_df, valid_codes, code_to_station):
    # Create a mapping from station code to coordinates
    coords_mapping = {}

    for code in valid_codes:
        station_name = code_to_station.get(code)
        if station_name:
            # Find stations with matching code or name
            matching_stations = stations_df[stations_df["CODIGO"] == code]
            if len(matching_stations) == 0:
                # Try partial matches in the name
                matching_stations = stations_df[
                    stations_df["NOMBRE"].str.contains(
                        station_name, case=False, na=False
                    )
                ]

            if len(matching_stations) > 0:
                # Use the first match
                coords_mapping[code] = (
                    matching_stations.iloc[0]["LON"],
                    matching_stations.iloc[0]["LAT"],
                )

    return coords_mapping


# Map station codes to coordinates
station_coords = create_station_coordinates_mapping(
    ute_stations, valid_codes, code_to_station
)
print(f"Found coordinates for {len(station_coords)} out of {len(valid_codes)} stations")

# Create a new graph with only the stations that have coordinates
G_map = nx.Graph()
for node, coords in station_coords.items():
    G_map.add_node(node, pos=coords)

# Add edges between stations that have coordinates
for u, v in G_sub.edges():
    if u in station_coords and v in station_coords:
        G_map.add_edge(u, v)

# %%
# Plot the network on a map
plt.figure(figsize=(14, 16))
ax = plt.gca()
ax.set_axis_off()

# Get positions from node attributes
pos = nx.get_node_attributes(G_map, "pos")

# Draw nodes and edges
nx.draw_networkx_nodes(
    G_map,
    pos,
    node_size=150,
    node_color="black",
    # edgecolor="black",
    linewidths=1,
    alpha=0.9,
)
nx.draw_networkx_edges(G_map, pos, width=1.5, alpha=0.7, edge_color="blue")
# Removed node labels to avoid cluttering the map
# nx.draw_networkx_labels(G_map, pos, font_size=10, font_weight="bold")

# Add basemap
ctx.add_basemap(
    ax,
    crs="EPSG:4326",
    source="OpenStreetMap.Mapnik",
    attribution=False,
    zoom=8,  # type: ignore
)

# Set reasonable map bounds based on node positions
lon_values = [coords[0] for coords in pos.values()]
lat_values = [coords[1] for coords in pos.values()]

# Add some padding around the bounds
padding = 0.5
plt.xlim(min(lon_values) - padding, max(lon_values) + padding)
plt.ylim(min(lat_values) - padding, max(lat_values) + padding)

# Define Montevideo area coordinates
montevideo_bounds = {
    "min_lon": -56.4,  # Include parts of San José
    "max_lon": -56,  # Include parts of Canelones
    "min_lat": -34.95,  # Southern boundary
    "max_lat": -34.75,  # Northern boundary
}

# Add a rectangle to highlight Montevideo area
rect = Rectangle(
    (montevideo_bounds["min_lon"], montevideo_bounds["min_lat"]),
    montevideo_bounds["max_lon"] - montevideo_bounds["min_lon"],
    montevideo_bounds["max_lat"] - montevideo_bounds["min_lat"],
    linewidth=2,
    edgecolor="red",
    facecolor="none",
    zorder=3,
)
ax.add_patch(rect)

# Create a larger inset for Montevideo area with reduced height
axins = plt.gcf().add_axes((0.20, 0.005, 0.18, 0.16))  # (left, bottom, width, height)
axins.set_axis_off()

# Filter nodes in Montevideo area
montevideo_nodes = {
    node: position
    for node, position in pos.items()
    if (
        montevideo_bounds["min_lon"] <= position[0] <= montevideo_bounds["max_lon"]
        and montevideo_bounds["min_lat"] <= position[1] <= montevideo_bounds["max_lat"]
    )
}

# Create a subgraph for Montevideo
G_montevideo = G_map.subgraph(montevideo_nodes.keys())

# Draw nodes and edges in the inset
if montevideo_nodes:
    nx.draw_networkx_nodes(
        G_montevideo,
        montevideo_nodes,
        node_size=100,
        node_color="black",
        linewidths=1,
        alpha=0.9,
        ax=axins,
    )
    nx.draw_networkx_edges(
        G_montevideo,
        montevideo_nodes,
        width=1.5,
        alpha=0.7,
        edge_color="blue",
        ax=axins,
    )
    # Removed node labels in the minimap as well
    # nx.draw_networkx_labels(
    #     G_montevideo, montevideo_nodes, font_size=8, font_weight="bold", ax=axins
    # )

# Set inset extent to Montevideo area
axins.set_xlim(montevideo_bounds["min_lon"], montevideo_bounds["max_lon"])
axins.set_ylim(montevideo_bounds["min_lat"], montevideo_bounds["max_lat"])

# Add basemap to inset
ctx.add_basemap(
    axins,
    crs="EPSG:4326",
    source="OpenStreetMap.Mapnik",
    attribution=False,
    zoom=12,  # type: ignore
)

# Add a border to the inset
axins.set_facecolor("white")
axins.patch.set_alpha(0.7)
for spine in axins.spines.values():
    spine.set_visible(True)
    spine.set_color("black")
    spine.set_linewidth(0.75)

# Add a box around the inset
inset_box = Rectangle(
    (0, 0),
    1,
    1,
    transform=axins.transAxes,
    fill=False,
    edgecolor="black",
    linewidth=1.5,
    zorder=5,
)
axins.add_patch(inset_box)

# Connect the inset to the Montevideo area with lines
con1 = ConnectionPatch(
    xyA=(montevideo_bounds["max_lon"], montevideo_bounds["min_lat"]),
    coordsA=ax.transData,
    xyB=(1, 0),
    coordsB=axins.transAxes,
    linestyle="--",
    color="black",
    alpha=0.7,
    linewidth=1,
)
plt.gcf().add_artist(con1)

con2 = ConnectionPatch(
    xyA=(
        montevideo_bounds["min_lon"],
        montevideo_bounds["max_lat"],
    ),
    coordsA=ax.transData,
    xyB=(1, 1),
    coordsB=axins.transAxes,
    linestyle="--",
    color="black",
    alpha=0.7,
    linewidth=1,
)
plt.gcf().add_artist(con2)

# Explicitly activate the main axes before adding the legend
plt.sca(ax)

# Add a legend in the top right of Uruguay
legend_elements = [
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor="black",
        markersize=25,
        label="Power Station",
    ),
    Line2D([0], [0], color="blue", lw=2, label="Grid Connection"),
]
# Use specific coordinates for legend placement
ax.legend(
    handles=legend_elements,
    fontsize=25,
    bbox_to_anchor=(1.0, 1.0),
    loc="upper right",
    borderaxespad=0.5,
)

plt.tight_layout()
plt.savefig(
    "research/datasets/plots/ute_power_grid_map.pdf", bbox_inches="tight", dpi=1000
)
plt.show()

# %%

# %%
