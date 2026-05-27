# %%
import os

import numpy as np
# %%
os.getcwd()
#%%
day_str = "20190501"
city_str = "Nancy"
app_str = "YouTube"
path_to_datasets = "../../datasets_files/mihaela"
city_dims = (151, 165)

traffic_file_dn = os.path.join(
    path_to_datasets,
    city_str,
    app_str,
    day_str,
    f"{city_str}_{app_str}_{day_str}_DL.txt",
)

# %%
import datetime

import pandas as pd

# let's make a list of 15 min time intervals to use as column names
day = datetime.datetime.strptime(day_str, "%Y%m%d")
times = [day + datetime.timedelta(minutes=15 * i) for i in range(96)]
times_str = [t.strftime("%H:%M") for t in times]

column_names = ["tile_id"] + times_str

# let's load the data of the downlink traffic
df_traffic_dn = pd.read_csv(traffic_file_dn, sep=" ", names=column_names)
df_traffic_dn

# %%

n_rows, n_cols = city_dims

# the first dimension is the time, the second and third are the rows and columns (spatial dimensions)
city_traffic = np.zeros((len(times_str), n_rows, n_cols))

# fill the array with the traffic values
for _, row in df_traffic_dn.iterrows():
    # FEDE: so, apparently the position in the map is encoded in the tile_id, this is the way to reconstruct it
    tile_id = row["tile_id"]
    row_index = int(tile_id // n_cols)
    col_index = int(tile_id % n_cols)

    traffic_values = np.array(row[times_str])
    city_traffic[:, row_index, col_index] = traffic_values
# %%
import matplotlib.cm as cm
import matplotlib.colors as colrs
from matplotlib import pyplot as plt

cmap_traffic = cm.get_cmap("Spectral_r").copy()
cmap_traffic.set_under("w", 0)
norm_traffic = colrs.LogNorm(vmin=1e0, vmax=1e7)

fig, axs = plt.subplots(4, 6, figsize=(60, 40))
axs = axs.flatten()

for hour in range(24):
    ax = axs[hour]

    # recall that we have 15 min intervals, so we need to multiply the hour by 4
    city_traffic_time = city_traffic[hour * 4]

    ax.imshow(city_traffic_time, origin="lower", cmap=cmap_traffic, norm=norm_traffic)
    ax.set_title(f"{str(hour).zfill(2)}:00", fontsize=30)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")

ax = fig.add_axes([0.95, 0.25, 0.02, 0.5])
sm = plt.cm.ScalarMappable(cmap=cmap_traffic, norm=norm_traffic)
sm.set_array([])
clb = plt.colorbar(sm, cax=ax, orientation="vertical")
clb.set_label("Traffic DN", rotation=90, fontsize=40, labelpad=50)
clb.ax.tick_params(labelsize=30)
clb.ax.xaxis.set_ticks_position("default")

plt.show()
# %%

# %%
# we are now going to construct a matrix for all the days
# the idea will be to have a 2D where the id is the date (with 15 min intervals) and the value is the traffic
# we are gonna have one column for each tile_id

days_str = [str(day) for day in range(20190501, 20190532)]
print(days_str)
city_str = "Nancy"
app_str = "YouTube"
path_to_datasets = "../../datasets_files/mihaela"
city_dims = (151, 165)

all_trafic_files_service = []
for day_str in days_str:
    all_trafic_files_service.append(
        os.path.join(
            path_to_datasets,
            city_str,
            app_str,
            day_str,
            f"{city_str}_{app_str}_{day_str}_DL.txt",
        )
    )
# %%
all_dfs = []

for day_str, traffic_file_dn in zip(days_str, all_trafic_files_service):
    day = datetime.datetime.strptime(day_str, "%Y%m%d")
    times = [day + datetime.timedelta(minutes=15 * i) for i in range(96)]
    day_times_str = [t.strftime("%Y-%m-%d %H:%M") for t in times]

    column_names = ["tile_id"] + day_times_str

    # Load the data of the downlink traffic
    df_traffic_dn = pd.read_csv(traffic_file_dn, sep=" ", names=column_names)

    # Pivot the dataframe
    df_pivot = df_traffic_dn.melt(
        id_vars=["tile_id"], var_name="time", value_name="traffic"
    )
    df_pivot = df_pivot.pivot(
        index="time", columns="tile_id", values="traffic"
    ).reset_index()

    # Append the pivoted dataframe to the list
    all_dfs.append(df_pivot)

# Concatenate all dataframes along the index (time)
final_df = pd.concat(all_dfs, axis=0)
final_df

# %%
# %%
import networkx as nx
import random

# Initialize the graph
G = nx.Graph()
for index, row in final_df[:10].iterrows():
    for tile_id, traffic_value in row.items():
        if tile_id == 'time':
            continue
        tile_id = int(tile_id)
        row_index = tile_id // n_cols
        col_index = tile_id % n_cols
        node = (row_index, col_index)
        
        # Add node with traffic value as attribute
        G.add_node(node, traffic=traffic_value)

        for adj in [(row_index - 1, col_index), (row_index + 1, col_index), (row_index, col_index - 1), (row_index, col_index + 1)]:
            if adj in G.nodes:
                G.add_edge(node, adj)


# Randomly sample a subset of nodes to visualize
sample_size = 1000  # Number of nodes to sample, adjust as necessary
sampled_nodes = random.sample(list(G.nodes), sample_size)
# Create a subgraph with the sampled nodes and their immediate neighbors
print(G)
subG = G.subgraph(sampled_nodes).copy()
for node in sampled_nodes:
    try:
        subG.nodes[node]["traffic"]
    except:
        print("no traffic")

for node in sampled_nodes:
    # Look at each neighbor in the original graph
    for neighbor in G.neighbors(node):
        if neighbor in sampled_nodes:
            # Add edge only if both nodes are in the sampled set
            subG.add_edge(node, neighbor)
print(subG)
# Visualize the subgraph
pos = {node: (node[1], -node[0]) for node in subG.nodes}  # Position nodes based on their grid coordinates
nx.draw(subG, pos, node_color=[subG.nodes[n]['traffic'] for n in subG.nodes], cmap=plt.cm.viridis)
plt.show()
# %%
