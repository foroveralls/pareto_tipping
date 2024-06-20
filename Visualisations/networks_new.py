# -*- coding: utf-8 -*-
"""
Created on Tue Jun node_size8 node_size4:42:3node_size 2024

@author: Jordan
"""
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 14:42:31 2024

@author: Jordan
"""
# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 14:42:31 2024

@author: Jordan
"""
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from netgraph import Graph
import pickle

# random_state = np.random.get_state()
# with open("random_state.pkl", "wb") as f:
#     pickle.dump(random_state, f)


# Load the random state
with open("random_state.pkl", "rb") as f:
    random_state = pickle.load(f)

np.random.set_state(random_state)


# General parameters
NETWORK_SIZE = 20
NODE_SIZE = 3
EDGE_WIDTH = 1

# Set the font to sans-serif
plt.rcParams.update({'font.size': 16, 'font.family': 'sans-serif'})

# Initialize figures and axes for subplots
fig, axes = plt.subplots(1, 5, figsize=(18, 8))

# Generate and plot Small-World Network
small_world = nx.watts_strogatz_graph(NETWORK_SIZE, 4, 0.2)
Graph(small_world, ax=axes[0], node_size=NODE_SIZE, edge_width=EDGE_WIDTH, node_edge_width=0.5) 

# Generate and plot Barabási-Albert Network
barabasi_albert = nx.barabasi_albert_graph(NETWORK_SIZE, 2)
Graph(barabasi_albert, ax=axes[1], node_size=NODE_SIZE, edge_width=EDGE_WIDTH, node_edge_width=0.5)

# Generate and plot Erdős-Rényi Network
erdos_renyi = nx.erdos_renyi_graph(NETWORK_SIZE, 0.3)
Graph(erdos_renyi, ax=axes[2], node_size=NODE_SIZE, edge_width=EDGE_WIDTH, node_edge_width=0.5)

# Generate and plot Regular Random Network
regular_random = nx.random_regular_graph(4, NETWORK_SIZE)
Graph(regular_random, ax=axes[3], node_size=NODE_SIZE, edge_width=EDGE_WIDTH, node_edge_width=0.5)

# Generate and plot Clustered Lattice with ring layout
clustered_lattice = nx.watts_strogatz_graph(NETWORK_SIZE, 4, 0)
Graph(clustered_lattice, ax=axes[4], node_size=NODE_SIZE, edge_width=EDGE_WIDTH, node_edge_width=0.5, node_layout="circular")

y = 0.21
# Set titles using fig.text
fig.text(0.1, y, "Small-World", ha='center', fontsize=16)
fig.text(0.3, y, "Barabási-Albert", ha='center', fontsize=16)
fig.text(0.5, y, "Erdős-Rényi", ha='center', fontsize=16)
fig.text(0.7, y, "Regular Random", ha='center', fontsize=16)
fig.text(0.9, y, "Clustered Lattice", ha='center', fontsize=16)

plt.tight_layout()
plt.savefig("../Figures/networks.png", dpi=300, bbox_inches = 'tight')
plt.show()

