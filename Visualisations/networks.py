# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 18:55:56 2023

@author: everall
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# Initialize figures and axes for subplots
fig, axes = plt.subplots(1, 5, figsize=(25, 5))
fig.suptitle("Network Types", fontsize=20)

# Generate and plot Small-World Network
small_world = nx.watts_strogatz_graph(100, 4, 0.1)
nx.draw(small_world, ax=axes[0], node_size=50, node_color='blue', with_labels=False)
axes[0].set_title("Small-World Network")

# Generate and plot Barabási-Albert Network
barabasi_albert = nx.barabasi_albert_graph(100, 2)
nx.draw(barabasi_albert, ax=axes[1], node_size=50, node_color='green', with_labels=False)
axes[1].set_title("Barabási-Albert Network")

# Generate and plot Erdős-Rényi Network
erdos_renyi = nx.erdos_renyi_graph(100, 0.05)
nx.draw(erdos_renyi, ax=axes[2], node_size=50, node_color='red', with_labels=False)
axes[2].set_title("Erdős-Rényi Network")

# Generate and plot Regular Random Network
regular_random = nx.random_regular_graph(4, 100)
nx.draw(regular_random, ax=axes[3], node_size=50, node_color='purple', with_labels=False)
axes[3].set_title("Regular Random Network")

# Generate and plot Clustered Lattice
clustered_lattice = nx.watts_strogatz_graph(100, 4, 0)
nx.draw(clustered_lattice, ax=axes[4], node_size=50, node_color='orange', with_labels=False)
axes[4].set_title("Clustered Lattice")

plt.show()
