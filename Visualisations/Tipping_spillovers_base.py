# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 15:38:09 2023

@author: everall
"""

import matplotlib.pyplot as plt
import numpy as np

# Create a figure and axis without grid lines
fig, ax = plt.subplots()

# Turn off grid lines
ax.grid(False)

# Define the x and y axis range and values
x = np.linspace(0, 1, 100)
y = x  # 45-degree line

# Plot the 45-degree line
ax.plot(x, y, label='Linear response', color='black')

# Add text and shading
ax.text(0.3, 0.7, 'Spillover', fontsize=12)
ax.fill_between(x, y, 1, color='lightblue')

# Set x and y axis limits
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])

# Add labels and title
ax.set_xlabel('X')
ax.set_ylabel('F(x)')
ax.set_title('Graph of a 45-degree Line')

# Show the legend
ax.legend()

plt.show()
