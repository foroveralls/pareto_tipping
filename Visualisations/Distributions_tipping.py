# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 17:12:45 2023

@author: everall
"""

import numpy as np
from scipy.stats import norm, expon, beta

import matplotlib.pyplot as plt

# Generate data for each distribution
x = np.linspace(0, 1, 1000)
gaussian = norm.pdf(x, 0.5, 0.1)
exponential = expon.pdf(x, scale=0.2)
bimodal = norm.pdf(x, 0.25, 0.1) + norm.pdf(x, 0.75, 0.1)
gen_beta = beta.pdf(x, 18, 4)

# Create a figure with four subplots
fig, axs = plt.subplots(1, 4, figsize=(12, 4), sharex=True)

# Set the style to black and white with white background
plt.style.use('grayscale')
fig.patch.set_facecolor('white')

# Plot the Gaussian distribution
axs[0].plot(x, gaussian, color='black')
axs[0].set_ylabel('Density')
axs[0].set_title('Reduced plastic usage', loc='center')
axs[0].set_xlim(0, 1)
axs[0].set_yticks([])
axs[0].spines['top'].set_visible(False)
axs[0].spines['right'].set_visible(False)
axs[0].spines['bottom'].set_visible(False)
axs[0].spines['left'].set_visible(False)

# Plot the Exponential distribution
axs[1].plot(x, exponential, color='black')
axs[1].set_title('Frequent flyer levy', loc='center')
axs[1].set_xlim(0, 1)
axs[1].set_yticks([])
axs[1].spines['top'].set_visible(False)
axs[1].spines['right'].set_visible(False)
axs[1].spines['bottom'].set_visible(False)
axs[1].spines['left'].set_visible(False)

# Plot the bimodal distribution
axs[2].plot(x, bimodal, color='black')
axs[2].set_title('Reduction of car usage', loc='center')
axs[2].set_xlim(0, 1)
axs[2].set_yticks([])
axs[2].spines['top'].set_visible(False)
axs[2].spines['right'].set_visible(False)
axs[2].spines['bottom'].set_visible(False)
axs[2].spines['left'].set_visible(False)

# Plot the Generalized Beta distribution
axs[3].plot(x, gen_beta, color='black')
axs[3].set_title('Reduction of Meat consumption', loc='center')
axs[3].set_xlim(0, 1)
axs[3].set_yticks([])
axs[3].spines['top'].set_visible(False)
axs[3].spines['right'].set_visible(False)
axs[3].spines['bottom'].set_visible(False)
axs[3].spines['left'].set_visible(False)

# Add a centered x-axis label to the figure
fig.text(0.5, 0.0008, 'Threshold', ha='center')

# Adjust the spacing between subplots
plt.subplots_adjust(wspace=0.3)

for i in range(len(axs)):
    axs[i].spines['top'].set_visible(False)
    axs[i].patch.set_edgecolor("white")

# Display the plot
plt.show()
