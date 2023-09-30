# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 02:25:12 2023

@author: everall
"""

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

#%%

# Define logistic function
def logistic(x, x0=5, k=1):
    return 1 / (1 + np.exp(-k * (x - x0)))

# Define the second derivative of the logistic function
def second_derivative_logistic(x, x0=5, k=1):
    fx = logistic(x, x0, k)
    return k * fx * (1 - fx) * (1 - 2 * fx) * k

# Define x range for the first plot
x1 = np.linspace(0, 1, 100)
# Define Pareto CDF
pareto_cdf = 1 - (0.1 / x1) ** 2

# Define x range for the second plot
x2 = np.linspace(0, 10, 100)

# Initialize the figure and axes with adjusted width ratio

#fig = plt.figure(figsize=(12, 6))
fig = plt.figure(figsize=(10, 5))
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])  # Set the width ratio to 1:1
ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1], sharey=ax0)  # Share the y-axis with the first plot

# First plot
sns.lineplot(x=x1, y=x1, ax=ax0, color='black', linestyle='--')
sns.lineplot(x=x1, y=pareto_cdf, ax=ax0, color='black', linestyle='-')
ax0.fill_between(x1, 1, x1, color='blue', alpha=0.4)
ax0.set_xlabel('$λ$')
ax0.set_ylabel('$F(λ) _t\\to \\infty$')
ax0.legend(['Linear response', 'Pareto CDF', 'Tipping Zone'], loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fontsize='small')
ax0.set_xlim([0, 1.1])
ax0.set_ylim([0, 1.1])
ax0.grid(False)
ax0.spines['top'].set_visible(False)  # Remove the top spine

# Second plot
sns.lineplot(x=x2, y=logistic(x2), ax=ax1, color='blue', linestyle='-')
sns.lineplot(x=x2, y=second_derivative_logistic(x2), ax=ax1, color='red', linestyle='--')
ax1.axhline(0.2, color='black', linestyle='--')
ax1.axhline(0.8, color='black', linestyle='--')
ax1.text(0.2, 0.13, 'Triggering Phase', fontsize=8, va='center', ha='left')
ax1.text(0.2, 0.5, 'Tipping Phase', fontsize=8, va='center', ha='left')
ax1.text(0.2, 0.9, 'Manifestation Phase', fontsize=8, va='center', ha='left')
ax1.set_xlabel('$t$')
ax1.set_ylabel('$F(t)$')  # Add back the y label
ax1.legend(['Adopters, $F(t)$', "Second Derivative, $F''(t)$"], loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, fontsize='small')
ax1.set_xlim([0, 10])
ax1.set_ylim([-0.1, 1.1])
ax1.grid(False)
ax1.spines['top'].set_visible(False)  # Remove the top spine

# Adjust space between plots and show the figure
plt.subplots_adjust(wspace=0.2)
plt.tight_layout()
plt.savefig("../Figures/concept_tipping_1.png", dpi=800 )
plt.show()

