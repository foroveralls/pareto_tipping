# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 17:12:45 2023

@author: everall
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import beta, norm
import seaborn as sns


def plot_left_skewed(ax):
    x = np.linspace(0, 1, 1000)
    ax.plot(x, beta.pdf(x, 8, 1), color='black', linewidth=0.5)
    ax.fill_between(x, beta.pdf(x, 8, 1), where=(x >= 0.85) & (x <= 0.95), color='gray', alpha=0.5)
    ax.set_title('(a)', loc='center')
    ax.grid(False)


def plot_normal(ax):
    x_low = norm.ppf(0.001)
    x_high = norm.ppf(0.999)
    x = np.linspace(x_low, x_high, 1000)
    x_transformed = (x - x_low) / (x_high - x_low)
    ax.plot(x_transformed, norm.pdf(x, 0, 1), color='black', linewidth=0.5)
    ax.fill_between(x_transformed, norm.pdf(x, 0, 1), where=(x_transformed >= 0.25) & (x_transformed <= 0.75), color='gray', alpha=0.5)
    ax.set_title('(b)', loc='center')
    ax.grid(False)


def plot_right_skewed(ax):
    x = np.linspace(0, 1, 1000)
    ax.plot(x, beta.pdf(x, 1, 8), color='black', linewidth=0.5)
    ax.fill_between(x, beta.pdf(x, 1, 8), where=(x >= 0.05) & (x <= 0.15), color='gray', alpha=0.5)
    ax.set_title('(c)', loc='center')
    ax.grid(False)


def main():
    sns.set(style="whitegrid")
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    
    plot_left_skewed(axs[0])
    plot_normal(axs[1])
    plot_right_skewed(axs[2])
    
    for ax in axs:
        ax.set_yticklabels([])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        



if __name__ == "__main__":
    main()

    plt.tight_layout()
    plt.savefig("../Figures/ditributinons final.png", dpi=600)
    plt.show()






# Holme-Kim network
# Scale free 
# Watts-Strogatz