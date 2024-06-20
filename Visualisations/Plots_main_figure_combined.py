# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 21:42:04 2023
@author: everall
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def harmonize_ref(ref):
    """Harmonizes the 'ref' column by removing any suffixes after 'traj'"""
    if "traj" in ref:
        return ref.split("_traj")[0]  
    return ref

def load_and_process_data(file_path):
    """Loads and processes the data for the jointplot"""
    df = pd.read_csv(file_path)
    df['harmonized_ref'] = df['ref'].apply(harmonize_ref)
    
    harmonized_balanced_dfs = []
    max_experimental_count = df[df['type'] == 'experimental'].groupby('harmonized_ref').size().max()

    for harmonized_ref, group in df.groupby('harmonized_ref'):
        experimental_group = group[group['type'] == 'experimental']
        modelling_group = group[group['type'] != 'experimental'].copy()
        modelling_group['type'].fillna('modelling', inplace=True)
        
        if len(modelling_group) > max_experimental_count:
            modelling_group = modelling_group.sample(n=max_experimental_count, random_state=42)
        
        harmonized_balanced_group = pd.concat([experimental_group, modelling_group], ignore_index=True)
        harmonized_balanced_dfs.append(harmonized_balanced_group)

    harmonized_balanced_df = pd.concat(harmonized_balanced_dfs, ignore_index=True)
    harmonized_balanced_df['type'] = harmonized_balanced_df['type'].apply(lambda x: 'empirical' if x == 'experimental' else 'modelling')
    harmonized_balanced_df['magnitude'] = harmonized_balanced_df['magnitude'].replace('[\%,]', '', regex=True).apply(pd.to_numeric, errors='coerce')
    
    harmonized_balanced_df["type"] = np.where(harmonized_balanced_df["harmonized_ref"] == "Amato_2012.csv", "empirical", harmonized_balanced_df["type"])
    
    return harmonized_balanced_df

def load_and_process_data_histogram(file_path):
    """Loads, processes, and filters the data for the histogram with ECDF"""
    df = pd.read_csv(file_path)
    df['harmonized_ref'] = df['ref'].apply(harmonize_ref)
    
    harmonized_balanced_dfs = []
    max_experimental_count = df[df['type'] == 'experimental'].groupby('harmonized_ref').size().max()

    for harmonized_ref, group in df.groupby('harmonized_ref'):
        # Separate the group into rows with NaN in 'attribute' or 'value' and rows without NaN in these columns.
        nan_rows = group[group[['attribute', 'value']].isna().any(axis=1)]
        non_nan_rows = group.drop_duplicates(subset=['tipping_point_c_t', 'magnitude'])
        
        # Concatenate the nan_rows and the filtered non_nan_rows back into the group.
        group = pd.concat([nan_rows, non_nan_rows], ignore_index=True)
        
        experimental_group = group[group['type'] == 'experimental']
        modelling_group = group[group['type'] != 'experimental'].copy()
        modelling_group['type'].fillna('modelling', inplace=True)
        
        if len(modelling_group) > max_experimental_count:
            modelling_group = modelling_group.sample(n=max_experimental_count, random_state=42)
        
        harmonized_balanced_group = pd.concat([experimental_group, modelling_group], ignore_index=True)
        harmonized_balanced_dfs.append(harmonized_balanced_group)

    harmonized_balanced_df = pd.concat(harmonized_balanced_dfs, ignore_index=True)
    harmonized_balanced_df['type'] = harmonized_balanced_df['type'].apply(lambda x: 'empirical' if x == 'experimental' else 'modelling')
    harmonized_balanced_df['magnitude'] = harmonized_balanced_df['magnitude'].replace('[\%,]', '', regex=True).apply(pd.to_numeric, errors='coerce')
    harmonized_balanced_df["type"] = np.where(harmonized_balanced_df["harmonized_ref"] == "Amato_2012.csv", "empirical", harmonized_balanced_df["type"])
    
    return harmonized_balanced_df

def plot_jointplot(df, ax):
    """Plots the processed DataFrame"""
    palette = sns.color_palette("husl", 2)
    type_palette = {'empirical': palette[0], 'modelling': palette[1]}
    
    sns.scatterplot(data=df, x='tipping_point_c_t', y='magnitude', hue='type', palette=type_palette, marker='o', s=60, ax=ax)
    ax.plot([0, 1], [0, 1], 'k--', label='Linear response', alpha=0.8)
    ax.tick_params(axis='both', direction='in', length=6, color="black")
    ax.grid(True, linestyle='dashed')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.1])
    ax.set_xlabel('$\lambda$', fontsize=16)
    ax.set_ylabel('$n/N$', fontsize=16)
    ax.legend(loc='lower right', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=16)

def plot_histogram_with_ecdf(df, ax1, ax2):
    """Plots the final adjusted enhanced normalized histogram with ECDF overlaid on a secondary y-axis with split legends and grid"""
    palette = sns.color_palette("husl", 2)
    type_palette = {'empirical': palette[0], 'modelling': palette[1]}
    
    # Plotting an enhanced normalized histogram on the primary y-axis
    hist = sns.histplot(data=df, x='tipping_point_c_t', hue='type', multiple="dodge", 
                        stat='probability', shrink=.5, palette=type_palette, alpha=0.6, ax=ax1)
    
    # Enable the grid
    ax1.grid(True)
    
    # Labeling the primary y-axis
    ax1.set_xlabel('$\lambda$', fontsize=16)
    ax1.set_ylabel('Count', fontsize=16)
    ax1.tick_params(axis='both', which='major', labelsize=16)

    # Format the count labels to 2 decimal places
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))

    # Overlaying ECDF plots on the secondary y-axis
    ecdf_lines = []
    for type_category, color in type_palette.items():
        subset = df[df['type'] == type_category]
        x = np.sort(subset['tipping_point_c_t'])
        y = np.arange(1, len(x) + 1) / len(x)
        line, = ax2.plot(x, y, color=color, label=f"{type_category} ECDF", linestyle='dashed')
        ecdf_lines.append(line)
    
    # Labeling the secondary y-axis and adding title
    ax2.set_ylabel('Proportion (ECDF)', fontsize=16)
    ax2.tick_params(axis='both', which='major', labelsize=16)
    
    # Extracting legend handles and labels for ECDF
    ecdf_legend_handles, ecdf_legend_labels = ax2.get_legend_handles_labels()

    return ecdf_legend_handles, ecdf_legend_labels

# File paths for the data
file_path_jointplot = '../Data/Compiled/Tipping_threshold_plot.csv'  # For the first plot
file_path_histogram = "../Data/Compiled/Tipping_points_fin_merged_1.csv"  # For the second plot

# Load and process data for each plot
df_jointplot = load_and_process_data(file_path_jointplot)
df_histogram = load_and_process_data_histogram(file_path_histogram)

# Create a figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# Plot the first graph
plot_jointplot(df_jointplot, ax1)

# Plot the second graph with histogram and ECDF
ax2_secondary = ax2.twinx()
ecdf_handles, ecdf_labels = plot_histogram_with_ecdf(df_histogram, ax2, ax2_secondary)

# Add labels below the x-axis
ax1.set_title('(a)', loc='center', pad=20, fontsize=16)
ax2.set_title('(b)', loc='center', pad=20, fontsize=16)

# Adding the combined legend to the second plot
ax2.legend(ecdf_handles, ecdf_labels, loc='center right', fontsize=16, handletextpad=1, borderpad=1)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("../Figures/combined_plots.png", dpi=600, bbox_inches="tight")
plt.show()
