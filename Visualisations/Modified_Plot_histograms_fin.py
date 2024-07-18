# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 05:43:51 2023

@author: everall
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def harmonize_ref(ref):
    if "traj" in ref:
        return ref.split("_traj")[0]  
    return ref

def load_and_process_data(file_path):
    """Loads, processes, and filters the data from the specified file path"""
    df = pd.read_csv(file_path)
    df['harmonized_ref'] = df['ref'].apply(harmonize_ref)
    
    harmonized_balanced_dfs = []
    max_experimental_count = df[df['type'] == 'experimental'].groupby('harmonized_ref').size().max()

    for harmonized_ref, group in df.groupby('harmonized_ref'):
        # Separate the group into rows with NaN in 'attribute' or 'value' and rows without NaN in these columns.
        nan_rows = group[group[['attribute', 'value']].isna().any(axis=1)]
        non_nan_rows = group.dropna(subset=['attribute', 'value'])
        
        # For non-NaN rows, drop duplicates based on 'tipping_point_c_t' and 'magnitude'.
        non_nan_rows = non_nan_rows.drop_duplicates(subset=['tipping_point_c_t', 'magnitude'])
        
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
    harmonized_balanced_df['magnitude'] = harmonized_balanced_df['magnitude'].replace('[\\%,]', '', regex=True).apply(pd.to_numeric, errors='coerce')
    harmonized_balanced_df["type"] = np.where(harmonized_balanced_df["harmonized_ref"] == "Amato_2012.csv", "empirical", harmonized_balanced_df["type"])
    
    return harmonized_balanced_df

def plot_final_adjusted_dual_axis_histogram_with_ecdf(df):
    """Plots the final adjusted enhanced normalized histogram with ECDF overlaid on a secondary y-axis with split legends and grid"""
    palette = sns.color_palette("husl", 2)
    type_palette = {'empirical': palette[0], 'modelling': palette[1]}
    
    # Initialize the figure and the primary axis
    fig, ax1 = plt.subplots(figsize=(9.4, 6.8))
    
    # Plotting an enhanced normalized histogram on the primary y-axis
    hist = sns.histplot(data=df, x='tipping_point_c_t', hue='type', multiple="dodge", 
                        stat='probability', shrink=.5, palette=type_palette, bins=10, 
                        alpha=0.6, ax=ax1)
    
    # Enable the grid
    ax1.grid(True)
    
    # Labeling the primary y-axis
    ax1.set_xlabel('$\lambda$')
    ax1.set_ylabel('Count')
    
    # Initialize the secondary y-axis
    ax2 = ax1.twinx()
    
    # Overlaying ECDF plots on the secondary y-axis
    ecdf_lines = []
    for type_category, color in type_palette.items():
        subset = df[df['type'] == type_category]
        x = np.sort(subset['tipping_point_c_t'])
        y = np.arange(1, len(x) + 1) / len(x)
        line, = ax2.plot(x, y, color=color, label=f"{type_category} ECDF", linestyle='dashed')
        ecdf_lines.append(line)
    
    # Labeling the secondary y-axis and adding title
    ax2.set_ylabel('Proportion (ECDF)')
    
    # Extracting legend handles and labels for histogram
    hist_legend_labels = [label.get_text() for label in hist.legend_.get_texts()]
    hist_legend_handles = hist.legend_.get_patches()
    
    # Creating separate legend for ECDF and positioning it
    fig.legend(ecdf_lines, [line.get_label() for line in ecdf_lines], loc='center',  bbox_to_anchor=[0.74, 0.53])
    
    # Removing the original histogram legend
    ax1.get_legend().remove()
    plt.tight_layout()
    plt.savefig("../Figures/critical_histogram.png", dpi=600, bbox_inches = "tight")
    plt.show()

file_path = '../Data/Compiled/Tipping_threshold_plot.csv'

df = pd.read_csv(file_path)
#df = load_and_process_data(file_path)
#df.to_csv(file_path_fin, index=False
df_unique = df.drop_duplicates(subset=['tipping_point_c_t', 'magnitude', 'type'])

plot_final_adjusted_dual_axis_histogram_with_ecdf(df_unique)
