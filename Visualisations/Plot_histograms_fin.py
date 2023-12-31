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
    return harmonized_balanced_df

def plot_final_adjusted_dual_axis_histogram_with_ecdf(df):
    """Plots the final adjusted enhanced normalized histogram with ECDF overlaid on a secondary y-axis with split legends and grid"""
    palette = sns.color_palette("husl", 2)
    type_palette = {'empirical': palette[0], 'modelling': palette[1]}
    
    # Initialize the figure and the primary axis
    fig, ax1 = plt.subplots(figsize=(10,6))
    
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
    
    # Creating two separate legends and placing them on top of the plot
    fig.legend(ecdf_lines, [line.get_label() for line in ecdf_lines], loc='upper left', bbox_to_anchor=(0,1.1))
    fig.legend(hist_legend_handles, hist_legend_labels, loc='upper right', bbox_to_anchor=(1,1.1))
    
    # Removing the original histogram legend
    ax1.get_legend().remove()
    plt.tight_layout()
    plt.savefig("../Figures/critical_histogram.png", dpi=600, bbox_inches = "tight")
    plt.show()
    
def calculate_95th_percentile(df, column_name, groupby_column):
    """
    Calculates the 95th percentile value of a specified column in the DataFrame,
    grouped by another column.

    :param df: DataFrame, the input DataFrame
    :param column_name: str, the name of the column for which to calculate the 95th percentile value
    :param groupby_column: str, the name of the column by which to group the DataFrame
    :return: DataFrame, a DataFrame with the 95th percentile values grouped by the specified column
    """
    if column_name not in df.columns:
        raise ValueError(f"{column_name} is not a column in the DataFrame")
    
    if groupby_column not in df.columns:
        raise ValueError(f"{groupby_column} is not a column in the DataFrame")

    grouped = df.groupby(groupby_column)
    percentiles_95 = grouped[column_name].quantile(0.95).reset_index()
    
    return percentiles_95


file_path = "../Data/Compiled/Tipping_points_fin_merged_1.csv"
file_path_fin = '../Data/Compiled/Tipping_threshold_plot.csv' 

#df = load_and_process_data(file_path)

percentile_95th_value = calculate_95th_percentile(df, 'tipping_point_c_t', "type")
print("95th percentile value:", percentile_95th_value)

df = pd.read_csv(file_path_fin)

df = df[df["magnitude"] > 0.5]

plot_final_adjusted_dual_axis_histogram_with_ecdf(df)

