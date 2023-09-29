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
    palette = sns.color_palette("husl", 2)
    type_palette = {'empirical': palette[0], 'modelling': palette[1]}
    fig, ax1 = plt.subplots(figsize=(10,6))
    hist = sns.histplot(data=df, x='tipping_point_c_t', hue='type', multiple="dodge", stat='probability', 
                        shrink=.5, palette=type_palette, bins=10, alpha=0.6, ax=ax1)
    ax1.grid(True)
    ax1.set_xlabel('$\lambda$')
    ax1.set_ylabel('Count')
    ax2 = ax1.twinx()
    ecdf_lines = []
    for type_category, color in type_palette.items():
        subset = df[df['type'] == type_category]
        x = np.sort(subset['tipping_point_c_t'])
        y = np.arange(1, len(x) + 1) / len(x)
        line, = ax2.plot(x, y, color=color, label=f"{type_category} ECDF", linestyle='dashed')
        ecdf_lines.append(line)
    ax2.set_ylabel('Proportion (ECDF)')
    #plt.title('Final Adjusted Dual Axis Histogram and ECDF of Tipping Points')
    hist_legend_labels = [label.get_text() for label in hist.legend_.get_texts()]
    hist_legend_handles = hist.legend_.get_patches()
    fig.legend(ecdf_lines + hist_legend_handles, [line.get_label() for line in ecdf_lines] + hist_legend_labels, 
               loc='upper left', bbox_to_anchor=(0,1.1))
    ax1.get_legend().remove()
    plt.tight_layout()
    plt.savefig("../Figures/critical_histogram.png", dpi=600 )
    plt.show()

file_path = "../Data/Compiled/Tipping_points_fin_merged_1.csv"
df = load_and_process_data(file_path)
plot_final_adjusted_dual_axis_histogram_with_ecdf(df)


