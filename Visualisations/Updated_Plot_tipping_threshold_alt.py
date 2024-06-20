
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 21:42:04 2023

@author: everall
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


#%%
def harmonize_ref(ref):
    """Harmonizes the 'ref' column by removing any suffixes after 'traj'"""
    if "traj" in ref:
        return ref.split("_traj")[0]  
    return ref

def load_and_process_data(file_path):
    """Loads and processes the data from the specified file path"""
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

def plot_data(df):
    """Plots the processed DataFrame"""
    palette = sns.color_palette("husl", 2)
    type_palette = {'empirical': palette[0], 'modelling': palette[1]}
    
    p = sns.jointplot(data=df, x='tipping_point_c_t', y='magnitude', hue='type', palette=type_palette, marker='o', height=8, s=60)
    ax_joint = p.ax_joint
    ax_joint.plot([0, 1], [0, 1], 'k--', label='Linear response', alpha = 0.8)
    ax_joint.tick_params(axis='both', direction='in', length = 6, color = "black")
    ax_joint.grid(True, linestyle='dashed')
    ax_joint.set_xlim([0, 1])
    ax_joint.set_ylim([0, 1.1])
    ax_joint.set_xlabel('$\lambda$')
    ax_joint.set_ylabel('$n/N$')
    
    
    plt.tight_layout()
    plt.show()
    
    # Show the integrated plot with the final modifications
    plt.savefig("../Figures/critical_values.png", dpi=600 )


#%%
file_path = '../Data/Compiled/Tipping_points_fin_merged_1.csv'  
file_path_fin = '../Data/Compiled/Tipping_threshold_plot.csv'  

#df.to_csv(file_path_fin, index=False)

#df = load_and_process_data(file_path)
df = pd.read_csv(file_path_fin)
plot_data(df)

