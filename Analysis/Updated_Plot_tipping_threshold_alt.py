
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
    if "traj" in ref:
        return ref.split("_traj")[0]  # Keep the string before "_traj"
    return ref

def load_and_process_data(file_path):
    # Load the data
    df = pd.read_csv(file_path)
    
    # Harmonize 'ref' column
    df['harmonized_ref'] = df['ref'].apply(harmonize_ref)
    
    # Initialize an empty list to hold the balanced dataframes for each 'harmonized_ref' group
    harmonized_balanced_dfs = []
    
    # Find the max experimental count within each 'harmonized_ref' group
    max_experimental_count = df[df['type'] == 'experimental'].groupby('harmonized_ref').size().max()
    
    # Group the DataFrame by 'harmonized_ref' and process each group
    for harmonized_ref, group in df.groupby('harmonized_ref'):
        
        # Separate the group into experimental and modelling (including NaN as modelling)
        experimental_group = group[group['type'] == 'experimental']
        modelling_group = group[group['type'] != 'experimental'].copy()
        modelling_group['type'].fillna('modelling', inplace=True)  # Fill NaN values in 'type' column with 'modelling'
        
        # If the number of 'modelling' points is more than max_experimental_count, randomly select max_experimental_count of them
        if len(modelling_group) > max_experimental_count:
            modelling_group = modelling_group.sample(n=max_experimental_count, random_state=42)
        
        # Concatenate the balanced experimental and modelling groups and append to the list
        harmonized_balanced_group = pd.concat([experimental_group, modelling_group], ignore_index=True)
        harmonized_balanced_dfs.append(harmonized_balanced_group)
    
    # Concatenate all the balanced 'harmonized_ref' groups to form the final balanced DataFrame with harmonized 'ref'
    harmonized_balanced_df = pd.concat(harmonized_balanced_dfs, ignore_index=True)
    
    return harmonized_balanced_df

def plot_data_new(df):
    
    # Creating a color palette based on unique 'type' values
    unique_types = ['empirical', 'modelling']  # Categorized all types into either 'empirical' or 'modelling'
    palette = sns.color_palette("husl", len(unique_types))
    type_palette = dict(zip(unique_types, palette))
    
    # Creating the joint plot
    p = sns.jointplot(data=df, x='tipping_point_c_t', y='magnitude', hue='type', palette=type_palette, marker='o', height=8, s=60)
    
    # Access the ax_joint to customize the plot
    ax_joint = p.ax_joint
    
    def set_spine_properties(ax, alpha, linewidth, color):
        for spine in ax.spines.values():
                spine.set_alpha(alpha)
                spine.set_linewidth(linewidth)
                spine.set_color(color)
          
    # Set the properties for all spines in the main plot and the marginal plots
    alpha_value = 1  # Adjust the alpha value
    linewidth_value = 1  # Adjust the linewidth
    color_value = 'black'  # Adjust the color
    
    # Apply the properties to the main plot spines, including the top spine
    set_spine_properties(ax_joint, alpha_value, linewidth_value, color_value)
    
    # Apply the properties to the marginal plot spines
    set_spine_properties(p.ax_marg_x, alpha_value, linewidth_value, color_value)
    set_spine_properties(p.ax_marg_y, alpha_value, linewidth_value, color_value)
    
    # Correctly plot the 45-degree line from (0,0) to (1,1) on ax_joint
    ax_joint.plot([0, 1], [0, 1], 'k--', label='Linear response', alpha = 0.8)
    
    ax_joint.tick_params(axis='both', direction='in', length = 6, color = "black")
    ax_joint.grid(True, linestyle='dashed')
   
    # Set x and y axis limits correctly on ax_joint
    ax_joint.set_xlim([0, 1])
    ax_joint.set_ylim([0, 1.1])
    
    # Add labels and title with the new label names
    ax_joint.set_xlabel('Tipping point (°C)')
    ax_joint.set_ylabel('Magnitude of change (°C)')
    ax_joint.set_title('Threshold and magnitude of tipping points')
    
    plt.show()

# Load and process the data
file_path = "<your_file_path>"
df = load_and_process_data(file_path)

# Plot the data
plot_data_new(df)
