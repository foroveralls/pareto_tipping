# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 06:53:27 2023

@author: everall
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#%%

def load_data():
    tipping_data_raw_df = pd.read_csv("../Data/Compiled/The Pareto effect in tipping social networks Tipping Data - Tipping_Data_Raw.csv")
    tipping_points_fin_df = pd.read_csv("../Data/Compiled/Tipping_points_fin_merged.csv")
    return tipping_points_fin_df, tipping_data_raw_df

#%%



# Process the Data
def process_data(tipping_data_raw_df):
    tipping_data_w_df = tipping_data_raw_df[tipping_data_raw_df['variable'] == 'w'].copy()
    tipping_data_w_df[['lower', 'center', 'upper']] = tipping_data_w_df['value'].apply(lambda x: pd.Series(process_value_column(x)))
    return tipping_data_w_df

# Extract the center and endpoints of the range
def process_value_column(value_str):
    split_values = value_str.split('-')
    if len(split_values) == 1:
        center = float(split_values[0])
        return center, center, center
    if len(split_values) == 2:
        lower = float(split_values[0])
        upper = float(split_values[1])
        center = (lower + upper) / 2
        return lower, center, upper
    return np.nan, np.nan, np.nan

# Set spine properties
def set_spine_properties(ax, alpha, linewidth, color):
    for spine in ax.spines.values():
        spine.set_alpha(alpha)
        spine.set_linewidth(linewidth)
        spine.set_color(color)

# Add caps to the range bars
def add_caps_to_range_bar(ax, lower, upper, y, cap_size=0.2):
    ax.plot([lower, upper], [y, y], color='black')  # Range bar
    ax.plot([lower, lower], [y - cap_size, y + cap_size], color='black')  # Left cap
    ax.plot([upper, upper], [y - cap_size, y + cap_size], color='black')  # Right cap

# Create the Plots
def create_plots(tipping_data_w_df, tipping_points_fin_df):
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    
    # Modified dot plot
    ax = axs[0]
    sns.scatterplot(x='center', y=np.arange(len(tipping_data_w_df)), data=tipping_data_w_df, ax=ax, color='lightblue', edgecolor='black', marker='o')
    for i, (lower, center, upper) in enumerate(zip(tipping_data_w_df['lower'], tipping_data_w_df['center'], tipping_data_w_df['upper'])):
        add_caps_to_range_bar(ax, lower, upper, i)
    set_spine_properties(ax, 1, 1, 'black')
    ax.set_title('Threshold fraction range that allows tipping  "w"')
    ax.set_xlabel('Individual tipping threshold')
    ax.set_ylabel('Index')
    ax.grid(color='lightgrey', linestyle='--', linewidth=0.5)
    
    # Dot plot
    ax = axs[1]
    sns.histplot(x=tipping_points_fin_df['tipping_point_c_t'], ax=ax, color='lightblue', edgecolor='black') # marker='o', jitter=0.2
    ax.set_title('Distribution of Tipping Points c')
    ax.set_xlabel('Tipping Point c')
    ax.set_xlim(0, 1)
    ax.grid(color='lightgrey', linestyle='--', linewidth=0.5)
    
    size = 10,5
    plt.tight_layout()
    plt.rcParams['figure.figsize'] = size
    plt.gcf().set_size_inches(size)
    plt.subplots_adjust(wspace=0.2)
    plt.savefig("../Figures/critical_histograms.png", dpi=800 )
    plt.show()

def main():
    tipping_points_fin_df, tipping_data_raw_df = load_data()
    tipping_data_w_df = process_data(tipping_data_raw_df)
    create_plots(tipping_data_w_df, tipping_points_fin_df)

if __name__ == "__main__":
    main()