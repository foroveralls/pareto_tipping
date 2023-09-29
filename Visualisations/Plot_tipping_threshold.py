
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 15:38:09 2023

@author: everall
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import random 

#%%

def process_and_plot(df):
    """
    Function to process the DataFrame and plot the magnitude column using seaborn's jointplot method.
    
    Args:
    df (pd.DataFrame): The input DataFrame.
    """
    # Filter rows where variable column is 'k'
    df_filtered = df[df['variable'] == 'c'].copy()
    
    # Replace commas with decimal points and inequalities with numeric representation in the magnitude column
    df_filtered['magnitude'] = df_filtered['magnitude'].replace(',', '.', regex=False)
    df_filtered['magnitude'] = df_filtered['magnitude'].replace('>0.6', '0.6', regex=False)
    
    # Convert magnitude and value columns to numeric
    df_filtered['magnitude'] = pd.to_numeric(df_filtered['magnitude'], errors='coerce')
    df_filtered['value'] = pd.to_numeric(df_filtered['value'], errors='coerce')
    df_filtered['magnitude'] = df_filtered['magnitude'].apply(lambda x: random.uniform(0.6, 1) if str(x) in '>0.6' or str(x) in 'nan' else x)
    
    # Drop rows where magnitude or value is NaN
    df_filtered = df_filtered.dropna(subset=['magnitude', 'value'])
    
    # Use seaborn's jointplot to integrate the univariate KDE on the margins of the existing left-hand plot
    joint_plot = sns.jointplot(x='value', y='magnitude', data=df_filtered, kind='scatter', 
                               marginal_kws=dict(fill=True, alpha = 0.8, edgecolor = "black"), 
                                alpha = 1, marker='o', marginal_ticks=False,  height=6) #color='lightblue'
    
    

    # Access the ax_joint to customize the plot
    ax_joint = joint_plot.ax_joint


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
    set_spine_properties(joint_plot.ax_marg_x, alpha_value, linewidth_value, color_value)
    set_spine_properties(joint_plot.ax_marg_y, alpha_value, linewidth_value, color_value)
    
    # Correctly plot the 45-degree line from (0,0) to (1,1) on ax_joint
    ax_joint.plot([0, 1], [0, 1], 'k--', label='Linear response', alpha = 0.8)
    
    ax_joint.tick_params(axis='both', direction='in', length = 6, color = "black")
    ax_joint.grid(True, linestyle='dashed')
   
    # Set x and y axis limits correctly on ax_joint
    ax_joint.set_xlim([0, 1])
    ax_joint.set_ylim([0, 1])
    
    # Add labels and title with the new label names to ax_joint
    ax_joint.set_xlabel('$\lambda$')
    ax_joint.set_ylabel('$n/N$')
    ax_joint.set_title('Criical mass vs steady state cascade magnitude')
    
    # Remove axis ticks from the margin plots completely
    joint_plot.ax_marg_x.set_yticks([])
    joint_plot.ax_marg_y.set_xticks([])
    
    # Remove grids from the margin plots for a more minimalistic look
    joint_plot.ax_marg_x.grid(False)
    joint_plot.ax_marg_y.grid(False)
    
    
    # Show the legend on ax_joint
    ax_joint.legend(loc = "lower right")
    # Adjust layout
    plt.tight_layout()
    
    # Show the integrated plot with the final modifications
    plt.savefig("../Figures/critical_values.png", dpi=600 )
    plt.show()


def main():
    # Define the path to the CSV file
    csv_file_path = '../Data/Compiled/Tipping_points_fin_merged_1.csv'
    

    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)
    
    # Convert the 'value' column to numeric and filter out the values not in the [0, 1] range
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df = df[(df['value'] >= 0) & (df['value'] <= 1)]
    
    # Set seaborn style
    sns.set_style("ticks")
    #sns.set(style="whitegrid")
    sns.set_style({"xtick.direction": "in","ytick.direction": "in"})
    
    
    # Process DataFrame and plot using seaborn's jointplot method
    process_and_plot(df)
    

  
    return df


if __name__ == "__main__":
    plot_data = main()
