# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 21:42:04 2023

@author: everall
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



#%%

def load_and_process_data(file_path):
    # Load the data
    df = pd.read_csv(file_path)
    
    # Assign "empirical" to rows where 'type' is "experimental" or "observational", 
    # and to rows associated with "Amato" in the 'ref' column.
    #df.loc[df['ref'].str.contains('Amato'), 'type'] = 'observational'
    df.loc[df['type'].isin(['experimental', 'observational']), 'type'] = 'empirical'
    print(df)
    
    df['tipping_point_c_t'] = df['tipping_point_c_t'].replace('%','', regex=True).astype('float') 
    df['magnitude'] = df['magnitude'].replace('%','', regex=True).astype('float') 
    
    # Assign "modelling" to rows where the 'type' column is NaN.
    df['type'].fillna('modelling', inplace=True)
    
    # Replace NaN values in the 'magnitude' column with 1.
    df['magnitude'].fillna(1, inplace=True)
    
    # Drop rows with NaN values in 'tipping_point_c_t' column for plotting.
    df = df.dropna(subset=['tipping_point_c_t'])
    
    return df


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
    
    # Add labels and title with the new label names to ax_joint
    ax_joint.set_xlabel('Tipping Point ($c_t$)')
    ax_joint.set_ylabel('$n/N$') 
    #ax_joint.set_title('Criical mass vs steady state cascade magnitude')
    
    # Remove axis ticks from the margin plots completely
    p.ax_marg_x.set_yticks([])
    p.ax_marg_y.set_xticks([])
    
    p.ax_marg_y.remove()
    # Remove grids from the margin plots for a more minimalistic look
    p.ax_marg_x.grid(False)
    p.ax_marg_y.grid(False)
    
    
    # Show the legend on ax_joint
    ax_joint.legend(loc = "lower right")
    # Adjust layout
    plt.tight_layout()
    
    # Show the integrated plot with the final modifications
    plt.savefig("../Figures/critical_values.png", dpi=600 )
    plt.show()
    
    
    

def plot_data(df):
    # Creating a color palette based on unique 'type' values
    unique_types = ['empirical', 'modelling']  # Categorized all types into either 'empirical' or 'modelling'
    palette = sns.color_palette("husl", len(unique_types))
    type_palette = dict(zip(unique_types, palette))
    
    # Creating the joint plot
    p = sns.jointplot(data=df, x='tipping_point_c_t', y='magnitude', hue='type', palette=type_palette, marker='o', height=8, s=60)
    
    p.ax_marg_y.remove()
    # Adding a 45-degree line
    p.ax_joint.plot([0, 1], [0, 1], 'k--')
    
    # Limiting the x-axis and y-axis
    p.ax_joint.set_xlim(0, 1)
    p.ax_joint.set_ylim(0, 1.1)
    
    # Customizing the plot
    p.set_axis_labels('Critical mass($c_t$)', '$N/N$', fontsize=12)
    p.fig.suptitle('Magnitude vs Tipping Point with Types', fontsize=14)
    
    plt.tight_layout()
    
    # Show the integrated plot with the final modifications
    plt.savefig("../Figures/critical_values.png", dpi=600 )
    # Displaying the plot
    plt.show()

def main():
    file_path = '../Data/Compiled/Tipping_points_fin_merged_1.csv' 
    df = load_and_process_data(file_path)
    
   
    plot_data_new(df)
    #plot_data(df)
    return df

if __name__ == "__main__":
    test = main()
