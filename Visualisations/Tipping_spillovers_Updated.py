
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 15:38:09 2023
@author: everall
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import random 


#%%

def process_and_plot(df, ax, density_ax):
    """
    Function to process the DataFrame and plot the magnitude column.
    
    Args:
    df (pd.DataFrame): The input DataFrame.
    ax (matplotlib.axes._subplots.AxesSubplot): The axes object to plot on.
    density_ax (matplotlib.axes._subplots.AxesSubplot): The axes object for the density plot.
    """
    # Filter rows where variable column is 'k'
    df_filtered = df[df['variable'] == 'k'].copy()
    
    # Replace commas with decimal points and inequalities with numeric representation in the magnitude column
    df_filtered['magnitude'] = df_filtered['magnitude'].replace(',', '.', regex=False)
 
    df_filtered['magnitude'] = df_filtered['magnitude'].apply(lambda x: random.uniform(0.6, 1) if str(x) in '>0.6' or str(x) in 'nan' else x)
    #df_filtered['magnitude'] = df_filtered['magnitude'].replace('>0.6', rand.uniform(0.6, 1), regex=False)
    
    print(
       df_filtered['magnitude'])
    # Convert magnitude and value columns to numeric
    df_filtered['magnitude'] = pd.to_numeric(df_filtered['magnitude'], errors='coerce')
    df_filtered['value'] = pd.to_numeric(df_filtered['value'], errors='coerce')
    
    # Drop rows where magnitude or value is NaN
    df_filtered = df_filtered.dropna(subset=['magnitude', 'value'])
    
    # Compute the weights for each data point for normalization
    #weights = np.ones_like(df_filtered['value']) / len(df_filtered['value'])
    
    # Plot a density plot of the distribution of lambda variables with normalized y-axis
    #sns.kdeplot(df_filtered['value'], ax=density_ax, fill=True, weights=weights)
    
    # Plot magnitude vs value using seaborn
    sns.scatterplot(x='value', y='magnitude', data=df_filtered, ax=ax, color='black', alpha=0.5, marker='o', label='Data Points')
    
    # Plot a density plot of the distribution of lambda variables
    #sns.displot(df_filtered['value'], ax=density_ax, kind="kde", fill=True )
    sns.kdeplot(df_filtered['value'], ax=density_ax, fill=True)
    density_ax.set_xlim([0, 1])  # Correct x-axis limit to [0, 1]
    density_ax.set_title('Density Plot of $\lambda$')
    density_ax.set_xlabel('$\lambda$')
    density_ax.set_ylabel('Density')


def main():
    # Define the path to the CSV file

    csv_file_path = '../Data/Compiled/The Pareto effect in tipping social networks Tipping Data - Tipping_Data_Raw.csv'
    
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)
    
    # Convert the 'value' column to numeric and filter out the values not in the [0, 1] range
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df = df[(df['value'] >= 0) & (df['value'] <= 1)]
    
    # Set seaborn style
    sns.set(style="whitegrid")

    # Create a figure and axes objects without grid lines
    fig, (ax, density_ax) = plt.subplots(1, 2, figsize=(10, 5))

    # Correctly plot the 45-degree line from (0,0) to (1,1)
    ax.plot([0, 1], [0, 1], 'k--', label='Linear response')  # Correcting the 45-degree line

    # Set x and y axis limits correctly
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])  # Corrected y-axis limit to [0, 1]

    # Add labels and title with the new label names
    ax.set_xlabel('$\lambda$')
    ax.set_ylabel('$n/N$')
    ax.set_title('Critical mass vs cascade magnitude')

    # Process DataFrame and plot on the axes object using seaborn
    process_and_plot(df, ax, density_ax)

    # Adjust layout
    plt.tight_layout()

    # Show the legend
    ax.legend()
    plt.savefig("../Figures/critical_values.png",dpi=600 )
    plt.show()
    return df


if __name__ == "__main__":
    df = main()
