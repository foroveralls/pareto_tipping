# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 00:46:16 2023

@author: everall
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# File paths
filepath = "../Data/Compiled/The Pareto effect in tipping social networks Tipping Data - Tipping_Data.csv"


def to_numeric_magnitude(value):
    try:
        return float(value)
    except:
        if value == '+':
            return 1
        elif value == '-':
            return -1
        return None


def preprocess_data(df):
    effect_magnitude_data = df[df['effect_magnitude'].notna()].copy()
    effect_cascade_data = df[df['effect_cascade_success'].notna()]

    effect_magnitude_data['numeric_magnitude'] = effect_magnitude_data['effect_magnitude'].apply(to_numeric_magnitude)
    
    def magnitude_to_symbol(row):
        if pd.isna(row['effect_magnitude']):
            return None

        # If monotonic is 'N', return '+/-'
        if row['monotonic'] == 'N':
            return '+/-'

        # Convert numeric magnitude to symbols
        magnitude = row['effect_magnitude']
        if '-' in str(magnitude):
            symbol = '-'
        else:
            symbol = '+'

        return symbol
    
    # Apply the function to the dataframe
    effect_magnitude_data['symbol'] = effect_magnitude_data.apply(magnitude_to_symbol, axis=1)
    
    
    # Find values in 'variable' column that have an empty 'grouping term' and occur only once
    mask = (effect_cascade_data.groupby('variable')['variable'].transform('count') == 1) & (effect_cascade_data['grouping_term'].isna())
    mask_1 = (effect_magnitude_data.groupby('variable')['variable'].transform('count') == 1) & (effect_magnitude_data['grouping_term'].isna())
    
    effect_magnitude_data.loc[effect_magnitude_data['grouping_term'].notna(), 'variable'] = effect_magnitude_data['grouping_term']
    # Remove rows with values that meet both conditions
    effect_cascade_data = effect_cascade_data[~mask]
    effect_magnitude_data = effect_magnitude_data[~mask_1]
    
    return effect_magnitude_data, effect_cascade_data

def plot_effect_cascade_success(effect_cascade_data):
    effect_cascade_data['plotting_variable'] = effect_cascade_data['grouping_term'].combine_first(effect_cascade_data['variable'])
    sns.set(style="whitegrid")
    f, ax = plt.subplots(figsize=(6, 6))
    effect_cascade_data = effect_cascade_data[effect_cascade_data['effect_cascade_success'].notna()]
    effect_cascade_data['colors'] = effect_cascade_data['effect_cascade_success'].map({'+': 'g', '-': 'r'}).fillna('b')
    size_data = effect_cascade_data.groupby('plotting_variable').size().reset_index(name='sizes')
    effect_cascade_data = effect_cascade_data.merge(size_data, on='plotting_variable', how='left')
    bins = np.linspace(size_data['sizes'].min(), size_data['sizes'].max(), 5)
    effect_cascade_data['size_category'] = pd.cut(effect_cascade_data['sizes'], bins=bins, include_lowest=True)
    range_labels = [f"{int(bins[i])}-{int(bins[i+1])}" for i in range(len(bins)-1)]
    effect_cascade_data['size_category'] = effect_cascade_data['size_category'].astype(str).map(dict(zip(effect_cascade_data['size_category'].astype(str).unique(), range_labels)))
    plot = sns.scatterplot(x="effect_cascade_success", y="plotting_variable", size="size_category", marker='o',
                           sizes=(50, 500), alpha=0.8, data=effect_cascade_data, ax=ax, legend="brief", hue='colors', palette={'g': 'g', 'r': 'r', 'b': 'b'})
    handles, labels = ax.get_legend_handles_labels()
    handles = handles[5:]
    labels = labels[5:]
    labels.reverse()
    leg = ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 1), title='Count Ranges')
    ax.yaxis.set_tick_params(pad=5)
    ax.set(xlabel='Measure of effect', ylabel='')
    ax.set_title('Social network characteristics and their effect on cascade success')
    ax.set_xlim(-0.5, 1.5)
    plt.tight_layout()
    plt.savefig("../Figures/Cascade_success_final.png", dpi=600)
    plt.show()

def plot_effect_magnitude(effect_magnitude_data):
    sns.set(style="whitegrid")
    plt.figure(figsize=(7, 5))
    ax = plt.gca()
    ax.set_xlim(-0.5, 2.5)
    variables = effect_magnitude_data['variable'].unique()
    symbols = effect_magnitude_data['symbol'].unique()
    y_positions = {variable: i for i, variable in enumerate(variables)}
    color_mapping = {'+': 'g', '-': 'r', '±': 'b'}
    size_mapping = {1.0: 50, 2.0: 100, 3.0: 150}
    jitter = 0.3
    
    for index, row in effect_magnitude_data.iterrows():
        y = y_positions[row['variable']] + np.random.uniform(-jitter, jitter)
        x_jitter = np.random.uniform(-1, 1)  # Adding jitter to the x-axis
        size = size_mapping.get(row['numeric_magnitude'], 50)
        symbol = row['symbol']
        color = color_mapping.get(symbol, 'b')
        sns.scatterplot(x=[symbol], y=[y], size=[size], sizes=(size, size), alpha=0.8, color=color, marker='D', edgecolor='w', linewidth=0.5)
    
    ax.set_yticks(list(y_positions.values()))
    ax.set_yticklabels(list(y_positions.keys()))
    
    legend_elements = [plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='grey', markersize=np.sqrt(size), label=f"{int(size_val)}")
                       for size_val, size in size_mapping.items()]
    
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1,0.8), title='Magnitude')
    plt.title('Social network characteristics and their effect on cascade magnitude')
    plt.xlabel('Effect Magnitude')
    #plt.ylabel('')
    plt.tight_layout()
    plt.savefig("../Figures/Cascade_magnitud_final.png", dpi=600)
    plt.show()

def main():
    df = pd.read_csv(filepath)
    effect_magnitude_data, effect_cascade_data = preprocess_data(df)
    plot_effect_cascade_success(effect_cascade_data)
    plot_effect_magnitude(effect_magnitude_data)  # Ensure 'symbol' column is properly created before this line
    return effect_cascade_data

if __name__ == "__main__":
    test = main()
