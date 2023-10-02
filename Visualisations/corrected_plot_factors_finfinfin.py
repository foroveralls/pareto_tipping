# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 05:14:38 2023

@author: everall
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# File paths
filepath = "../Data/Compiled/The Pareto effect in tipping social networks Tipping Data - Tipping_Data.tsv"

def to_numeric_magnitude(value):
    try:
        return float(value)
    except:
        if value == '+':
            return 1
        elif value == '-':
            return -1
        return None

def magnitude_to_symbol(row):
    if pd.isna(row['effect_magnitude']):
        return None
    if row['monotonic'] == 'N':
        return '±'
    magnitude = row['effect_magnitude']
    if '-' in str(magnitude):
        symbol = '-'
    else:
        symbol = '+'
    return symbol

def preprocess_data(df):
    effect_magnitude_data = df[df['effect_magnitude'].notna()].copy()
    effect_cascade_data = df[df['effect_cascade_success'].notna()].copy()
    
    effect_magnitude_data['numeric_magnitude'] = effect_magnitude_data['effect_magnitude'].apply(to_numeric_magnitude)
    effect_magnitude_data['symbol'] = effect_magnitude_data.apply(magnitude_to_symbol, axis=1)
    
    effect_cascade_data['plotting_variable'] = effect_cascade_data['grouping_term'].combine_first(effect_cascade_data['variable'])
    effect_cascade_data['colors'] = effect_cascade_data['effect_cascade_success'].map({'+': 'g', '-': 'r'}).fillna('b')
    
    count_data = effect_cascade_data.groupby(['plotting_variable', 'effect_cascade_success']).size().reset_index(name='counts')
    effect_cascade_data = effect_cascade_data.merge(count_data, on=['plotting_variable', 'effect_cascade_success'], how='left')
    
    mask = (effect_cascade_data.groupby('variable')['variable'].transform('count') == 1) & (effect_cascade_data['grouping_term'].isna())
    mask_1 = (effect_magnitude_data.groupby('variable')['variable'].transform('count') == 1) & (effect_magnitude_data['grouping_term'].isna())
    
    effect_cascade_data = effect_cascade_data[~mask]
    effect_magnitude_data = effect_magnitude_data[~mask_1]
    max_count = effect_cascade_data['counts'].max()
    bins = np.linspace(0, max_count, 5)
    labels = [f"{int(bins[i]) + 1}-{int(bins[i + 1])}" for i in range(4)]
    dot_sizes = [50, 150, 300, 500]
    size_mapping_legend = {label: size for label, size in zip(labels, dot_sizes)}
    
    effect_cascade_data['size_category'] = pd.cut(effect_cascade_data['counts'], bins=bins, labels=labels, include_lowest=True, right=True)
    effect_cascade_data['dot_sizes'] = effect_cascade_data['size_category'].map(size_mapping_legend)
    
    return effect_magnitude_data, effect_cascade_data

def plot_effect_cascade_success(effect_cascade_data, size_mapping_legend):
    sns.set(style="whitegrid")
    f, ax = plt.subplots(figsize=(6, 6))
    
    sns.scatterplot(x="effect_cascade_success", y="plotting_variable", 
                    size="size_category", sizes=size_mapping_legend, 
                    data=effect_cascade_data, ax=ax, legend="full", 
                    hue='colors', palette={'g': 'g', 'r': 'r', 'b': 'b'})
    
    handles, labels = ax.get_legend_handles_labels()
    size_handles = handles[-4:]
    size_labels = labels[-4:]
    
    ax.legend(size_handles, size_labels, title='Count Ranges', loc='center left', bbox_to_anchor=(1, 0.5))
    ax.yaxis.set_tick_params(pad=5)
    ax.set(xlabel='Measure of effect', ylabel='')
    ax.set_xlim(-0.5, 1.5)
    plt.tight_layout()
    plt.savefig("../Figures/Cascade_success_final.png", dpi=600)
    plt.show()
    
    
def plot_effect_magnitude(effect_magnitude_data):
    sns.set(style="whitegrid")
    plt.figure(figsize=(7, 6))
    ax = plt.gca()
    variables = effect_magnitude_data['variable'].unique()
    symbols = {'+': 0, '-': 1, '±': 2}
    y_positions = {variable: i * 2 for i, variable in enumerate(variables)}
    color_mapping = {'+': 'g', '-': 'r', '±': 'b'}
    size_mapping = {1.0: 50, 2.0: 100, 3.0: 150}
    x_jitter = 0.2
    y_jitter = 0
    
    for index, row in effect_magnitude_data.iterrows():
        y = y_positions[row['variable']] + np.random.uniform(-y_jitter, y_jitter)
        x = symbols.get(row['symbol'], 2) + np.random.uniform(-x_jitter, x_jitter)
        size = size_mapping.get(row['numeric_magnitude'], 50)
        color = color_mapping.get(row['symbol'], 'b')
        sns.scatterplot(x=[x], y=[y], size=[size], sizes=(size, size), alpha=0.8, color=color, marker='D', edgecolor='w', linewidth=0.5)
    
    ax.set_yticks(list(y_positions.values()))
    ax.set_yticklabels(list(y_positions.keys()))
    ax.set_xticks(list(symbols.values()))
    ax.set_xticklabels(list(symbols.keys()))
    
    legend_elements = [plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='grey', markersize=np.sqrt(size), label=f"{int(size_val)}")
                       for size_val, size in size_mapping.items()]
    
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1,0.8), title='Magnitude')
    plt.xlabel('Effect Magnitude')
    plt.tight_layout()
    plt.savefig("../Figures/Cascade_magnitude_final.png", dpi=600)
    plt.show()

def main():
    df = pd.read_csv(filepath, sep="\\t")
    effect_magnitude_data, effect_cascade_data = preprocess_data(df)
    max_count = effect_cascade_data['counts'].max()
    bins = np.linspace(0, max_count, 5)
    labels = [f"{int(bins[i]) + 1}-{int(bins[i + 1])}" for i in range(4)]
    dot_sizes = [50, 150, 300, 500]
    size_mapping_legend = {label: size for label, size in zip(labels, dot_sizes)}
    
    plot_effect_cascade_success(effect_cascade_data, size_mapping_legend)
    plot_effect_magnitude(effect_magnitude_data)

if __name__ == "__main__":
    main()
