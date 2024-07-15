import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# File paths
filepath = "../Data/Compiled/The Pareto effect in tipping social networks Tipping Data - Tipping_Data.tsv"

palette = sns.color_palette("colorblind", 2)

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
    
    effect_cascade_data['colors'] = effect_cascade_data['effect_cascade_success'].map({'+': palette[1], '-': palette[0], '-/+': 'purple'}).fillna('gray')
    
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

def plot_effect_cascade_success(effect_cascade_data, size_mapping_legend, ax):
    sns.set(style="whitegrid")
    
    # Reverse the color mapping
    color_mapping = {'+': palette[0], '-': palette[1], '-/+': 'purple', 'n': 'gray'}
    
    sns.scatterplot(x="effect_cascade_success", y="plotting_variable", 
                    size="size_category", sizes=size_mapping_legend, 
                    data=effect_cascade_data, ax=ax, legend="full", 
                    hue='effect_cascade_success', palette=color_mapping)
    
    handles, labels = ax.get_legend_handles_labels()
    size_handles = handles[-4:]
    size_labels = labels[-4:]
    
    ax.legend(size_handles, size_labels, title='Count Ranges', loc='center', fontsize=14, title_fontsize=14)
    ax.yaxis.set_tick_params(pad=5)
    ax.set_xlabel('Measure of effect', fontsize=16)
    ax.set_ylabel('', fontsize=16)
    ax.set_xlim(-0.6, 1.6)
    ax.tick_params(axis='both', which='major', labelsize=16)

def plot_effect_magnitude(effect_magnitude_data, ax):
   
    sns.set(style="whitegrid")
    variables = effect_magnitude_data['variable'].unique()
    symbols = {'+': 0, '-': 1, '±': 2}
    y_positions = {variable: i * 3 for i, variable in enumerate(variables)}
    color_mapping = {'+': palette[0], '-': palette[1], '±': 'gray'}
    size_mapping = {1.0: 50, 2.0: 100, 3.0: 150}
    x_jitter = 0.2
    y_jitter = 0
    
    for index, row in effect_magnitude_data.iterrows():
        y = y_positions[row['variable']] + np.random.uniform(-y_jitter, y_jitter)
        x = symbols.get(row['symbol'], 2) + np.random.uniform(-x_jitter, x_jitter)
        size = size_mapping.get(row['numeric_magnitude'], 50)
        color = color_mapping.get(row['symbol'], 'gray')
        sns.scatterplot(x=[x], y=[y], size=[size], sizes=(size, size), alpha=0.8, color=color, marker='D', edgecolor='w', linewidth=0.5, ax=ax)
    
    ax.set_yticks(list(y_positions.values()))
    ax.set_yticklabels(list(y_positions.keys()), fontsize=16)
    ax.set_xticks(list(symbols.values()))
    ax.set_xticklabels(list(symbols.keys()), fontsize=16)
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    
    legend_elements = [plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='grey', markersize=np.sqrt(size), label=f"{int(size_val)}")
                       for size_val, size in size_mapping.items()]
    
    ax.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, 0.32), fontsize=14, title='Magnitude', title_fontsize=14)
    ax.set_xlabel('Effect Magnitude', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=16)

def main():
    df = pd.read_csv(filepath, sep="\t")
    effect_magnitude_data, effect_cascade_data = preprocess_data(df)
    max_count = effect_cascade_data['counts'].max()
    bins = np.linspace(0, max_count, 5)
    labels = [f"{int(bins[i]) + 1}-{int(bins[i + 1])}" for i in range(4)]
    dot_sizes = [50, 150, 300, 500]
    size_mapping_legend = {label: size for label, size in zip(labels, dot_sizes)}
    
    fig = plt.figure(figsize=(12, 8), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[0.55, 0.45])
    
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    
    plot_effect_cascade_success(effect_cascade_data, size_mapping_legend, ax1)
    plot_effect_magnitude(effect_magnitude_data, ax2)
    
    # Adjust the subplot layouts
    ax1.set_xlim(-0.6, 1.6)
    ax2.set_xlim(-0.5, 2.5)
    
    line = plt.Line2D((0.625, 0.625), (0.1, 0.9), color='black', linewidth=1, transform=fig.transFigure, figure=fig)
    fig.add_artist(line)
    
    # Adjust the position and size of the axes
    ax1.set_position([0.25, 0.1, 0.35, 0.8])
    ax2.set_position([0.65, 0.1, 0.35, 0.8 ])
    
    # Add (a) and (b) labels
    ax1.text(0.02, 1.02, '(a)', transform=ax1.transAxes, fontsize=16, fontweight='bold', va='bottom', ha='left')
    ax2.text(0.02, 1.02, '(b)', transform=ax2.transAxes, fontsize=16, fontweight='bold', va='bottom', ha='left')
    
    # Save in SVG format
    plt.savefig("../Figures/Combined_plot_final.svg", format='svg', bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    main()
