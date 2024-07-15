import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator


def harmonize_ref(ref):
    if "traj" in ref:
        return ref.split("_traj")[0]  
    return ref

def load_and_process_data_histogram(file_path):
    df = pd.read_csv(file_path)
    df['harmonized_ref'] = df['ref'].apply(harmonize_ref)
    
    harmonized_balanced_dfs = []
    max_experimental_count = df[df['type'] == 'experimental'].groupby('harmonized_ref').size().max()

    for harmonized_ref, group in df.groupby('harmonized_ref'):
        nan_rows = group[group[['attribute', 'value']].isna().any(axis=1)]
        non_nan_rows = group.dropna(subset=['attribute', 'value'])
        non_nan_rows = non_nan_rows.drop_duplicates(subset=['tipping_point_c_t', 'magnitude'])
        group = pd.concat([nan_rows, non_nan_rows], ignore_index=True)
        
        experimental_group = group[group['type'] == 'experimental']
        modelling_group = group[group['type'] != 'experimental'].copy()
        modelling_group['type'].fillna('modelling', inplace=True)
        
        if len(modelling_group) > max_experimental_count:
            modelling_group = modelling_group.sample(n=max_experimental_count, random_state=42)
        
        harmonized_balanced_group = pd.concat([experimental_group, modelling_group], ignore_index=True)
        harmonized_balanced_dfs.append(harmonized_balanced_group)

    harmonized_balanced_df = pd.concat(harmonized_balanced_dfs, ignore_index=True)
    harmonized_balanced_df['type'] = harmonized_balanced_df['type'].apply(lambda x: 'empirical' if x == 'experimental' else 'modelling')
    harmonized_balanced_df['magnitude'] = harmonized_balanced_df['magnitude'].replace('[\\%,]', '', regex=True).apply(pd.to_numeric, errors='coerce')
    harmonized_balanced_df["type"] = np.where(harmonized_balanced_df["harmonized_ref"] == "Amato_2012.csv", "empirical", harmonized_balanced_df["type"])
    
    return harmonized_balanced_df

def load_and_process_data_scatter(file_path):
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

def set_axis_style(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1)  # Increased spine thickness
    ax.spines['bottom'].set_linewidth(1)
    # Increased spine thickness
    ax.spines['bottom'].set_color("black")
    ax.spines['left'].set_color("black") 
    ax.tick_params(width=2, length=6, direction='in')  # Changed tick direction to 'in'
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(which='minor', width=1, length=3, direction='in')  # Changed minor tick direction to 'in'
    ax.grid(True, linestyle='--', alpha=0.7)

def plot_scatter(df, ax_joint, ax_marg_x, ax_marg_y):
    palette = sns.color_palette("husl", 2)
    type_palette = {'empirical': palette[0], 'modelling': palette[1]}
    
    sns.scatterplot(data=df, x='tipping_point_c_t', y='magnitude', hue='type', 
                    palette=type_palette, ax=ax_joint, s=50,  linewidth=0.5)
    
    for type_category in type_palette:
        subset = df[df['type'] == type_category]
        sns.kdeplot(data=subset, x='tipping_point_c_t', ax=ax_marg_x, color=type_palette[type_category], 
                    alpha=0.5, linewidth=1.5, shade=True)
        sns.kdeplot(data=subset, y='magnitude', ax=ax_marg_y, color=type_palette[type_category], 
                    alpha=0.5, linewidth=1.5, shade=True)
    
    ax_joint.plot([0, 1], [0, 1], 'k--', label='Linear response', alpha=0.8, linewidth=1.5)
    set_axis_style(ax_joint)
    ax_joint.set_xlim([0, 1])
    ax_joint.set_ylim([0, 1.1])
    ax_joint.set_xlabel('$\lambda$', fontsize=16)
    ax_joint.set_ylabel('$n/N$', fontsize=16)
    ax_joint.tick_params(labelsize=16)
    
    handles, labels = ax_joint.get_legend_handles_labels()
    ax_joint.legend(handles=handles, labels=labels, loc='lower right', fontsize=16, frameon=True, framealpha=0.8, borderpad=1)
    ax_marg_x.axis('off')
    ax_marg_y.axis('off')

def plot_histogram(df, ax1, ax2):
    palette = sns.color_palette("husl", 2)
    type_palette = {'empirical': palette[0], 'modelling': palette[1]}
    
    hist = sns.histplot(data=df, x='tipping_point_c_t', hue='type', multiple="dodge", 
                        stat='probability', shrink=.8, palette=type_palette, bins=10, 
                        alpha=0.6, ax=ax1, edgecolor='black', linewidth=1)
    
    set_axis_style(ax1)
    ax1.set_xlabel('$\lambda$', fontsize=16)
    ax1.set_ylabel('Normalised count', fontsize=16)
    ax1.tick_params(labelsize=16)
    
    ecdf_lines = []
    for type_category, color in type_palette.items():
        subset = df[df['type'] == type_category]
        x = np.sort(subset['tipping_point_c_t'])
        y = np.arange(1, len(x) + 1) / len(x)
        line, = ax2.plot(x, y, color=color, label=f"{type_category} ECDF", linestyle='dashed', linewidth=2)
        ecdf_lines.append(line)
    set_axis_style(ax2)
    ax2.set_ylabel('Proportion (ECDF)', fontsize=16)    
    ax2.tick_params(labelsize=16)
    ax1.get_legend().remove()
    ax2.grid(False)
    return ecdf_lines

#TODO plot a is currently missing some data points compared to the OG script

def main():
    file_path = "../Data/Compiled/Tipping_points_fin_merged_1.csv"
    
    df_scatter = load_and_process_data_scatter(file_path)
    df_histogram = load_and_process_data_histogram(file_path)
    
    fig = plt.figure(figsize=(16, 7))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.2, 5], width_ratios=[3.8, 0.8, 0.5, 4])   # Adjusted width of y-axis marginal

    
    ax_joint = fig.add_subplot(gs[1, 0])
    ax_marg_x = fig.add_subplot(gs[0, 0], sharex=ax_joint)
    ax_marg_y = fig.add_subplot(gs[1, 1], sharey=ax_joint)
    
    plot_scatter(df_scatter, ax_joint, ax_marg_x, ax_marg_y)
    
    ax_hist = fig.add_subplot(gs[1, 3])
    ax_ecdf = ax_hist.twinx()
    ecdf_lines = plot_histogram(df_histogram, ax_hist, ax_ecdf)
    
    fig.text(0.05, 0.95, '(a)', fontsize=16, fontweight='bold')
    fig.text(0.60, 0.95, '(b)', fontsize=16, fontweight='bold')  # Adjusted position of (b) label
    
    fig.legend(ecdf_lines, [line.get_label() for line   in ecdf_lines], 
               loc='center right', bbox_to_anchor=(0.94, 0.42), ncol=1, fontsize=16)
    

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05)
      
    # Ensure ticks are visible on scatter plot and histogram
    for ax in [ax_joint, ax_hist, ax_marg_x, ax_marg_y]:
        ax.tick_params(axis='both', which='both', direction='in', length=6, width=2)
        ax.xaxis.set_tick_params(which='both', direction='in', length=6, width=2)
        ax.yaxis.set_tick_params(which='both', direction='in', length=6, width=2)
    
    ax_joint.yaxis.tick_left()
    ax_joint.xaxis.tick_bottom()
    ax_hist.yaxis.tick_left()
    ax_hist.xaxis.tick_bottom()

    plt.savefig("../Figures/combined_plots.svg",  bbox_inches="tight") #dpi=600,
    plt.show()

if __name__ == "__main__":
    main()