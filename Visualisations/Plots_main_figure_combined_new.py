import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from scipy.optimize import curve_fit

def sigmoid(x, k, x0):
    return 1 / (1 + np.exp(-k * (x - x0)))

def evaluate_ecdf(data, point):
    """
    Evaluate the ECDF of the given data at a specific point.
    
    Parameters:
    data (array-like): The data for which to calculate the ECDF.
    point (float): The point at which to evaluate the ECDF.
    
    Returns:
    float: The value of the ECDF at the given point.
    """
    sorted_data = np.sort(data)
    n = len(sorted_data)
    y = np.arange(1, n + 1) / n
    
    # Find the index of the largest value in sorted_data that is <= point
    idx = np.searchsorted(sorted_data, point, side='right') - 1
    
    if idx < 0:  # point is smaller than all data
        return 0
    elif idx >= n:  # point is larger than all data
        return 1
    else:
        return y[idx]
    
def find_ecdf_value(data, ecdf_point):
    """
    Find the data value at which the ECDF reaches a specific point.
    
    Parameters:
    data (array-like): The data for which to calculate the ECDF.
    ecdf_point (float): The ECDF value (between 0 and 1) for which to find the corresponding data value.
    
    Returns:
    float: The data value at which the ECDF reaches the given point.
    """
  
    sorted_data = np.sort(data)
    n = len(sorted_data)
    ecdf_values = np.arange(1, n + 1) / n
    
    # Find the index where the ECDF is closest to (but not exceeding) the desired point
    idx = np.searchsorted(ecdf_values, ecdf_point, side='right') - 1
    
    if idx < 0:  # ecdf_point is smaller than the smallest ECDF value
        return sorted_data[0]
    elif idx >= n:  # ecdf_point is larger than the largest ECDF value
        return sorted_data[-1]
    else:
        return sorted_data[idx]
def set_axis_style(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1)  # Increased spine thickness
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['bottom'].set_color("black")
    ax.spines['left'].set_color("black") 
    ax.tick_params(width=2, length=6, direction='in')  # Changed tick direction to 'in'
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(which='minor', width=1, length=3, direction='in')  # Changed minor tick direction to 'in'
    ax.grid(True, linestyle='--', alpha=0.7)

def plot_scatter(df, ax_joint, ax_marg_x, ax_marg_y, show_sigmoid_fit=False):
    palette = sns.color_palette("husl", 2)
    type_palette = {'empirical': palette[0], 'modelling': palette[1]}

    # Debugging: Ensure all 'experimental' values are converted to 'empirical'
    df['type'] = df['type'].replace('experimental', 'empirical')

    # Debugging: Print unique values in the 'type' column
    print("\nUnique values in 'type' column in scatter plot:")
    print(df['type'].unique())

    sns.scatterplot(data=df, x='tipping_point_c_t', y='magnitude', hue='type', 
                    palette=type_palette, ax=ax_joint, s=50, linewidth=0.5)
    
    for type_category in type_palette:
        subset = df[df['type'] == type_category]
        sns.kdeplot(data=subset, x='tipping_point_c_t', ax=ax_marg_x, color=type_palette[type_category], 
                    alpha=0.5, linewidth=1.5, shade=True)
        sns.kdeplot(data=subset, y='magnitude', ax=ax_marg_y, color=type_palette[type_category], 
                    alpha=0.5, linewidth=1.5, shade=True)
    
    ax_joint.plot([0, 1], [0, 1], 'k--', label='Linear response', alpha=0.8, linewidth=1.5)
    
    if show_sigmoid_fit:
        # Filter out inf and NaN values
        valid_data = df[np.isfinite(df['tipping_point_c_t']) & np.isfinite(df['magnitude'])]
        
        if len(valid_data) > 0:
            try:
                # Fit sigmoid function
                popt, _ = curve_fit(sigmoid, valid_data['tipping_point_c_t'], valid_data['magnitude'], p0=[25, 0.25])
                x_sigmoid = np.linspace(0, 1, 100)
                y_sigmoid = sigmoid(x_sigmoid, *popt)
                ax_joint.plot(x_sigmoid, y_sigmoid, 'r-', label='Sigmoid fit', linewidth=2)
            except RuntimeError:
                print("Curve fit failed. The sigmoid might not be a good fit for this data.")
        else:
            print("No valid data points for curve fitting after filtering out inf and NaN values.")
    
        # Calculate and plot average tipping threshold
    avg_tipping_threshold = df['tipping_point_c_t'].mean()
    ax_joint.axvline(x=avg_tipping_threshold, color='purple', linestyle='dotted', linewidth=1.5, 
                     label=f'Avg. λ = {avg_tipping_threshold:.3f}')
    
    # # Add text annotation for average
    # ax_joint.text(avg_tipping_threshold + 0.02, 0.05, f'Avg. λ = {avg_tipping_threshold:.3f}', 
    #               rotation=90, verticalalignment='bottom', fontsize=12, color='purple')
    
    set_axis_style(ax_joint)
    ax_joint.set_xlim([0, 1])
    ax_joint.set_ylim([0, 1.1])
    ax_joint.set_xlabel('$\lambda$', fontsize=16)
    ax_joint.set_ylabel('$F_{\infty}$', fontsize=16)
    ax_joint.tick_params(labelsize=16)
    
    handles, labels = ax_joint.get_legend_handles_labels()
    ax_joint.legend(handles=handles, labels=labels, loc='lower right', fontsize=14, frameon=True, framealpha=0.8, borderpad=0.6)
    ax_marg_x.axis('off')
    ax_marg_y.axis('off')
def plot_histogram(df, ax1, ax2):
    # Ensure the 'magnitude' column is numeric
    df['magnitude'] = pd.to_numeric(df['magnitude'], errors='coerce')
    
    # Filter to only include critical points with magnitude > 0.5
    df = df[df['magnitude'] > 0.5]

    palette = sns.color_palette("husl", 2)
    type_palette = {'empirical': palette[0], 'modelling': palette[1]}

    # Debugging: Ensure all 'experimental' values are converted to 'empirical'
    df['type'] = df['type'].replace('experimental', 'empirical')
    
    
    # Calculate average tipping threshold
    avg_tipping_threshold = df['tipping_point_c_t'].mean()
    print(avg_tipping_threshold)

    # Plot vertical line for average tipping threshold
    avg_line = ax1.axvline(x=avg_tipping_threshold, color='purple', linestyle='dotted', linewidth=2, label=f'Average λ = {avg_tipping_threshold:.3f}')
    #ax2.axvline(x=avg_tipping_threshold, color='red', linestyle='--', linewidth=2)

    
    # Debugging: Print unique values in the 'type' column
    print("\nUnique values in 'type' column in histogram plot:")
    print(df['type'].unique())
    
    sns.histplot(data=df, x='tipping_point_c_t', hue='type', multiple="dodge", 
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
    

  
    return ecdf_lines, avg_line

#%%
def main():
    file_path_scatter = "../Data/Compiled/Tipping_threshold_plot.csv"
    file_path_histogram = "../Data/Compiled/Tipping_threshold_plot.csv"
    
    df_full = pd.read_csv(file_path_scatter)

    # Remove duplicate tipping points
    df_unique = df_full.drop_duplicates(subset=['tipping_point_c_t', 'magnitude', 'type'])
    

    # Calculate average tipping threshold
    avg_tipping_threshold = df_unique['tipping_point_c_t'].mean()
    print(f"Average tipping threshold: {avg_tipping_threshold:.3f}")
    
     # Print unique data info
    print("\nUnique data shape:", df_unique.shape)
    print("Unique type counts:")
    print(df_unique['type'].value_counts())
 
    # Print unique values in the 'type' column
    print("\nUnique values in 'type' column:")
    print(df_unique['type'].unique())
        
    show_sigmoid = False
    
    fig = plt.figure(figsize=(16, 7))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.2, 5], width_ratios=[3.8, 0.8, 0.5, 4.5])  # Increased last value from 4 to 4.5  # Adjusted width of y-axis marginal

    
    ax_joint = fig.add_subplot(gs[1, 0])
    ax_marg_x = fig.add_subplot(gs[0, 0], sharex=ax_joint)
    ax_marg_y = fig.add_subplot(gs[1, 1], sharey=ax_joint)
    
    plot_scatter(df_unique, ax_joint, ax_marg_x, ax_marg_y, show_sigmoid_fit=show_sigmoid)
    
    ax_hist = fig.add_subplot(gs[1, 3])
    ax_ecdf = ax_hist.twinx()
    
    
    ecdf_lines, avg_line = plot_histogram(df_unique, ax_hist, ax_ecdf)

    # Merge legends
    all_lines = ecdf_lines + [avg_line]
    all_labels = [line.get_label() for line in all_lines]
    
    fig.legend(all_lines, all_labels, 
               loc='center right', bbox_to_anchor=(0.73, 0.70), borderpad = 0.6, framealpha=0.8, ncol=1, fontsize=14)
    
    fig.text(0.05, 0.95, '(a)', fontsize=16, fontweight='bold')
    fig.text(0.538, 0.95, '(b)', fontsize=16, fontweight='bold')  # Adjusted position of (b) label
    
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05)
      
    for ax in [ax_joint, ax_hist, ax_marg_x, ax_marg_y]:
        ax.tick_params(axis='both', which='both', direction='in', length=6, width=2)
        ax.xaxis.set_tick_params(which='both', direction='in', length=6, width=2)
        ax.yaxis.set_tick_params(which='both', direction='in', length=6, width=2)
    
    ax_joint.yaxis.tick_left()
    ax_joint.xaxis.tick_bottom()
    ax_hist.yaxis.tick_left()
    ax_hist.xaxis.tick_bottom()

    plt.savefig("../Figures/combined_plots.pdf", bbox_inches="tight", dpi = 300) #dpi=600,
    plt.show()

if __name__ == "__main__":
    main()
    
#%%
file_path_scatter = "../Data/Compiled/Tipping_threshold_plot.csv"

df_full = pd.read_csv(file_path_scatter)

# Remove duplicate tipping points
df_unique = df_full.drop_duplicates(subset=['tipping_point_c_t', 'magnitude', 'type'])

df_unique = df_unique[df_unique['magnitude'] > 0.5]

data_em = df_unique[df_unique['type'] == 'empirical']['tipping_point_c_t']

data_mod = df_unique[df_unique['type'] == 'modelling']['tipping_point_c_t']

point = 0.95
ecdf_value = find_ecdf_value(df_unique['tipping_point_c_t'], point)

smallest = min(df_unique = df_unique[df_unique['magnitude'] > 0.5])

modelling_ecdf, emp_ecdf = [find_ecdf_value(x, 0.95) for x in [data_mod, data_em]]
(lambda x: find_ecdf_value(x, point), )

modelling_ecdf, emp_ecdf = [evaluate_ecdf(x, 0.25) for x in [data_mod, data_em]]
avg_vals = data_mod.mean(), data_em.mean() 
total = df_unique["tipping_point_c_t"].mean()
print(f"ECDF value at {point}: {ecdf_value}")



  