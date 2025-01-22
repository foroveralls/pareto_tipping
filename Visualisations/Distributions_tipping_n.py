import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import beta, norm
import seaborn as sns
from matplotlib import rcParams

pad = 10

def set_publication_params():
    # Set publication-quality parameters
    rcParams['axes.linewidth'] = 0.5        # Thinner axis lines
    rcParams['axes.grid'] = False           # Remove default grid
    rcParams['lines.linewidth'] = 1.0       # Consistent line width
    rcParams['axes.spines.top'] = False     # Remove top spine
    rcParams['axes.spines.right'] = False   # Remove right spine
    rcParams['figure.dpi'] = 300            # High DPI for clear rendering
    rcParams['savefig.bbox'] = 'tight'      # Tight layout when saving
    rcParams['savefig.pad_inches'] = 0.02   # Minimal padding

def plot_left_skewed(ax):
    x = np.linspace(0, 1, 1000)
    ax.plot(x, beta.pdf(x, 8, 1), color='black', linewidth=1)
    ax.fill_between(x, beta.pdf(x, 8, 1), where=(x >= 0.85) & (x <= 0.95), color='gray', alpha=0.5)
    ax.set_title('(a)', loc='left', pad=pad)
    ax.grid(False)

def plot_normal(ax):
    x_low = norm.ppf(0.001)
    x_high = norm.ppf(0.999)
    x = np.linspace(x_low, x_high, 1000)
    x_transformed = (x - x_low) / (x_high - x_low)
    ax.plot(x_transformed, norm.pdf(x, 0, 1), color='black', linewidth=1)
    ax.fill_between(x_transformed, norm.pdf(x, 0, 1), 
                   where=(x_transformed >= 0.25) & (x_transformed <= 0.75), 
                   color='gray', alpha=0.5)
    ax.set_title('(b)', loc='left', pad=pad)
    ax.grid(False)

def plot_right_skewed(ax):
    x = np.linspace(0, 1, 1000)
    ax.plot(x, beta.pdf(x, 1, 8), color='black', linewidth=1)
    ax.fill_between(x, beta.pdf(x, 1, 8), where=(x >= 0.05) & (x <= 0.15), 
                   color='gray', alpha=0.5)
    ax.set_title('(c)', loc='left', pad=pad)
    ax.grid(False)

def multimodal_pdf(x):
    # Much smaller weights for transition components
    pdf = (0.5 * norm.pdf(x, -0.05, 0.02) +      # Sharp left peak
           0.008 * norm.pdf(x, 0.009, 0.09) +        # Slight soft slope after left peak (reduced from 0.1)
           0.0015 * norm.pdf(x, 0.5, 0.05) +     # Tiny central peak
           0.01 * norm.pdf(x, 0.5, 0.15) +       # Wider component for middle transition
           0.005 * norm.pdf(x, 0.95, 0.1) +        # Slight soft slope before right peak (reduced from 0.1)
           0.5 * norm.pdf(x, 1.05, 0.02) +       # Sharp right peak
           0.15)                                  # Uniform background
    return pdf

def plot_multimodal(ax):
    x = np.linspace(0, 1, 1000)
    y = multimodal_pdf(x)
    
    # Scale to match the original scale
    y = y / np.max(y) * 4
    
    ax.plot(x, y, color='black', linewidth=1)
    ax.set_title('(d)', loc='left', pad=pad)
    ax.grid(False)

def main():
    set_publication_params()
    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(16, 4))
    
    gs = fig.add_gridspec(1, 4)
    axs = [fig.add_subplot(gs[0, i]) for i in range(4)]
    
    plot_left_skewed(axs[0])
    plot_normal(axs[1])
    plot_right_skewed(axs[2])
    plot_multimodal(axs[3])
    
    for ax in axs:
        ax.set_yticklabels([])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
    
    fig.supxlabel(r"Individual threshold: $\phi$", fontsize=16)
    fig.supylabel("Density: f(x)", fontsize=16)
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.2) 
    return fig

if __name__ == "__main__":
    fig = main()
    plt.savefig("../Figures/distributions_final.pdf", bbox_inches='tight', dpi=300)
    plt.show()