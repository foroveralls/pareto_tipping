# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 18:47:05 2024

@author: Jordan
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

file_path = '../Data/Compiled/The Pareto effect in tipping social networks_ Tipping Data - Tipping_Data_Raw.csv'
# Read the CSV file
df = pd.read_csv(file_path)

# Filter for 'w' variable
df_w = df[df['variable'] == 'w'].drop_duplicates(subset=['value'])

# Extract lower and upper bounds from the 'value' column
df_w[['lower', 'upper']] = df_w['value'].str.split('-', expand=True).astype(float)
df_w['mean'] = df_w[['lower', 'upper']].mean(axis=1)

# Sort by upper bound value
df_w = df_w.sort_values('upper', ascending=True)

# Create the plot
fig, ax = plt.subplots(figsize=(12, 10))

# Plot horizontal bars for ranges
bars = ax.barh(range(len(df_w)), df_w['upper'] - df_w['lower'], left=df_w['lower'], height=0.5, 
        color='lightblue', alpha=0.8)

# Plot points for mean values
ax.scatter(df_w['mean'], range(len(df_w)), color='navy', s=50, zorder=3, label='Mean')

# Customize the plot
ax.set_xlabel('Individual Tipping Threshold ($\phi$)', fontsize=16)
ax.set_title('Threshold Fraction Ranges that Allow Tipping', fontsize=14)
ax.set_xlim(0, 1)
ax.grid(axis='x', linestyle='--', alpha=0.3)

# Add value labels for upper bounds
for i, (index, row) in enumerate(df_w.iterrows()):
    if not np.isnan(row['upper']):
        ax.text(row['upper'], i, f'{row["upper"]:.2f}', va='center', ha='left', fontsize=9)

# Adjust y-axis
ax.set_yticks(range(len(df_w)))
ax.set_yticklabels(df_w['ref'], fontsize=10)
ax.set_ylabel("Reference", fontsize = 16, labelpad = 15)
# Add legend
ax.legend(loc='upper right')

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Adjust layout and display
plt.tight_layout()
plt.savefig("../Figures/indivdidual_thresholds.pdf", dpi = 300, bbox_inches="tight") #dpi=600,
plt.show()
