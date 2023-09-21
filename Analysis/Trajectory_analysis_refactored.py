# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 16:39:19 2023

@author: everall
"""

import os
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt



#%%
folder_path = "../Data/Model"

# List of CSV files
filepaths = os.listdir(folder_path)
filepaths.pop()

#%%

def read_csv_file(filepath):
    return pd.read_csv(filepath)

def extract_author_name(filepath):
    return os.path.basename(filepath).split('_')[0]

def fill_missing_values(df, non_xy_cols):
    df[non_xy_cols] = df[non_xy_cols].fillna(method='ffill').fillna(method='bfill')
    return df

def gaussian_filter_smoothing(x, y):
    sigma = len(y) // 10
    smooth = gaussian_filter1d(y, sigma=sigma)
    d2 = np.gradient(np.gradient(smooth))
    max_d2_idx = np.argmax(d2)
    return smooth, d2, max_d2_idx

def plot_trajectory(x, y, smooth, d2, max_d2_idx):
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'o', label='Original Data Points')
    plt.plot(x, smooth, label='Smoothed Trajectory (Gaussian Filter)')
    plt.plot(x, d2, label='2nd Derivative')
    plt.axvline(x[max_d2_idx], color='r', linestyle='--', label=f'Max 2nd Derivative\ny={y[max_d2_idx]:.4f}')
    plt.xlabel('x')
    plt.ylabel('y / 2nd Derivative')
    plt.legend()
    plt.tight_layout()
    plt.show()

def process_file(filepath):
    df = read_csv_file(filepath)
    author = extract_author_name(filepath)
    non_xy_cols = [col for col in df.columns if col not in ['x', 'y']]
    df = fill_missing_values(df, non_xy_cols)
    
    results = []
    unique_values = df[non_xy_cols].drop_duplicates()
    for _, unique_value in unique_values.iterrows():
        condition = (df[non_xy_cols] == unique_value).all(axis=1)
        trajectory = df[condition]
        x, y = trajectory['x'].values, trajectory['y'].values
        smooth, d2, max_d2_idx = gaussian_filter_smoothing(x, y)
        
        unique_value_str = ', '.join(f"{col}: {val}" for col, val in unique_value.items())
        results.append({
            'Author': author,
            'Unique_Value': unique_value_str,
            'Max_2nd_Derivative_Y': y[max_d2_idx],
            'Steady_State_Y': y[-1]
        })
        
        plot_trajectory(x, y, smooth, d2, max_d2_idx)
        
    return results

#%%
def main():
    
    results_df = pd.DataFrame(columns=['Author', 'Unique_Value', 'Max_2nd_Derivative_Y', 'Steady_State_Y'])
    for filepath in filepaths:
        filepath = os.path.join(folder_path, filepath)
        results = process_file(filepath)
        results_df = results_df.append(results, ignore_index=True)
        
    print(results_df)
    results_df.to_csv('results.csv', index=False)

if __name__ == "__main__":
    main()
