# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 16:19:26 2023

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

# Initialize the results DataFrame
results_df = pd.DataFrame(columns=['Author', 'Unique_Value', 'Max_2nd_Derivative_Y', 'Steady_State_Y'])

# Process each file
for filepath in filepaths:
    # Read the CSV file
    df = pd.read_csv(os.path.join(folder_path, filepath))
    
    # Extract the author name from the filename
    author = os.path.basename(filepath).split('_')[0]
    
    # Identify non-x and y columns
    non_xy_cols = [col for col in df.columns if col not in ['x', 'y']]
    
    # Fill missing values in non-x and y columns
    df[non_xy_cols] = df[non_xy_cols].fillna(method='ffill').fillna(method='bfill')
    
    # Identify unique trajectories and process each
    unique_values = df[non_xy_cols].drop_duplicates()
    for _, unique_value in unique_values.iterrows():
        # Extract the trajectory
        condition = (df[non_xy_cols] == unique_value).all(axis=1)
        trajectory = df[condition]
        
        # Gaussian filter method for smoothing
        x = trajectory['x'].values
        y = trajectory['y'].values
        sigma = len(y) // 10
        smooth = gaussian_filter1d(y, sigma=sigma)
        d2 = np.gradient(np.gradient(smooth))
        max_d2_idx = np.argmax(d2)
        
        # Extract results
        unique_value_str = ', '.join(f"{col}: {val}" for col, val in unique_value.items())
        max_2nd_derivative_y = y[max_d2_idx]
        steady_state_y = y[-1]  # Assuming the last y value is the steady state value
        
        # Append results to the DataFrame
        results_df = results_df.append({
            'Author': author,
            'Unique_Value': unique_value_str,
            'Max_2nd_Derivative_Y': max_2nd_derivative_y,
            'Steady_State_Y': steady_state_y
        }, ignore_index=True)
        
        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(x, y, 'o', label='Original Data Points')
        plt.plot(x, smooth, label='Smoothed Trajectory (Gaussian Filter)')
        plt.plot(x, d2, label='2nd Derivative')
        plt.axvline(x[max_d2_idx], color='r', linestyle='--', label=f'Max 2nd Derivative\ny={max_2nd_derivative_y:.4f}')
        plt.title(f'{author} - {unique_value_str}')
        plt.xlabel('x')
        plt.ylabel('y / 2nd Derivative')
        plt.legend()
        plt.tight_layout()
        plt.show()

# Output the results DataFrame
print(results_df)
