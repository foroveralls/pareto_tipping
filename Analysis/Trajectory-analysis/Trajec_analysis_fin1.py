# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 04:29:53 2023

@author: everall
"""

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt  # For plotting
from collections import defaultdict
import os

#%%
folder_path = "../Data/Time_series"

# List of CSV files
file_paths = os.listdir(folder_path)
file_paths = ["Centola_2015_norm_ecology_data.csv"]


#%%
def calculate_second_derivative_adaptive(x, y, i):
    if i == 0  or i == len(x) - 1:
        return 0  # Handle boundary cases
    h1 = x[i] - x[i - 1]
    h2 = x[i + 1] - x[i]
    d2 = 2 * (h2 * (y[i] - y[i - 1]) / h1 - h1 * (y[i + 1] - y[i]) / h2) / (h1 * h2 * (h1 + h2))
    return d2

def calculate_second_derivative_numpy(x, y, i):
    if i == 0 or i == len(x) - 1:
        return 0  # Handle boundary cases

    d2 = np.gradient(np.gradient(y, x), x)
    return d2[i]

def calculate_first_derivative_adaptive(x, y, i):
    if i == 0 or i == len(x) - 1:
        return 0  # Handle boundary cases
    h1 = x[i] - x[i - 1]
    h2 = x[i + 1] - x[i]
    d1 = (y[i + 1] - y[i - 1]) / (h1 + h2)
    return d1

def calculate_second_derivative_centered_uniform(x, y, i):
    if i == 0 or i == len(x) - 1:
        return 0  # Handle boundary cases

    h = x[1] - x[0]  # Assuming uniform grid spacing
    
    # Centered difference formula for uniform grid
    d2 = (y[i + 1] - 2 * y[i] + y[i - 1]) / (h * h)
    
    return d2

def normalize_time_series(x):
    min_x = min(x)
    return [xi - min_x for xi in x]

def identify_trajectories_based_on_attributes(df):
    trajectories = []
    trajectory = {'start': 0, 'end': 0, 'attrs': {}}
    last_observed_attrs = df.iloc[0].drop(['x', 'y']).to_dict()

    for idx, row in df.iterrows():
        if idx == 0:  # Skip the first row, as it is used to initialize last_observed_attrs
            continue

        current_attrs = row.drop(['x', 'y']).to_dict()
        if current_attrs != last_observed_attrs:  # Compare non-x, non-y columns to identify the start of a new trajectory
            trajectory['end'] = idx - 1
            trajectories.append(trajectory)
            trajectory = {'start': idx, 'end': idx, 'attrs': last_observed_attrs.copy()}

        last_observed_attrs = current_attrs  # Update last_observed_attrs for the next iteration

    trajectory['end'] = df.index[-1]  # The end of the last trajectory is the last row in the DataFrame
    trajectories.append(trajectory)
    #print(len(trajectories))
    return trajectories


def process_trajectories_refined_corrected(df, trajectories, deriv_fn, file_name, plot=False):
    result_data = []
    for idx, traj in enumerate(trajectories):
        traj_df = df.iloc[traj['start']:traj['end'] + 1].copy()
        
        # Normalize x values for this trajectory
        x = traj_df['x'].to_numpy()
        if np.min(x) < 0:
            x_normalized = normalize_time_series(x)
            traj_df['x'] = x_normalized
        else:
            x_normalized = x
        
        y = traj_df['y'].to_numpy()
        y_max = max(y)

        # Calculate second derivative using normalized x values
        d2_adaptive = np.array([deriv_fn(x_normalized, y, i) for i in range(len(x_normalized))])
        
        # Find indices where second derivative is positive and finite
        positive_idx = np.where((d2_adaptive > 0) & np.isfinite(d2_adaptive))[0]
        
        if len(positive_idx) > 0:
            # Sort positive indices by their second derivative value in descending order
            sorted_positive_idx = positive_idx[np.argsort(d2_adaptive[positive_idx])[::-1]]
            
            for i in sorted_positive_idx:
                if y[i] <= 0.50:
                    max_d2_idx = i
                    max_d2_y = y[max_d2_idx]
                    print(max_d2_y)
                    break
            else:
                continue
            
            ref = f"{file_name[:-4]}_traj{idx + 1}"
            result_data.append({'ref': ref, 'magnitude': y_max, 'y_max_d2': max_d2_y, **traj['attrs']})
            
            if plot:
                plt.figure(figsize=(10, 6))
                plt.plot(x_normalized, y, label='y')
                plt.plot(x_normalized, d2_adaptive, label="2nd Derivative")
                plt.ylim(-1,1)
                plt.axvline(x=x_normalized[max_d2_idx], color='r', linestyle='--', label=f'Critical Point: {x_normalized[max_d2_idx]}')
                plt.legend()
                plt.show()
                
    return pd.DataFrame(result_data)

def main(file_paths, output_file_path, deriv_fn, plot=False):
    final_df = pd.DataFrame()
    for file_path in file_paths:
        df = pd.read_csv(os.path.join(folder_path, file_path))
        print(f"Processing {file_path}")
        
        trajectories = identify_trajectories_based_on_attributes(df)
        result_df = process_trajectories_refined_corrected(df, trajectories, deriv_fn, os.path.basename(file_path), plot)
        result_df = result_df.rename(columns={'y_max_d2': 'tipping_point_c_t'})
        
        if not result_df.empty:
            result_df_long = result_df.melt(id_vars=['ref', 'tipping_point_c_t', "magnitude"], var_name='attribute', value_name='value')
            final_df = pd.concat([final_df, result_df_long], ignore_index=True)
    
    #final_df.to_csv(output_file_path, index=False)
    return final_df

#%%

output_file_path = "../Data/Compiled/Tipping_points_fin_merged_1.csv"
#additional_columns = ['topology', 'magnitude']  # Add any additional columns needed
plot = True # Set to True to plot graphs for each trajectory

if __name__ == "__main__":
    final = main(file_paths, output_file_path, calculate_second_derivative_centered_uniform,  plot = plot)
#calculate_second_derivative_adaptive