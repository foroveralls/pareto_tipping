# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 04:29:53 2023

@author: everall
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # For plotting
from collections import defaultdict
import os

#%%
folder_path = "../Data/Time_series"

# List of CSV files
file_paths = os.listdir(folder_path)
file_paths = ["Lacopini_2022.csv"]


#%%
def calculate_second_derivative_adaptive(x, y, i):
    h1 = x[i] - x[i - 1]
    h2 = x[i + 1] - x[i]
    d2 = 2 * (h2 * (y[i] - y[i - 1]) / h1 + h1 * (y[i + 1] - y[i]) / h2) / (h1 + h2)
    return d2

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

def process_trajectories_refined_corrected(df, trajectories, file_name, plot=False):
    result_data = []
    for idx, traj in enumerate(trajectories):
        traj_df = df.iloc[traj['start']:traj['end'] + 1]
        x = traj_df['x'].to_numpy()
        y = traj_df['y'].to_numpy()
        y_max= max(y)
        print(y_max)
        d2_adaptive = np.array([calculate_second_derivative_adaptive(x, y, i) for i in range(1, len(x) - 1)])
        positive_idx_adaptive = np.where(d2_adaptive > 0)[0]
        
        if len(positive_idx_adaptive) > 0:
            sorted_idx = np.argsort(d2_adaptive[positive_idx_adaptive])[::-1]  # Sort in descending order
            for i in sorted_idx:
                #if y[1:-1][i] <= 0.50:
                max_d2_idx = positive_idx_adaptive[i]
                max_d2_y = y[1:-1][max_d2_idx]
                if max_d2_y <= 0.50:
                    break
                else:
                    continue  # No suitable critical point found, skip to the next trajectory
            
            ref = f"{file_name[:-4]}_traj{idx + 1}"  # Corrected unique reference for each trajectory
            result_data.append({'ref': ref, 'magnitude': y_max, 'y_max_d2': max_d2_y, **traj['attrs']})
            

      
            if plot:
                plt.figure(figsize=(10, 6))
                plt.plot(x, y, label='y')
                plt.plot(x[1:-1], d2_adaptive, label="2nd Derivative")
                plt.ylim(-1,1)
                plt.axvline(x=x[1:-1][max_d2_idx], color='r', linestyle='--', label=f'Critical Point: {x[1:-1][max_d2_idx]}')
                plt.legend()
                plt.show()
          # Extract trajectory attributes and append them to result_data along with y_max
            
                
    return pd.DataFrame(result_data)

def main(file_paths, output_file_path, plot=False):
    final_df = pd.DataFrame()
    for file_path in file_paths:
        df = pd.read_csv(os.path.join(folder_path, file_path))
        print(f"Processing {file_path}")
        
        trajectories = identify_trajectories_based_on_attributes(df)
        result_df = process_trajectories_refined_corrected(df, trajectories, os.path.basename(file_path), plot)
        result_df = result_df.rename(columns={'y_max_d2': 'tipping_point_c_t'})
        
        if not result_df.empty:
            result_df_long = result_df.melt(id_vars=['ref', 'tipping_point_c_t', "magnitude"], var_name='attribute', value_name='value')
            final_df = final_df.append(result_df_long, ignore_index=True)
    
    #final_df.to_csv(output_file_path, index=False)
    return final_df

#%%

output_file_path = "../Data/Compiled/Tipping_points_fin_merged_1.csv"
#additional_columns = ['topology', 'magnitude']  # Add any additional columns needed
plot = True # Set to True to plot graphs for each trajectory

if __name__ == "__main__":
    final = main(file_paths, output_file_path, plot = plot)
