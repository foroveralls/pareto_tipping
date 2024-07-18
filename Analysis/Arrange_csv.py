# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 05:21:40 2024

@author: Jordan
"""

import pandas as pd

# Load data from CSV file
input_file = "../Data/Raw/Model/Karsai_2016_2c_r_0.csv"

df = pd.read_csv(input_file)

# Sort the dataframe by the 'N' and 'x' columns in ascending order
df_sorted = df.sort_values(by=[' r', 'x'])

# Save the sorted dataframe to a new CSV file
output_file = '../Data/Raw/Model/Karsai_2016_2c_r_0_sorted.csv'
df_sorted.to_csv(output_file, index=False)

print(f"Data has been sorted by trajectory and saved to {output_file}")
