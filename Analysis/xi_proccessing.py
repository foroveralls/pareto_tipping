# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 23:00:33 2023

@author: everall

"""


import pandas as pd


xie_et_al_df = pd.read_csv('../Data/Model/Xie et al_2011_1_minority_size.csv', delimiter=';')


# Load the Tipping_points_fin_merged data
tipping_points_fin_df = pd.read_csv('../Data/Compiled/Tipping_points_fin_merged_1.csv')

# Load the Xie et al_2011 data with correct delimiter
xie_et_al_df = pd.read_csv('../Data/Model/Xie et al_2011_1_minority_size.csv', delimiter=';')

# Initialize lists to hold the new rows to be appended to tipping_points_fin_df
new_rows = []

# Iterate over rows in xie_et_al_df and create new rows according to the instructions
for index, row in xie_et_al_df.iterrows():
    # Create a row with 'tipping_point_c_t' and 'magnitude'
 
    
    new_row_base = {
        'ref': 'Xie et al_2011',
        'tipping_point_c_t': row['minority'],
        'magnitude': 1 - row['size'],  # Modify magnitude to be 1 - size
        'type': "modelling"
    }
    
    
    
    # Append the row with 'tipping_point_c_t' and 'magnitude'
    #new_rows.append(new_row_base.copy())
    
    # Create and append rows for 'topology', 'N', and 'k'
    for attr in ['topology', 'N', 'k']:
        new_row = new_row_base.copy()
        new_row['attribute'] = attr
        new_row['value'] = row[attr]
        new_rows.append(new_row)
        
# Append the new rows to tipping_points_fin_df
extended_tipping_points_fin_df = tipping_points_fin_df.append(new_rows, ignore_index=True)

# Save the extended dataframe to a new CSV file
extended_tipping_points_fin_df.to_csv('../Data/Compiled/Tipping_points_fin_merged_1.csv', index=False)

# Display a segment of the extended dataframe
print(extended_tipping_points_fin_df.sample(10))

