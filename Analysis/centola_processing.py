# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 23:37:04 2023

@author: everall
"""

import pandas as pd
import os 

#%%

folder_path = "../Data/"


#%%


def load_data(centola_filepath, merged_filepath):
    centola_data = pd.read_csv(centola_filepath,  skiprows=3, header=0)
    merged_data = pd.read_csv(merged_filepath)
    return centola_data, merged_data

def filter_last_round(centola_data):
    return centola_data.groupby('trial').last().reset_index()

def calculate_magnitude_threshold(centola_data):
    new_rows = []
    for _, row in centola_data.iterrows():
        magnitude = row['adoption_alternative']
        tipping_point_c_t = row['cm_count'] / row['N']  # Assuming 'cm_count' and 'N' are column names in centola_data
        new_row = {
            'ref': 'Centola_2018',
            'tipping_point_c_t': tipping_point_c_t,
            'attribute': None,
            'value': None,
            'type': "experimental",
            'magnitude': magnitude
        }
        new_rows.append(new_row)
    return new_rows

def append_to_merged(merged_data, new_rows):
    return merged_data.append(pd.DataFrame(new_rows), ignore_index=True)

def save_merged_data(merged_data, filepath):
    merged_data.to_csv(filepath, index=False)
    
def convert_to_fraction(merged_data):
    # Define the columns to be converted
    columns_to_convert = ['tipping_point_c_t', 'magnitude']
    
    # Loop over each row in the DataFrame
    for index, row in merged_data.iterrows():
        for column in columns_to_convert:
            value = row[column]
            
            # Remove the percentage sign and convert string values to float
            if isinstance(value, str):
                try:
                    # Remove the percentage sign if present and convert to float
                    value = float(value.strip('%'))
                except ValueError:
                    print(f"Could not convert value '{value}' to float in column '{column}', index {index}. Skipping...")
                    continue
            
            # Check if the value is in percentage format (i.e., greater than 1)
            if value > 1:
                # Convert to fraction
                merged_data.at[index, column] = value / 100
                
    return merged_data



def main():
    centola_filepath = os.path.join(folder_path, 
                                    'Experimental/Centola-Becker-Brackbill-Baronchelli_Complete-Data_2018_size_threshold.csv')
    merged_filepath = os.path.join(folder_path,'Compiled/Tipping_points_fin_merged_1.csv')
    
    centola_data, merged_data = load_data(centola_filepath, merged_filepath)
    centola_data = filter_last_round(centola_data)
    new_rows = calculate_magnitude_threshold(centola_data)
    merged_data = append_to_merged(merged_data, new_rows)
    merged_data = convert_to_fraction(merged_data)
    save_merged_data(merged_data, merged_filepath)
    
    return merged_data
    print("Data has been successfully processed and appended to the merged DataFrame.")

if __name__ == "__main__":
    merged = main()

