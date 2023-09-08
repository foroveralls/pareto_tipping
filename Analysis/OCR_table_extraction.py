# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 22:59:05 2023

@author: everall
"""

import pandas as pd
import pdfplumber
import os
import string

# Path to your pdf
files = "../Data/Experimental/Centola-Becker-Brackbill-Baronchelli_Complete-Data_2018_size_threshold.pdf"

#%%
with pdfplumber.open(files) as pdf:
# Get the first page
    first_page = pdf.pages[0]

# Extract the table from the first page
    table = first_page.extract_table(table_settings={"horizontal_strategy": "text", "vertical_strategy": "text"})
    #table = table[5:]
# Convert the table data into a pandas dataframe
    df = pd.DataFrame(table[6:], columns=table[5])
    df.drop(columns =df.columns[0], axis =1, inplace = True)
    nan_value = float("NaN")
    df.replace("", nan_value, inplace=True)
    df = df.dropna(axis = 0)
    df = df.reset_index(drop = True)
    df[["adoption_established", "adoption_other", "adoption_alternative"]] = df[[
"adoption_established", "adoption_other", "adoption_alternative"]].apply(lambda x: x.str.rstrip("%")).astype("float")
    df.apply(pd.to_numeric).plot(x = "rounds_played", y = "adoption_established" )
#%%



# for i in os.listdir(files):
#     with pdfplumber.open(i) as pdf:
#     # Get the first page
#         first_page = pdf.pages[0]

#     # Extract the table from the first page
#         table = first_page.extract_table(table_settings={"horizontal_strategy": "text", 
#                                                      "vertical_strategy": "text"})

#     # Convert the table data into a pandas dataframe
#         df = pd.DataFrame(table[1:], columns=table[0])

#     # Save the dataframe to a csv file
#         df.to_csv('output.csv', index=False)