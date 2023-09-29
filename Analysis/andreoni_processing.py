# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 23:24:31 2023

@author: everall
"""

import pandas as pd
import numpy as np

#%%

def load_data(tt_filepath, merged_filepath):
    tt_data = pd.read_csv(tt_filepath)
    merged_data = pd.read_csv(merged_filepath)
    return tt_data, merged_data


def filter_data(tt_data):
    return tt_data[tt_data['name of treatment'].str.contains('TT')]


def calculate_abandonment_percentages(tt_data):
    unique_treatments = tt_data['name of treatment'].unique()
    recalculated_session_results = []
    
    for treatment in unique_treatments:
        treatment_data = tt_data[tt_data['name of treatment'] == treatment]
        unique_sessions = treatment_data['session'].unique()
        
        for session in unique_sessions:
            session_data = treatment_data[treatment_data['session'] == session]
            max_period = session_data['period'].max()
            start_period = max(1, max_period - 4)
            last_five_periods_data = session_data[session_data['period'] >= start_period]
            
            num_subjects = len(last_five_periods_data)
            num_abandoned_blue = len(last_five_periods_data[last_five_periods_data['color choice'] != 1])
            
            if num_subjects > 0:
                percentage_abandoned_blue = (num_abandoned_blue / num_subjects) * 100
                threshold = 40 if treatment == 'TT-Endo' else int(treatment[-2:])
                
                result = {
                    'treatment': treatment,
                    'session': session,
                    'threshold': threshold,
                    'percentage_abandoned_blue': percentage_abandoned_blue
                }
                recalculated_session_results.append(result)
    
    return recalculated_session_results


def update_merged_data(merged_data, recalculated_session_results):
    all_session_recalculated_rows = []
    
    for result in recalculated_session_results:
        new_row = {
            'ref': 'Andreoni_2021',
            'tipping_point_c_t': result['threshold'],
            'attribute': np.nan,
            'value': np.nan,
            'type': 'experimental',
            'magnitude': result['percentage_abandoned_blue']
        }
        all_session_recalculated_rows.append(new_row)
    
    return merged_data.append(pd.DataFrame(all_session_recalculated_rows), ignore_index=True)


def main():
    tt_filepath = '_andreoni_2021_threshold_magnitude_file'
    merged_filepath = 'path_to_Tipping_points_fin_merged_file'
    output_filepath = 'path_to_output_file'
    
    tt_data, merged_data = load_data(tt_filepath, merged_filepath)
    tt_data = filter_data(tt_data)
    recalculated_session_results = calculate_abandonment_percentages(tt_data)
    updated_merged_data = update_merged_data(merged_data, recalculated_session_results)
    
    updated_merged_data.to_csv(output_filepath, index=False)


if __name__ == "__main__":
    main()
