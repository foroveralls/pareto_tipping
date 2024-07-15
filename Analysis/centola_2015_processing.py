import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import os

def process_file(filename):
    topology = filename.split('_')[2]
    N = filename.split('_')[3].split('.')[0].replace('n', '')
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        with open(filename, 'r', encoding='latin-1') as f:
            lines = f.readlines()

    names = []
    timesteps = []
    current_timestep = 0
    unique_names = set()

    for i, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) == 3:
            name1, name2, _ = parts
            names.append(name1.lower())
            names.append(name2.lower())
            
            if name1.lower() != 'response_not_given':
                unique_names.add(name1.lower())
            if name2.lower() != 'response_not_given':
                unique_names.add(name2.lower())
            
            if (i + 1) % 24 == 0:  # Every 24 interactions is a timestep
                current_timestep += 1
                timesteps.extend([current_timestep for _ in range(48)])  # 48 name entries per timestep

    # Calculate name frequencies
    name_freq = {}
    for t in range(max(timesteps) + 1):
        names_at_t = [name for name, timestep in zip(names, timesteps) if timestep == t and name != 'response_not_given']
        total_names = len(names_at_t)
        freq = {name: names_at_t.count(name) / total_names for name in set(names_at_t)}
        name_freq[t] = freq

    # Plot the results
    plt.figure(figsize=(10, 6))

    max_freq_norm = None
    max_freq = 0

    for name in unique_names:
        freq_over_time = [name_freq[t].get(name, 0) for t in range(max(timesteps) + 1)]
        plt.plot(range(max(timesteps) + 1), freq_over_time, label=name if max(freq_over_time) > 0.1 else '')
        
        if max(freq_over_time) > max_freq:
            max_freq = max(freq_over_time)
            max_freq_norm = name

    plt.xlabel('Rounds')
    plt.ylabel('Norm Frequency')
    plt.title(f'Evolving Ecology of Norms ({N}, {topology.capitalize()})')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()
    


    # Create dataframe for the norm with highest frequency
    df_list = []
    for t in range(max(timesteps) + 1):
        df_list.append({
            'x': t,
            'y': name_freq[t].get(max_freq_norm, 0),
            'N': N,
            'Topology': topology,
            'Norm': max_freq_norm
        })
    
    return pd.DataFrame(df_list)

# Process all files
all_files = glob.glob('../Data/Raw/Experimental/Centola_2015_*_n*.txt')
all_data = pd.DataFrame()

for file in all_files:
    print(f"Processing file: {file}")
    df = process_file(file)
    all_data = pd.concat([all_data, df], ignore_index=True)

# Save to CSV
all_data.to_csv('../Data/Experimental/Centola_2015_norm_ecology_data.csv', index=False)

print("Processing complete. Graphs saved in 'plots' directory and data saved to norm_ecology_data.csv")