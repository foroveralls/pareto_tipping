import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import os
from collections import deque

#N = int(filename.split('_')[3].split('.')[0].replace("n", ""))

def process_file(filename):
    topology = filename.split('_')[2]
    N = int(filename.split('_')[3].split('.')[0].replace("n", ""))

    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    names = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 3:
            name1, name2, _ = parts
            if name1.lower() != 'response_not_given':
                names.append(name1.lower())
            if name2.lower() != 'response_not_given':
                names.append(name2.lower())

    # Calculate name frequencies for each Round Played
    interactions_per_round = N // 2
    max_rounds = 40 if N == 48 else 30  # Adjust based on N
    name_freq = []

    for round in range(min(max_rounds, len(names) // interactions_per_round)):
        start = round * interactions_per_round
        end = start + interactions_per_round
        round_names = names[start:end]
        total_names = len(round_names)
        freq = {name: round_names.count(name) / total_names for name in set(round_names) if name != 'response_not_given'}
        name_freq.append(freq)

    rounds_played = len(name_freq)

    # Plot the results
    plt.figure(figsize=(10, 6))

    unique_names = set(name for freq in name_freq for name in freq.keys() if name != 'response_not_given')
    for name in unique_names:
        freq_over_time = [freq.get(name, 0) for freq in name_freq]
        plt.plot(range(rounds_played), freq_over_time, label=name if max(freq_over_time) > 0.1 else '')

    plt.xlabel('Rounds Played')
    plt.ylabel('Norm Frequency')
    plt.title(f'Evolving Ecology of Norms ({N}, {topology.capitalize()})')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.ylim(0, 1)  # Set y-axis limits from 0 to 1
    plt.tight_layout()
    plt.show()

    # Find the norm with the highest frequency at the end
    final_freq = name_freq[-1]
    max_freq_norm = max(final_freq, key=final_freq.get)
    max_freq = final_freq[max_freq_norm]

    # Create dataframe for the norm with highest frequency
    df_list = []
    for t, freq in enumerate(name_freq):
        df_list.append({
            'x': t,
            'y': freq.get(max_freq_norm, 0),
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