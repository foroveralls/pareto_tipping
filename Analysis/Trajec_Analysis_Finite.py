import pandas as pd
import numpy as np
from collections import defaultdict


#%%


def calculate_second_derivative_adaptive(x, y, idx):
    if idx <= 0 or idx >= len(x) - 1:
        return 0  # The second derivative is not defined at the endpoints
    dx1 = x[idx] - x[idx - 1]
    dx2 = x[idx + 1] - x[idx]
    d2 = 2 * ((y[idx] - y[idx - 1]) / dx1 - (y[idx + 1] - y[idx]) / dx2) / (dx1 + dx2)
    return d2


def identify_trajectories(df):
    trajectories = defaultdict(list)
    last_trajectory_attrs = None
    for idx, row in df.iterrows():
        current_attrs = tuple(row.drop(['x', 'y']))
        if current_attrs != last_trajectory_attrs:
            trajectories[current_attrs].append({'start': idx, 'end': idx})
        else:
            trajectories[current_attrs][-1]['end'] = idx
        last_trajectory_attrs = current_attrs
    return trajectories


def process_trajectories(df, trajectories):
    result_data = []
    for attrs, traj_list in trajectories.items():
        for traj in traj_list:
            traj_df = df.iloc[traj['start']:traj['end'] + 1]
            x = traj_df['x'].to_numpy()
            y = traj_df['y'].to_numpy()
            d2_adaptive = np.array([calculate_second_derivative_adaptive(x, y, i) for i in range(1, len(x) - 1)])
            positive_idx_adaptive = np.where(d2_adaptive > 0)[0]
            if len(positive_idx_adaptive) > 0:
                max_d2_idx = positive_idx_adaptive[np.argmax(d2_adaptive[positive_idx_adaptive])]
                max_d2_y = y[1:-1][max_d2_idx]
                result_data.append({'ref': file_name, 'y_max_d2': max_d2_y, **dict(zip(df.columns.drop(['x', 'y']), attrs))})
    return pd.DataFrame(result_data)


def main(file_name):
    df = pd.read_csv(file_name)
    trajectories = identify_trajectories(df)
    result_df = process_trajectories(df, trajectories)
    print(result_df)
    # Optionally, you can save the result_df to a new CSV file or perform other tasks as needed.


if __name__ == "__main__":
    file_name = "your_file_name_here.csv"  # Replace with your actual file name
    main(file_name)
    

        