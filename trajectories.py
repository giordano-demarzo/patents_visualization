#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 16:08:37 2024

@author: giordano
"""

import pandas as pd
import numpy as np
import glob
from scipy.interpolate import interp1d

# Step 1: Load all data files
def load_data():
    data_list = []
    csv_files = glob.glob('data/codes_data/code_*.csv')
    print(csv_files)
    for file in csv_files:
        year = int(file.split('code_')[1].split('.')[0])
        df = pd.read_csv(file)
        df['year'] = year
        data_list.append(df)
    full_data = pd.concat(data_list, ignore_index=True)
    return full_data

# Step 2: Precompute trajectories
def precompute_trajectories(full_data):
    # Group data by 'code'
    grouped = full_data.groupby('code')
    smooth_data = []

    for code, group in grouped:
        group = group.sort_values('year')
        if len(group) >= 2:
            years = group['year']
            x = group['x']
            y = group['y']

            # Create interpolation functions
            interp_years = np.arange(years.min(), years.max() + 1)
            f_x = interp1d(years, x, kind='linear', fill_value='extrapolate')
            f_y = interp1d(years, y, kind='linear', fill_value='extrapolate')

            # Compute smooth trajectories
            x_smooth = f_x(interp_years)
            y_smooth = f_y(interp_years)

            code_smooth_df = pd.DataFrame({
                'year': interp_years,
                'x_smooth': x_smooth,
                'y_smooth': y_smooth,
                'code': code,
                'name': group['name'].iloc[0]
            })

            smooth_data.append(code_smooth_df)

    if smooth_data:
        traj_df = pd.concat(smooth_data, ignore_index=True)
        return traj_df
    else:
        return pd.DataFrame()

# Step 3: Save precomputed trajectories
if __name__ == '__main__':
    full_data = load_data()
    trajectory_data = precompute_trajectories(full_data)
    # Save to Parquet format for efficient storage and quick loading
    trajectory_data.to_parquet('data/precomputed_trajectories.parquet', index=False)