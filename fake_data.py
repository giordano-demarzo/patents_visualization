#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 14:35:51 2024

@author: giordano
"""

# generate_dataset.py

import pandas as pd
import numpy as np
import sqlite3

def generate_synthetic_dataset(num_records=100000, num_clusters=5):
    np.random.seed(42)  # For reproducibility

    # Number of records per cluster
    records_per_cluster = num_records // num_clusters
    extra_records = num_records % num_clusters

    # Store the generated data
    x_coords = []
    y_coords = []

    # Generate Gaussian clusters
    for _ in range(num_clusters):
        mean = np.random.uniform(-50, 50, 2)  # Random mean for each cluster
        cov = np.diag(np.random.uniform(5, 20, 2))  # Random covariance matrix (diagonal)
        cluster_data = np.random.multivariate_normal(mean, cov, records_per_cluster)
        x_coords.extend(cluster_data[:, 0])
        y_coords.extend(cluster_data[:, 1])

    # Handle any remaining records if the number is not divisible by num_clusters
    if extra_records > 0:
        mean = np.random.uniform(-50, 50, 2)
        cov = np.diag(np.random.uniform(5, 20, 2))
        cluster_data = np.random.multivariate_normal(mean, cov, extra_records)
        x_coords.extend(cluster_data[:, 0])
        y_coords.extend(cluster_data[:, 1])

    # Convert to numpy arrays
    x_coords = np.array(x_coords)
    y_coords = np.array(y_coords)

    # Normalize coordinates to the range (-100, 100)
    def normalize_data(data, new_min, new_max):
        data_min = np.min(data)
        data_max = np.max(data)
        normalized_data = (new_max - new_min) * (data - data_min) / (data_max - data_min) + new_min
        return normalized_data

    x_coords = normalize_data(x_coords, -100, 100)
    y_coords = normalize_data(y_coords, -100, 100)

    # Generate random years between 1900 and 2023
    years = np.random.randint(1900, 2024, num_records)

    # Generate random titles
    titles = [f"Patent Title {i}" for i in range(num_records)]

    # Generate random abstracts
    abstracts = [f"This is the abstract of patent {i}. It covers technology related to area {np.random.randint(1, 100)}." for i in range(num_records)]

    # Generate random technological codes
    codes = [f"Code-{np.random.randint(1000, 9999)}" for _ in range(num_records)]

    # Create a DataFrame
    df = pd.DataFrame({
        'x': x_coords,
        'y': y_coords,
        'title': titles,
        'abstract': abstracts,
        'codes': codes,
        'year': years
    })

    # Save to SQLite database
    conn = sqlite3.connect('patents.db')
    df.to_sql('patents', conn, if_exists='replace', index=False)
    conn.close()
    print(f"Synthetic dataset with {num_records} records created and saved to 'patents.db'.")

if __name__ == "__main__":
    generate_synthetic_dataset()