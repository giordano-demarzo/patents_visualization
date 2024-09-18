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

def generate_synthetic_dataset(num_records=1000000):
    np.random.seed(42)  # For reproducibility

    # Generate random coordinates
    x_coords = np.random.uniform(-100, 100, num_records)
    y_coords = np.random.uniform(-100, 100, num_records)

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