#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 16:02:16 2024

@author: giordano
"""

import pandas as pd
import os

# Define the base folder containing the data partitions
base_folder = 'data_2006_2010_embeddings_abstractclaims_combined_llama38B.parquet'

# Initialize an empty list to store dataframes
dataframes = []

# Loop through each partition folder and read the parquet file
for partition in range(1):
    partition_folder = os.path.join(base_folder, f'partition={partition}')
    parquet_file = os.path.join(partition_folder, os.listdir(partition_folder)[0])  # Assumes there's only one file per partition
    df = pd.read_parquet(parquet_file)
    dataframes.append(df)

# Concatenate all the dataframes into a single dataframe
merged_df = pd.concat(dataframes, ignore_index=True)

# Display or save the merged dataframe
print(merged_df.head())  # Show the first few rows of the merged dataframe

#%%TSNE 

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Assuming you have the merged dataframe with the 'embedding' column
# merged_df = pd.read_parquet('merged_data_2006_2010_embeddings.parquet')

# Convert the 'embedding' column to a NumPy array
embeddings = np.array(merged_df['embedding'].to_list())

# Apply t-SNE to reduce the embeddings to 2 dimensions
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)

# Plot the 2D t-SNE embeddings
plt.figure(figsize=(10, 8))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=10, alpha=0.7)
plt.title('t-SNE visualization of Embeddings', fontsize=16)
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.grid(True)
plt.show()

#%%EXPORT TO DB

import pandas as pd
import sqlite3

# Assuming you already have the merged_df dataframe and the t-SNE results (embeddings_2d)

# Add the x and y coordinates from t-SNE to the dataframe
merged_df['x'] = embeddings_2d[:, 0]
merged_df['y'] = embeddings_2d[:, 1]

# Create a new dataframe with the required column names
final_df = merged_df[['numeric_id', 'x', 'y', 'APPLN_TITLE', 'combined_text', 'IPC', 'APPLN_YR']].copy()
final_df.columns = ['id', 'x', 'y', 'title', 'abstract', 'codes', 'year']

# Connect to an SQLite database (or create it if it doesn't exist)
conn = sqlite3.connect('patents.db')

# Export the dataframe to a table in the SQLite database
final_df.to_sql('patents', conn, if_exists='replace', index=False)

# Close the connection to the database
conn.close()

print("Data successfully exported to patents.db")