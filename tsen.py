#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 16:02:16 2024

@author: giordano
"""

import pandas as pd
import os

# Define the base folder containing the data partitions
base_folder = 'data/data_gt2018_embeddings_llama38B.parquet'

# Initialize an empty list to store dataframes
dataframes = []

# Loop through each partition folder and read the parquet file
for partition in range(0, 17, 2):
    print(partition)
    partition_folder = os.path.join(base_folder, f'partition={partition}')
    parquet_file = os.path.join(partition_folder, os.listdir(partition_folder)[0])  # Assumes there's only one file per partition
    df = pd.read_parquet(parquet_file)
    
    # Randomly sample 3000 rows from the dataframe
    sampled_df = df.sample(n=2500, random_state=42)  # Set random_state for reproducibility
    
    # Append the sampled dataframe to the list
    dataframes.append(sampled_df)

# Concatenate all the sampled dataframes into a single dataframe
merged_df = pd.concat(dataframes, ignore_index=True)

# Display or save the merged dataframe
print(merged_df.head())  # Show the first few rows of the merged dataframe


#%%TSNE 

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import umap



# Convert the 'embedding' column to a NumPy array
embeddings = np.array(merged_df['embedding'].to_list())


# # Apply t-SNE to reduce the embeddings to 2 dimensions
# tsne = TSNE(n_components=2, random_state=42)
# embeddings_2d = tsne.fit_transform(embeddings)

# Perform UMAP on the embeddings
umap_model = umap.UMAP(n_components=2, n_jobs=4, n_neighbors=15)
embeddings_2d = umap_model.fit_transform(embeddings)

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
final_df = merged_df[['numeric_id', 'x', 'y', 'APPLN_TITLE', 'APPLN_ABSTR', 'IPC', 'APPLN_YR']].copy()
final_df.columns = ['id', 'x', 'y', 'title', 'abstract', 'codes', 'year']
final_df = final_df.sample(20000)

# Connect to an SQLite database (or create it if it doesn't exist)
conn = sqlite3.connect('patents.db')

# Export the dataframe to a table in the SQLite database
final_df.to_sql('patents', conn, if_exists='replace', index=False)

# Close the connection to the database
conn.close()

print("Data successfully exported to patents.db")

#%%LDA TOPICS 

import pandas as pd
import sqlite3
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import numpy as np
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')


# Connect to SQLite database
conn = sqlite3.connect('data/patents.db')

# Load the patents table into a pandas dataframe
df = pd.read_sql_query("SELECT * FROM patents", conn)

# Close the connection
conn.close()
# Use the 'APPLN_TITLE' column for LDA
titles = df['abstract'].fillna('')  # Fill NaN values with empty strings

# Define stopwords
stop_words = list(set(stopwords.words('english')))
additional_stop_words = ['method', 'first', 'second', 'third', 'claim', 'claims', 'according', 'wherein', 'device', 'technology', 'apparatus', 'system', 'machine', 'methods', 'thereof', 'use', 'therefor', 'uses', 'using', 'systems', 'one']
stop_words = stop_words + additional_stop_words 

# Vectorize the titles for LDA using a count vectorizer
vectorizer = CountVectorizer(stop_words=stop_words, max_df=0.95, min_df=2)
title_matrix = vectorizer.fit_transform(titles)


# Define the number of topics
num_topics = 20

# Create and fit the LDA model
lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=20)
lda_topics = lda_model.fit_transform(title_matrix)


# Function to display the top words in each topic
def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print(f"Topic {topic_idx+1}:")
        print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))

no_top_words = 25
display_topics(lda_model, vectorizer.get_feature_names_out(), no_top_words)




#%%VISUALIZE AND EXPORT 
import matplotlib.pyplot as plt
import seaborn as sns

topic_title_dict = {
    0: "Optical Systems and Imaging Technologies",
    1: "Mechanical Devices and Surgical Instruments",
    2: "Vehicle Electronics and Battery Management Systems",
    3: "Electrical Circuits and Power Management",
    4: "Wireless Communication and Signal Processing",
    5: "Data Processing, Machine Learning, and AI Models",
    6: "Chemical Engineering and Material Processing",
    7: "Structural Components and Manufacturing",
    8: "Organic Chemistry and Polymer Technologies",
    9: "Telecommunication Networks and Data Transmission",
    10: "Object Detection and Recognition Technologies",
    11: "Mechanical Assemblies and Industrial Tools",
    12: "Pharmaceutical Formulations and Chemical Compounds",
    13: "Fluid Control and Heat Transfer Systems",
    14: "Battery Technology and Energy Storage",
    15: "Biotechnology and Genetic Engineering",
    16: "User Interface Design and Mobile Computing",
    17: "Robotics, Sensors, and Control Systems",
    18: "Advanced Chemical Compounds and Catalysts",
    19: "Composite Materials and Advanced Manufacturing"
}

# Use the topic distribution to cluster the patents based on their dominant topic
df['dominant_topic'] = lda_topics.argmax(axis=1)

# Assign topic titles to a new column
df['topic_title'] = df['dominant_topic'].map(topic_title_dict)

# Set the color palette based on the number of unique topics
unique_topics = df['dominant_topic'].nunique()
palette = sns.color_palette("hsv", unique_topics)

# Create a scatter plot using the 'x' and 'y' columns, colored by 'topic_title'
plt.figure(figsize=(12, 8))
sns.scatterplot(
    data=df,
    x='x', y='y',
    hue='topic_title',
    palette=palette,
    legend='full',
    alpha=0.7,
    s=50
)

# Customize the plot
plt.title('Patent Clusters by Topic', fontsize=16)
plt.xlabel('UMAP/T-SNE Dimension 1')
plt.ylabel('UMAP/T-SNE Dimension 2')
plt.legend(title='Topic', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)

# Show the plot
plt.show()


# Save the updated dataframe with clusters back to the SQLite database
conn = sqlite3.connect('data/patents_topic.db')
df.to_sql('patents', conn, if_exists='replace', index=False)
conn.close()

print("Data with topics successfully exported to patents.db")

#%%TSNE FOR CODES 
import pandas as pd
import pickle
import numpy as np
import os
from sklearn.manifold import TSNE
import umap

# Load the 8 dictionaries and merge them
dict_path = 'data/IPC_DESCR_DEF_AGGREGATE'
merged_dict = {}

# Iterate over the files in the folder
for file in os.listdir(dict_path):
    if file.endswith('.pkl'):
        with open(os.path.join(dict_path, file), 'rb') as f:
            temp_dict = pickle.load(f)
            merged_dict.update(temp_dict)

# Load the main .pkl file
with open('data/average_embeddings_2019_2023_aggregated_num_patents.pkl', 'rb') as f:
    data = pickle.load(f)

# Create a DataFrame
df = pd.DataFrame(data)

# Filter rows based on 'number of patents' column
df = df[df['number of patents'] > 7]

# Convert the list of embeddings into a NumPy array
embeddings = np.array(df['embedding'].tolist())

# Perform UMAP on the embeddings
umap_model = umap.UMAP(n_components=2, n_jobs=4, n_neighbors=100)
umap_results = umap_model.fit_transform(embeddings)

# Add the UMAP results (x, y) to the DataFrame
df['x'] = umap_results[:, 0]
df['y'] = umap_results[:, 1]

# Map the 'tech_code' column to the 'name' column using the merged dictionary
df['name_full'] = df['tech_code'].map(merged_dict)
df['name'] = df['tech_code'].map(merged_dict)

# Function to truncate names longer than 50 characters and add '...'
def truncate_name(name, max_length=50):
    if isinstance(name, str) and len(name) > max_length:
        return name[:max_length-3] + '...'  # Subtract 3 to account for the length of '...'
    return name

# Apply the truncation function to the 'name' column
df['name'] = df['name'].apply(truncate_name)

# Create a CSV file for each year
for year in df['year'].unique():
    df_year = df[df['year'] == year][['x', 'y', 'tech_code', 'name', 'name_full']].copy()
    df_year.columns = ['x', 'y', 'code', 'name', 'name_full']  # Renaming columns to match required output
    df_year.to_csv(f'data/codes_data/code_{year}.csv', index=False)

print("UMAP and file creation complete.")
#%%COMPUTE TRAJECTORIES FOR CODES 

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