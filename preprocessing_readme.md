# Preprocessing and Visualization Script - README

This document provides detailed instructions for using the preprocessing and visualization pipeline for patents data. It covers all necessary requirements, input and output files, and how to run the script effectively.

## Requirements

To run this script, you need to install the following Python libraries:

- pandas
- numpy
- os
- sqlite3
- scikit-learn
- matplotlib
- seaborn
- umap-learn
- nltk

## Folder Structure

Before running the script, ensure you have the following folder structure in place:

graphql
Copia codice
root/
│
|──data/
   ├── data_gt2018_embeddings_llama38B.parquet/ 
   │   ├── partition=0/
   │   ├── partition=2/
   │   ├── ... (up to partition=16)
   ├── IPC_DESCR_DEF_AGGREGATE/  # Contains 8 dictionary `.pkl` files for code descriptions
   └── average_embeddings_2019_2023_aggregated_num_patents.pkl

##Input Files
Embeddings Data: You need a folder (data_gt2018_embeddings_llama38B.parquet/) containing partitioned parquet files with patent embeddings. Each partition folder should contain a parquet file.
IPC Description Dictionaries: A folder (IPC_DESCR_DEF_AGGREGATE/) with 8 dictionary .pkl files mapping tech codes to their descriptions.
Average Embeddings File: A file (average_embeddings_2019_2023_aggregated_num_patents.pkl) that contains aggregated embedding data for tech codes between 2019-2023.
Similar Codes Data: A file (top_10_codes_2019_2023_llama8b_abstracts.pkl) mapping codes to their similar codes based on the embeddings.

##Output Files
SQLite Database (patents.db): Contains a table of patents with reduced 2D embedding coordinates, patent titles, abstracts, IPC codes, and years.
SQLite Database with Topics (patents_topic.db): This adds a dominant topic and topic title for each patent, based on LDA.
Yearly CSV Files: These files are saved in the codes_data/ directory and contain tech codes, 2D UMAP coordinates, and code descriptions for each year.
Precomputed Trajectories File (precomputed_trajectories.parquet): This file stores smoothed trajectories for each tech code, computed from yearly embeddings.
Precomputed Similar Codes (precomputed_similar_codes.pkl): Stores data required to visualize similar codes, including their coordinates and hover text for tooltips.

##Running the Script
**Data Sampling and Merging:** The first section of the script reads in patent embeddings from the parquet files, samples 2500 rows from each partition, and merges them into a single DataFrame.

Output: The merged DataFrame is printed and used for subsequent t-SNE/UMAP dimensionality reduction.

**Dimensionality Reduction:** The script applies UMAP to reduce the patent embeddings to 2D space for visualization.

Output: A scatter plot of the embeddings is generated.

**Exporting to SQLite:** The reduced embeddings and metadata (e.g., title, abstract, year) are exported to a SQLite database (patents.db).

**Topic Modeling with LDA:** The script uses Latent Dirichlet Allocation (LDA) to extract topics from patent abstracts. These topics are assigned to each patent in the database.

Output: A database with topic information (patents_topic.db) and a scatter plot of patents by topic.

**Yearly CSV Creation:** The script generates CSV files with UMAP-reduced coordinates for each year.

Output: Files saved in the codes_data/ directory.

**Precomputing Trajectories**: Smooth trajectories of tech codes over time are computed and saved in a parquet file.

Output: precomputed_trajectories.parquet.

**Precomputing Similar Codes:** The final step generates precomputed data for visualizing similar tech codes. This is saved as a pickle file.

Output: precomputed_similar_codes.pkl.

##Notes
Make sure your system has enough memory to process large patent datasets.
Adjust sampling sizes or UMAP parameters as needed for performance optimization.
For visualization, install seaborn and ensure that matplotlib works on your system.
