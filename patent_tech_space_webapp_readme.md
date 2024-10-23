
# Patent and Technology Space Visualization Web App - README

This README provides detailed instructions on setting up and running the Patent and Technology Space visualization web app locally using Dash, Jekyll, and Python. It includes the folder structure, necessary dependencies, and the steps to get everything working.

## Requirements

Before running the web app, ensure you have the following installed:

- Python 3.8+
- `dash`
- `dash-bootstrap-components`
- `plotly`
- `pandas`
- `sqlite3`
- `joblib`
- `plotly.express`
- `umap-learn`

To install the necessary Python libraries, run:

```bash
pip install dash dash-bootstrap-components plotly pandas sqlite3 joblib umap-learn
```

### File Structure

Ensure the following file and folder structure is in place for the app to work:

```
root/
│
├── app.py                          # Main app instance initialization
├── index.py                        # App routing and layout definitions
├── home.py                         # Home page layout and visualization
├── vis_patents.py                  # Patent space visualization logic
├── vis_codes.py                    # Technology space visualization logic
│
│
├── data/
│   ├── tech_code_embeddings_2d_with_shap.pkl  # Embedding data for home.py
│   ├── patents_topic.db            # SQLite database for patents
│   ├── precomputed_trajectories.parquet  # Trajectory data for tech codes
│   ├── top_10_codes_2019_2023_llama8b_abstracts.pkl  # Precomputed similar codes
│   └── codes_data/                 # Folder containing CSV files for yearly code embeddings
│       ├── code_2019.csv
│       ├── code_2020.csv
│       ├── code_2021.csv
│       ├── code_2022.csv
│       └── code_2023.csv
```

### Data Files

- **tech_code_embeddings_2d_with_shap.pkl**: Contains t-SNE embeddings for technological codes along with SHAP values used in `home.py`.
- **patents_topic.db**: SQLite database for patents, used in `vis_patents.py`.
- **precomputed_trajectories.parquet**: Contains precomputed smooth trajectories of technological codes over time, used in `vis_codes.py`.
- **top_10_codes_2019_2023_llama8b_abstracts.pkl**: Precomputed similar technological codes and their similarity scores.
- **codes_data/**: Contains CSV files with tech code embeddings for each year from 2019 to 2023.

## Running the Web App Locally

1. **Ensure the Folder Structure**: Make sure the necessary files and folders, as outlined above, are correctly structured.

2. **Install Dependencies**: Run the following command to install the required Python libraries:

   ```bash
   pip install -r requirements.txt
   ```

   Or use the individual install command provided earlier.

3. **Run the Web App**: Use Python to run the app.

   ```bash
   python index.py
   ```

4. **Access the Web App**: Open your browser and go to `http://127.0.0.1:8050/` to access the app.

## Explanation of Scripts

### `app.py`

This script initializes the Dash app and sets the Bootstrap theme for styling. The app is also set up to suppress callback exceptions to allow different pages to have independent callbacks.

### `index.py`

This script defines the overall layout of the web app and routes between the different pages:
- `/`: Home page visualization of technological code embeddings.
- `/vis_patents`: Patent space visualization.
- `/vis_codes`: Technology space visualization.

### `home.py`

This script defines the layout and logic for the home page. It uses a t-SNE projection of the technological code embeddings and provides an interactive scatter plot where users can hover over points to see corresponding patent titles and SHAP values.

### `vis_patents.py`

This script is responsible for visualizing the patent space, displaying a 2D projection of patent embeddings. Users can filter patents by year using a slider, search for patents by title, and click on points to see patent details.

### `vis_codes.py`

This script visualizes the technology space and allows users to see the UMAP projection of technological codes, with trajectories showing how codes evolve over time. Users can filter by year, search for specific codes, and explore similar codes using precomputed trajectories.

## Running Jekyll for Static Hosting (Optional)

If you want to integrate this web app with a Jekyll site, ensure that the `_config.yml` in your Jekyll site allows serving of Python apps. You can then deploy the Dash app using a server like Gunicorn or Render for hosting.

For local static site testing:
```bash
bundle exec jekyll serve
```

## Additional Information

- **Customization**: The app uses Bootstrap for layout and Plotly for visualizations, making it easy to extend or modify the styles.
- **Deployment**: The app is ready for deployment to platforms that support Dash apps, such as Heroku or Render. Be sure to include necessary files like `requirements.txt` for installation.

## License

This project is licensed under the MIT License.
