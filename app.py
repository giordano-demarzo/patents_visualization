# app.py

import pandas as pd
import numpy as np
import sqlite3
import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
import dash_bootstrap_components as dbc  # For modal components

# Initialize Dash app with Bootstrap stylesheet
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server  # Expose the WSGI server

# Database connection
def get_db_connection():
    conn = sqlite3.connect('patents.db')
    return conn

# Function to get data boundaries
def get_data_bounds():
    conn = get_db_connection()
    query = "SELECT MIN(x), MAX(x), MIN(y), MAX(y) FROM patents"
    cursor = conn.cursor()
    cursor.execute(query)
    result = cursor.fetchone()
    conn.close()
    if result:
        x_min, x_max, y_min, y_max = result
        # Ensure that bounds are numbers
        x_min = float(x_min) if x_min is not None else -100
        x_max = float(x_max) if x_max is not None else 100
        y_min = float(y_min) if y_min is not None else -100
        y_max = float(y_max) if y_max is not None else 100
        return x_min, x_max, y_min, y_max
    else:
        # Default values if no data
        return -100, 100, -100, 100

# Get data boundaries at startup
X_MIN, X_MAX, Y_MIN, Y_MAX = get_data_bounds()

# Helper functions
def get_data_from_db(xmin, xmax, ymin, ymax, year_range, exclude_ids=[], limit=200):
    conn = get_db_connection()
    placeholders = ','.join(['?'] * len(exclude_ids))
    params = [xmin, xmax, ymin, ymax, year_range[0], year_range[1]]
    query = """
        SELECT rowid AS id, x, y, title, abstract, codes, year FROM patents
        WHERE x BETWEEN ? AND ?
        AND y BETWEEN ? AND ?
        AND year BETWEEN ? AND ?
    """
    if exclude_ids:
        query += f" AND rowid NOT IN ({placeholders})"
        params.extend(exclude_ids)
    query += """
        ORDER BY RANDOM()
        LIMIT ?
    """
    params.append(limit)
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df

def search_patent_in_db(title):
    conn = get_db_connection()
    cursor = conn.cursor()
    query = "SELECT rowid AS id, x, y FROM patents WHERE title = ? LIMIT 1;"
    cursor.execute(query, (title,))
    result = cursor.fetchone()
    conn.close()

    if result:
        id, x, y = result
        return {'id': id, 'x': x, 'y': y}
    else:
        return None

def get_patent_details_by_id(patent_id):
    conn = get_db_connection()
    query = """
        SELECT title, abstract FROM patents
        WHERE rowid = ? LIMIT 1;
    """
    cursor = conn.cursor()
    cursor.execute(query, (patent_id,))
    result = cursor.fetchone()
    conn.close()

    if result:
        title, abstract = result
        return {'title': title, 'abstract': abstract}
    else:
        return None

# Dash app layout
app.layout = html.Div([
    html.H1("Patent Visualization Platform", style={'textAlign': 'center'}),
    dcc.Graph(
        id='patent-graph',
        config={'displayModeBar': True, 'scrollZoom': True},
        style={'height': '80vh'}
    ),
    html.Div([
        html.Label("Filter by Year Range:"),
        dcc.RangeSlider(
            id='year-slider',
            min=1900,
            max=2023,
            value=[2000, 2023],
            marks={str(year): str(year) for year in range(1900, 2024, 10)},
            tooltip={"placement": "bottom", "always_visible": True}
        )
    ], style={'width': '80%', 'margin': 'auto'}),
    html.Div([
        dcc.Input(
            id='search-input',
            type='text',
            placeholder='Search Patent Title',
            style={'width': '300px'}
        ),
        html.Button('Search', id='search-button', n_clicks=0)
    ], style={'textAlign': 'center', 'marginTop': '20px'}),
    html.Div(id='search-output', style={'textAlign': 'center', 'color': 'red', 'marginTop': '10px'}),

    # Store components
    dcc.Store(id='searched-patent-coords', data=None),
    dcc.Store(id='graph-data', data=None),
    dcc.Store(id='prev-viewport', data=None),
    dcc.Store(id='previous-searched-id', data=None),

    # Modal component for displaying patent details
    dbc.Modal(
        [
            dbc.ModalHeader(
                [
                    dbc.ModalTitle(id='modal-title'),
                    dbc.Button(
                        "×",
                        id="modal-close-button",
                        n_clicks=0,
                        className="btn-close",
                        style={
                            "position": "absolute",
                            "top": "15px",
                            "right": "15px",
                            "background": "transparent",
                            "border": "none",
                            "fontSize": "1.5rem",
                            "cursor": "pointer",
                        },
                    ),
                ],
                style={"position": "relative"},
                close_button=False,  # Disable the built-in close button
            ),
            dbc.ModalBody(id='modal-body'),
        ],
        id='modal',
        is_open=False,
    )
])

# Update graph based on viewport, year slider, and searched patent
@app.callback(
    Output('patent-graph', 'figure'),
    Output('graph-data', 'data'),
    Output('prev-viewport', 'data'),
    Output('previous-searched-id', 'data'),
    Input('patent-graph', 'relayoutData'),
    Input('year-slider', 'value'),
    Input('searched-patent-coords', 'data'),
    State('patent-graph', 'figure'),
    State('graph-data', 'data'),
    State('prev-viewport', 'data'),
    State('previous-searched-id', 'data')
)
def update_graph(relayoutData, year_range, searched_coords, existing_fig, existing_data, prev_viewport, previous_searched_id):
    ctx = dash.callback_context

    # Determine what triggered the callback
    triggered_input = ctx.triggered[0]['prop_id'].split('.')[0]

    # Determine current viewport bounds
    if relayoutData and ('xaxis.range[0]' in relayoutData):
        xmin = relayoutData['xaxis.range[0]']
        xmax = relayoutData['xaxis.range[1]']
        ymin = relayoutData['yaxis.range[0]']
        ymax = relayoutData['yaxis.range[1]']
    else:
        xmin, xmax = X_MIN, X_MAX
        ymin, ymax = Y_MIN, Y_MAX

    # Adjust viewport bounds to stay within data limits and prevent zooming out beyond initial view
    x_range = xmax - xmin
    y_range = ymax - ymin
    full_x_range = X_MAX - X_MIN
    full_y_range = Y_MAX - Y_MIN

    # Prevent zooming out beyond initial view
    if x_range > full_x_range:
        xmin, xmax = X_MIN, X_MAX
    if y_range > full_y_range:
        ymin, ymax = Y_MIN, Y_MAX

    # Prevent panning beyond data boundaries
    xmin = max(xmin, X_MIN)
    xmax = min(xmax, X_MAX)
    ymin = max(ymin, Y_MIN)
    ymax = min(ymax, Y_MAX)

    # Initialize variables
    zoomed_out = False
    zoomed_in = False
    panned = False
    year_slider_changed = False
    search_triggered = False

    # Detect zooming, panning, and searching
    if prev_viewport is not None and triggered_input == 'patent-graph':
        prev_xmin, prev_xmax = prev_viewport['xmin'], prev_viewport['xmax']
        prev_ymin, prev_ymax = prev_viewport['ymin'], prev_viewport['ymax']
        prev_area = (prev_xmax - prev_xmin) * (prev_ymax - prev_ymin)
        current_area = (xmax - xmin) * (ymax - ymin)
        if current_area > prev_area:
            zoomed_out = True
        elif current_area < prev_area:
            zoomed_in = True
        elif (xmin != prev_xmin) or (xmax != prev_xmax) or (ymin != prev_ymin) or (ymax != prev_ymax):
            panned = True

    # Detect if year slider changed
    if triggered_input == 'year-slider':
        year_slider_changed = True

    # Detect if a search occurred
    if triggered_input == 'searched-patent-coords' and searched_coords:
        search_triggered = True

    # Number of points to limit
    max_total_points = 1000  # Adjust as needed

    # Existing data handling based on interaction
    if zoomed_out or year_slider_changed or search_triggered:
        # Start fresh, keep only the searched patent if any
        existing_ids = []
        if searched_coords:
            searched_id = searched_coords['id']
            # Fetch the searched patent
            conn = get_db_connection()
            query = """
                SELECT rowid AS id, x, y, title, abstract, codes, year FROM patents
                WHERE rowid = ? LIMIT 1;
            """
            df_searched = pd.read_sql_query(query, conn, params=(searched_id,))
            conn.close()
            df_searched['is_searched'] = True
            existing_df = df_searched
            existing_ids = df_searched['id'].tolist()

            # Center the visualization on the searched patent
            x = searched_coords['x']
            y = searched_coords['y']
            delta_x = (X_MAX - X_MIN) / 10  # Adjust delta as needed
            delta_y = (Y_MAX - Y_MIN) / 10  # Adjust delta as needed
            xmin = max(x - delta_x, X_MIN)
            xmax = min(x + delta_x, X_MAX)
            ymin = max(y - delta_y, Y_MIN)
            ymax = min(y + delta_y, Y_MAX)
        else:
            existing_df = pd.DataFrame(columns=['id', 'x', 'y', 'title', 'abstract', 'codes', 'year', 'is_searched'])

        # Fetch new random data within the current viewport
        points_to_fetch = max_total_points - len(existing_ids)
        if points_to_fetch > 0:
            new_data = get_data_from_db(
                xmin, xmax, ymin, ymax, year_range, exclude_ids=existing_ids, limit=points_to_fetch
            )
            # Initialize 'is_searched' column in new data
            new_data['is_searched'] = False
            # Combine the new data with existing (searched patent)
            combined_data = pd.concat([existing_df, new_data], ignore_index=True, sort=False).drop_duplicates(subset='id')
            combined_data['is_searched'] = combined_data['is_searched'].fillna(False)
        else:
            combined_data = existing_df.copy()

    elif zoomed_in or panned:
        # Zoomed in or panned: Keep existing points within new viewport and add new points
        if existing_data:
            existing_df = pd.DataFrame(existing_data)
            # Keep existing points within the new viewport
            existing_df = existing_df[
                (existing_df['x'] >= xmin) & (existing_df['x'] <= xmax) &
                (existing_df['y'] >= ymin) & (existing_df['y'] <= ymax)
            ]
            existing_ids = existing_df['id'].tolist()
        else:
            existing_df = pd.DataFrame(columns=['id', 'x', 'y', 'title', 'abstract', 'codes', 'year', 'is_searched'])
            existing_ids = []

        # Calculate number of points to add
        points_to_add = max_total_points - len(existing_df)
        if points_to_add > 0:
            # Fetch new data within the new viewport, excluding existing ids
            new_data = get_data_from_db(
                xmin, xmax, ymin, ymax, year_range, exclude_ids=existing_ids, limit=points_to_add
            )
            # Initialize 'is_searched' column in new data
            new_data['is_searched'] = False
            # Combine existing data and new data
            combined_data = pd.concat([existing_df, new_data], ignore_index=True, sort=False).drop_duplicates(subset='id')
            combined_data['is_searched'] = combined_data['is_searched'].fillna(False)
        else:
            combined_data = existing_df.copy()
    else:
        # Other interactions or initial load
        if existing_data:
            combined_data = pd.DataFrame(existing_data)
        else:
            # Fetch initial data if no existing data
            combined_data = get_data_from_db(
                xmin, xmax, ymin, ymax, year_range, limit=max_total_points
            )
            combined_data['is_searched'] = False

    # Ensure 'id' column is of integer type
    if not combined_data.empty:
        combined_data['id'] = combined_data['id'].astype(int)

        # Reset 'is_searched' flag for previous searched patent
        if previous_searched_id is not None:
            combined_data.loc[combined_data['id'] == previous_searched_id, 'is_searched'] = False

        # Handle the new searched patent
        if searched_coords:
            searched_id = searched_coords['id']
            # Add the new searched patent to combined_data if not present
            if searched_id not in combined_data['id'].values:
                conn = get_db_connection()
                query = """
                    SELECT rowid AS id, x, y, title, abstract, codes, year FROM patents
                    WHERE rowid = ? LIMIT 1;
                """
                df_searched = pd.read_sql_query(query, conn, params=(searched_id,))
                conn.close()
                df_searched['is_searched'] = True
                combined_data = pd.concat([combined_data, df_searched], ignore_index=True, sort=False).drop_duplicates(subset='id')
                combined_data['is_searched'] = combined_data['is_searched'].fillna(False)
            else:
                # Set 'is_searched' flag for the searched patent
                combined_data.loc[combined_data['id'] == searched_id, 'is_searched'] = True

            # Update previous_searched_id
            previous_searched_id = searched_id

    else:
        # If combined_data is empty, create an empty DataFrame with necessary columns
        combined_data = pd.DataFrame(columns=['id', 'x', 'y', 'title', 'abstract', 'codes', 'year', 'is_searched'])
        previous_searched_id = None

    # Create the figure
    if not combined_data.empty:
        fig = px.scatter(
            combined_data,
            x='x',
            y='y',
            custom_data=['id'],  # Include 'id' in custom data
            hover_data={'x': False, 'y': False, 'title': True, 'year': True, 'codes': True},
            title='Patent Visualization',
            labels={'x': 'X Coordinate', 'y': 'Y Coordinate'},
        )

        # Adjust marker properties to highlight the searched patent
        colors = np.where(combined_data['is_searched'], 'red', 'blue')
        sizes = np.where(combined_data['is_searched'], 12, 6)
        fig.update_traces(marker=dict(color=colors, size=sizes))

        # Update figure layout
        fig.update_layout(
            clickmode='event+select',
            dragmode='pan',
            uirevision='constant',
            xaxis=dict(
                range=[xmin, xmax],
                constrain='range',
                fixedrange=False,
                autorange=False,
                mirror=True,
                ticks='outside',
                showline=True,
                linewidth=1,
                linecolor='black',
                gridcolor='lightgray',
                zeroline=False,
            ),
            yaxis=dict(
                range=[ymin, ymax],
                constrain='range',
                fixedrange=False,
                autorange=False,
                mirror=True,
                ticks='outside',
                showline=True,
                linewidth=1,
                linecolor='black',
                gridcolor='lightgray',
                zeroline=False,
            )
        )
    else:
        # If no data, create an empty figure
        fig = px.scatter(
            pd.DataFrame(columns=['x', 'y']),
            x='x',
            y='y',
            title='Patent Visualization',
            labels={'x': 'X Coordinate', 'y': 'Y Coordinate'},
        )
        fig.update_layout(
            clickmode='event+select',
            dragmode='pan',
            uirevision='constant'
        )

    # Prepare data to store in 'graph-data' store
    data_to_store = combined_data.to_dict('records')

    # Store the current viewport bounds
    viewport_data = {'xmin': xmin, 'xmax': xmax, 'ymin': ymin, 'ymax': ymax}

    return fig, data_to_store, viewport_data, previous_searched_id

# Search functionality
@app.callback(
    Output('search-output', 'children'),
    Output('searched-patent-coords', 'data'),
    Input('search-button', 'n_clicks'),
    State('search-input', 'value'),
    prevent_initial_call=True
)
def search_patent(n_clicks, title):
    if not title:
        return "Please enter a patent title.", None

    coords = search_patent_in_db(title)
    if coords:
        id = coords['id']
        x = coords['x']
        y = coords['y']
        return "", {'id': id, 'x': x, 'y': y}
    else:
        return "Patent not found.", None

# Callback to display modal with patent details when a point is clicked
@app.callback(
    Output('modal', 'is_open'),
    Output('modal-title', 'children'),
    Output('modal-body', 'children'),
    Input('patent-graph', 'clickData'),
    Input('modal-close-button', 'n_clicks'),
    State('modal', 'is_open'),
    prevent_initial_call=True
)
def display_patent_details(clickData, n_clicks_close, is_open_state):
    ctx = dash.callback_context

    if not ctx.triggered:
        return dash.no_update, dash.no_update, dash.no_update

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger_id == 'patent-graph' and clickData:
        # Get the 'id' from customdata
        point = clickData['points'][0]
        patent_id = point['customdata'][0]
        details = get_patent_details_by_id(patent_id)
        if details:
            title = details['title']
            abstract = details['abstract']
            return True, title, html.P(abstract)
        else:
            return dash.no_update, dash.no_update, dash.no_update
    elif trigger_id == 'modal-close-button':
        return False, dash.no_update, dash.no_update
    else:
        return dash.no_update, dash.no_update, dash.no_update

if __name__ == '__main__':
    # Run the Dash app
    app.run_server(debug=False)