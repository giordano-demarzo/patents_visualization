#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 20:34:59 2024

@author: giordano
"""

# vis_patents.py

import pandas as pd
import numpy as np
import sqlite3
from dash import dcc, html, Input, Output, State, callback, callback_context
import plotly.express as px
import dash
import dash_bootstrap_components as dbc  # For modal components

# --- Database Connection and Helper Functions ---

# Database connection
def get_db_connection():
    conn = sqlite3.connect('data/patents.db')
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

# --- Dash App Layout ---

layout = html.Div([
    
    html.Div(
    [
        html.H1("The Patent Space", style={'textAlign': 'center', 'color': 'white'}),
        # Home button
        html.Div(
            dcc.Link('Home', href='/', className='home-button', style={'color': 'white'}),
            style={'position': 'absolute', 'top': '15px', 'left': '15px', 'color': 'white'}
        ),
    ],
    style={'backgroundColor': '#2c2c2c', 'position': 'relative'}
    ),

    # Wrapper for graph and slider
    html.Div([
        # Graph
        dcc.Graph(
            id='patents-graph',
            config={'displayModeBar': True, 'scrollZoom': True},
            style={
                'height': '80vh',
                'width': '90%',
                'display': 'inline-block',
                'backgroundColor': '#2c2c2c'  # Same dark grey background for the graph area
            }
        ),

        # Year slider (now vertical and centered on the right)
        html.Div([
            html.Label("", style={'color': 'white', 'textAlign': 'center'}),
            dcc.RangeSlider(
                id='patents-year-slider',
                min=2006,
                max=2010,
                value=[2006, 2010],
                marks={str(year): {'label': str(year), 'style': {'color': 'white'}} for year in range(2006, 2010, 5)},
                tooltip={"placement": "left", "always_visible": True},
                vertical=True,
                verticalHeight=400
            )
        ], style={
            'width': '10%',
            'padding': '20px',
            'display': 'flex',
            'align-items': 'center',
            'justify-content': 'center'
        }),

    ], style={'display': 'flex', 'justifyContent': 'space-between', 'align-items': 'center', 'backgroundColor': '#2c2c2c'}),

    # Search input and button
    html.Div([
        dcc.Input(
            id='patents-search-input',
            type='text',
            placeholder='Search Patent Title',
            style={'width': '300px', 'color': 'white', 'backgroundColor': '#3a3a3a', 'border': '1px solid #555'}
        ),
        html.Button('Search', id='patents-search-button', n_clicks=0, style={'backgroundColor': '#3a3a3a', 'color': 'white', 'border': '1px solid #555'})
    ], style={'textAlign': 'center', 'marginTop': '20px'}),
    
    html.Div(id='patents-search-output', style={'textAlign': 'center', 'color': 'red', 'marginTop': '10px'}),

    # Store components
    dcc.Store(id='patents-searched-patent-coords', data=None),
    dcc.Store(id='patents-graph-data', data=None),
    dcc.Store(id='patents-prev-viewport', data=None),
    dcc.Store(id='patents-previous-searched-id', data=None),
    dcc.Store(id='patents-selected-id', data=None),

    # Modal component for displaying patent details
    dbc.Modal(
        [
            dbc.ModalHeader(
                [
                    dbc.ModalTitle(id='patents-modal-title', style={'color': 'white'}),
                    dbc.Button(
                        "×",
                        id="patents-modal-close-button",
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
                style={"position": "relative", 'backgroundColor': '#2c2c2c'},
                close_button=False,  # Disable the built-in close button
            ),
            dbc.ModalBody(id='patents-modal-body', style={'color': 'white', 'backgroundColor': '#2c2c2c'}),
        ],
        id='patents-modal',
        is_open=False,
    )
], style={'backgroundColor': '#2c2c2c', 'height': '100vh'})  # Dark grey background for the whole page

# --- Callbacks ---
@callback(
    Output('patents-graph', 'figure'),
    Output('patents-graph-data', 'data'),
    Output('patents-prev-viewport', 'data'),
    Output('patents-selected-id', 'data'),  # Output for selected patent ID
    Input('patents-graph', 'relayoutData'),
    Input('patents-year-slider', 'value'),
    Input('patents-searched-patent-coords', 'data'),
    Input('patents-graph', 'clickData'),
    State('patents-graph', 'figure'),
    State('patents-graph-data', 'data'),
    State('patents-prev-viewport', 'data'),
    State('patents-selected-id', 'data')
)
def update_graph(relayoutData, year_range, searched_coords, clickData, existing_fig, existing_data, prev_viewport, selected_patent_id):
    ctx = callback_context

    # Get the triggered property id
    triggered_prop_id = ctx.triggered[0]['prop_id']

    # Initialize variables
    xmin, xmax = X_MIN, X_MAX
    ymin, ymax = Y_MIN, Y_MAX

    # Handle selection and reset appropriate variables
    if triggered_prop_id == 'patents-graph.clickData':
        if clickData and 'points' in clickData and len(clickData['points']) > 0:
            # User clicked on a point
            selected_patent_id = clickData['points'][0]['customdata'][0]
            # Reset searched patent
            searched_coords = None
        else:
            # Clicked on empty space
            selected_patent_id = None
            searched_coords = None  # Reset both
    elif triggered_prop_id == 'patents-searched-patent-coords.data':
        if searched_coords:
            # New search occurred
            selected_patent_id = searched_coords['id']  # Simulate a click on the searched patent
            # Adjust viewport to center on the searched patent
            x = searched_coords['x']
            y = searched_coords['y']
            delta_x = (X_MAX - X_MIN) / 10  # Adjust delta as needed
            delta_y = (Y_MAX - Y_MIN) / 10  # Adjust delta as needed
            xmin = max(x - delta_x, X_MIN)
            xmax = min(x + delta_x, X_MAX)
            ymin = max(y - delta_y, Y_MIN)
            ymax = min(y + delta_y, Y_MAX)
        else:
            # Searched coords is None (search input was cleared)
            selected_patent_id = None
            # Reset viewport to previous or default
            if relayoutData and ('xaxis.range[0]' in relayoutData):
                xmin = relayoutData['xaxis.range[0]']
                xmax = relayoutData['xaxis.range[1]']
                ymin = relayoutData['yaxis.range[0]']
                ymax = relayoutData['yaxis.range[1]']
            elif prev_viewport:
                xmin = prev_viewport.get('xmin', X_MIN)
                xmax = prev_viewport.get('xmax', X_MAX)
                ymin = prev_viewport.get('ymin', Y_MIN)
                ymax = prev_viewport.get('ymax', Y_MAX)
            else:
                xmin, xmax = X_MIN, X_MAX
                ymin, ymax = Y_MIN, Y_MAX
    else:
        # For other triggers (e.g., panning, zooming), get viewport from relayoutData or previous viewport
        if relayoutData and ('xaxis.range[0]' in relayoutData):
            xmin = relayoutData['xaxis.range[0]']
            xmax = relayoutData['xaxis.range[1]']
            ymin = relayoutData['yaxis.range[0]']
            ymax = relayoutData['yaxis.range[1]']
        elif prev_viewport:
            xmin = prev_viewport.get('xmin', X_MIN)
            xmax = prev_viewport.get('xmax', X_MAX)
            ymin = prev_viewport.get('ymin', Y_MIN)
            ymax = prev_viewport.get('ymax', Y_MAX)
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

    # Detect zooming, panning, and searching
    zoomed_out = False
    zoomed_in = False
    panned = False
    year_slider_changed = False
    search_triggered = False

    if prev_viewport is not None and relayoutData:
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

    if triggered_prop_id == 'patents-year-slider.value':
        year_slider_changed = True

    if triggered_prop_id == 'patents-searched-patent-coords.data' and searched_coords:
        search_triggered = True

    # Number of points to limit
    max_total_points = 1000  # Adjust as needed

    # Existing data handling based on interaction
    if zoomed_out or year_slider_changed or search_triggered:
        existing_df = pd.DataFrame(columns=['id', 'x', 'y', 'title', 'abstract', 'codes', 'year'])
        existing_ids = []

        # Fetch the selected patent if it exists
        if selected_patent_id is not None:
            conn = get_db_connection()
            query = """
                SELECT rowid AS id, x, y, title, abstract, codes, year FROM patents
                WHERE rowid = ? LIMIT 1;
            """
            df_selected = pd.read_sql_query(query, conn, params=(selected_patent_id,))
            conn.close()
            existing_df = pd.concat([existing_df, df_selected], ignore_index=True, sort=False)
            existing_ids.append(selected_patent_id)

        # Fetch new random data within the current viewport
        points_to_fetch = max_total_points - len(existing_ids)
        if points_to_fetch > 0:
            new_data = get_data_from_db(
                xmin, xmax, ymin, ymax, year_range, exclude_ids=existing_ids, limit=points_to_fetch
            )
            combined_data = pd.concat([existing_df, new_data], ignore_index=True, sort=False).drop_duplicates(subset='id')
        else:
            combined_data = existing_df.copy()

    elif zoomed_in or panned:
        if existing_data:
            existing_df = pd.DataFrame(existing_data)
            # Keep existing points within the new viewport
            existing_df = existing_df[
                (existing_df['x'] >= xmin) & (existing_df['x'] <= xmax) &
                (existing_df['y'] >= ymin) & (existing_df['y'] <= ymax)
            ]
            existing_ids = existing_df['id'].tolist()
        else:
            existing_df = pd.DataFrame(columns=['id', 'x', 'y', 'title', 'abstract', 'codes', 'year'])
            existing_ids = []

        # Fetch the selected patent if it's not in existing data
        if selected_patent_id is not None and selected_patent_id not in existing_ids:
            conn = get_db_connection()
            query = """
                SELECT rowid AS id, x, y, title, abstract, codes, year FROM patents
                WHERE rowid = ? LIMIT 1;
            """
            df_selected = pd.read_sql_query(query, conn, params=(selected_patent_id,))
            conn.close()
            existing_df = pd.concat([existing_df, df_selected], ignore_index=True, sort=False)
            existing_ids.append(selected_patent_id)

        # Calculate number of points to add
        points_to_add = max_total_points - len(existing_df)
        if points_to_add > 0:
            new_data = get_data_from_db(
                xmin, xmax, ymin, ymax, year_range, exclude_ids=existing_ids, limit=points_to_add
            )
            combined_data = pd.concat([existing_df, new_data], ignore_index=True, sort=False).drop_duplicates(subset='id')
        else:
            combined_data = existing_df.copy()

    else:
        if existing_data:
            combined_data = pd.DataFrame(existing_data)
        else:
            combined_data = get_data_from_db(
                xmin, xmax, ymin, ymax, year_range, limit=max_total_points
            )

        # Fetch the selected patent if it's not in combined_data
        if selected_patent_id is not None and selected_patent_id not in combined_data['id'].values:
            conn = get_db_connection()
            query = """
                SELECT rowid AS id, x, y, title, abstract, codes, year FROM patents
                WHERE rowid = ? LIMIT 1;
            """
            df_selected = pd.read_sql_query(query, conn, params=(selected_patent_id,))
            conn.close()
            combined_data = pd.concat([combined_data, df_selected], ignore_index=True, sort=False).drop_duplicates(subset='id')

    # Ensure 'id' column is of integer type
    if not combined_data.empty:
        combined_data['id'] = combined_data['id'].astype(int)

        # Set 'is_selected' flag for the selected patent
        combined_data['is_selected'] = combined_data['id'] == selected_patent_id
    else:
        # Handle empty combined_data
        combined_data = pd.DataFrame(columns=['id', 'x', 'y', 'title', 'abstract', 'codes', 'year', 'is_selected'])

    # Create the figure
    if not combined_data.empty:
        combined_data['year'] = pd.to_numeric(combined_data['year'], errors='coerce')

        fig = px.scatter(
            combined_data,
            x='x',
            y='y',
            color='year',  # Color points based on the 'year' column
            custom_data=['id'],
            hover_name='title',  # Use 'hover_name' to display only the title
            hover_data={'x': False, 'y': False, 'year': False, 'codes': False},
            color_continuous_scale='Viridis',  # Use the 'Viridis' color scale
            range_color=[2006, 2010],  # Adjust the range as needed
            labels={'x': '', 'y': '', 'color': 'Year'},
        )

        # Force the use of the coloraxis and treat 'year' as continuous
        fig.update_traces(marker=dict(coloraxis='coloraxis'))

        # Adjust marker sizes to highlight the selected patent
        sizes = np.where(
            combined_data['is_selected'],
            20,  # Larger size for selected patent
            6    # Default size for other patents
        )
        fig.update_traces(marker=dict(size=sizes))

        # Update figure layout
        fig.update_layout(
            clickmode='event',
            dragmode='pan',
            uirevision='constant',
            plot_bgcolor='#2c2c2c',
            paper_bgcolor='#2c2c2c',
            font_color='white',
            title_font_color='white',
            coloraxis=dict(
                cmin=2005,
                cmax=2010,
                colorscale='hot',
                colorbar=dict(
                    title="Year",
                    ticks="outside",
                    tickcolor='white',
                    ticklen=5,
                    len=0.7,
                    yanchor="middle",
                    y=0.5,
                    xanchor="left",
                    x=-0.10,
                ),
            ),
            xaxis=dict(
                range=[xmin, xmax],
                constrain='range',
                fixedrange=False,
                autorange=False,
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                showline=False,
            ),
            yaxis=dict(
                range=[ymin, ymax],
                constrain='range',
                fixedrange=False,
                autorange=False,
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                showline=False,
            ),
            margin=dict(
                l=150,
                r=50,
                t=50,
                b=50
            )
        )
    else:
        fig = px.scatter()  # Return an empty figure if no data

    # Prepare data to store in 'graph-data' store
    data_to_store = combined_data.to_dict('records')

    # Store the current viewport bounds
    viewport_data = {'xmin': xmin, 'xmax': xmax, 'ymin': ymin, 'ymax': ymax}

    return fig, data_to_store, viewport_data, selected_patent_id

# Search functionality
@callback(
    Output('patents-search-output', 'children'),
    Output('patents-searched-patent-coords', 'data'),
    Input('patents-search-button', 'n_clicks'),
    State('patents-search-input', 'value'),
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
@callback(
    Output('patents-modal', 'is_open'),
    Output('patents-modal-title', 'children'),
    Output('patents-modal-body', 'children'),
    Input('patents-graph', 'clickData'),
    Input('patents-modal-close-button', 'n_clicks'),
    State('patents-modal', 'is_open'),
    prevent_initial_call=True
)
def display_patent_details(clickData, n_clicks_close, is_open_state):
    ctx = callback_context

    if not ctx.triggered:
        return dash.no_update, dash.no_update, dash.no_update

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger_id == 'patents-graph' and clickData:
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
    elif trigger_id == 'patents-modal-close-button':
        return False, dash.no_update, dash.no_update
    else:
        return dash.no_update, dash.no_update, dash.no_update