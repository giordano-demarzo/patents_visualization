#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 19:38:13 2024

@author: giordano
"""

# visualization_codes.py

import pandas as pd
import numpy as np
import glob
from dash import dcc, html, Input, Output, State, ctx
import plotly.graph_objs as go
import dash_bootstrap_components as dbc
from app import app

# --- Data Loading and Preparation ---

# Load data from CSV files
def load_data():
    data_dict = {}
    csv_files = glob.glob('data/generated_files/code_*.csv')
    for file in csv_files:
        year = int(file.split('code_')[1].split('.')[0])
        df = pd.read_csv(file)
        df['year'] = year  # Add a 'year' column
        data_dict[year] = df
    return data_dict

# Load all data at startup
DATA_DICT = load_data()

# Load precomputed trajectories
TRAJECTORY_DATA = pd.read_parquet('data/precomputed_trajectories.parquet')

# Get data boundaries
def get_data_bounds_codes():
    x_min = min(df['x'].min() for df in DATA_DICT.values())
    x_max = max(df['x'].max() for df in DATA_DICT.values())
    y_min = min(df['y'].min() for df in DATA_DICT.values())
    y_max = max(df['y'].max() for df in DATA_DICT.values())
    return x_min, x_max, y_min, y_max

# Get data boundaries at startup
X_MIN_CODES, X_MAX_CODES, Y_MIN_CODES, Y_MAX_CODES = get_data_bounds_codes()

# Get available years
AVAILABLE_YEARS_CODES = sorted(DATA_DICT.keys())

# Simplify slider marks by selecting a subset of years
def get_slider_marks_codes(years, step):
    marks = {}
    for i, year in enumerate(years):
        if i % step == 0:
            marks[str(year)] = str(year)
        else:
            marks[str(year)] = ''
    return marks

SLIDER_MARKS_CODES = get_slider_marks_codes(AVAILABLE_YEARS_CODES, step=5)  # Adjust 'step' as needed

# --- Layout ---

codes_layout = html.Div([
    # Navigation Link
    html.Div([
        dbc.Button("Home", color="link", href="/", className='mr-2', style={'color': 'white', 'fontSize': '18px'}),
    ], style={'position': 'fixed', 'top': '10px', 'left': '10px', 'zIndex': '1000'}),
    # Title with dark background
    html.Div(
        html.H1("Technological Codes Space", style={'textAlign': 'center', 'color': 'white', 'margin': '0', 'padding': '10px'}),
        style={'backgroundColor': '#2c2c2c'}
    ),
    # Graph Container
    html.Div(
        dcc.Graph(
            id='code-graph',
            config={'displayModeBar': True, 'scrollZoom': True},
            style={'height': '80vh', 'width': '80%', 'backgroundColor': '#2c2c2c', 'margin': '0 auto'},
            clear_on_unhover=True,  # Clear hover data when not hovering
        ),
        style={'textAlign': 'center'}  # Center the graph container
    ),
    # Controls
    html.Div([
        # Year slider
        html.Div([
            html.Label("Select Year", style={'color': 'white'}),
            dcc.Slider(
                id='code-year-slider',
                min=AVAILABLE_YEARS_CODES[0],
                max=AVAILABLE_YEARS_CODES[-1],
                value=AVAILABLE_YEARS_CODES[-1],
                marks=SLIDER_MARKS_CODES,
                step=None
            ),
        ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'middle'}),
        # Trajectory buttons
        html.Div([
            html.Button('Show All Trajectories', id='show-all-trajectories-button', n_clicks=0,
                        style={'backgroundColor': '#3a3a3a', 'color': 'white', 'border': '1px solid #555'}),
            html.Button('Remove All Trajectories', id='remove-all-trajectories-button', n_clicks=0,
                        style={'backgroundColor': '#3a3a3a', 'color': 'white', 'border': '1px solid #555', 'marginLeft': '10px'}),
        ], style={'width': '40%', 'display': 'inline-block', 'textAlign': 'center', 'verticalAlign': 'middle'}),
        # Search input
        html.Div([
            dcc.Input(
                id='code-search-input',
                type='text',
                placeholder='Search Code',
                style={'width': '200px', 'color': 'white', 'backgroundColor': '#3a3a3a', 'border': '1px solid #555'}
            ),
            html.Button('Search', id='code-search-button', n_clicks=0,
                        style={'backgroundColor': '#3a3a3a', 'color': 'white', 'border': '1px solid #555'}),
        ], style={'width': '30%', 'display': 'inline-block', 'textAlign': 'right', 'verticalAlign': 'middle'}),
    ], style={'backgroundColor': '#2c2c2c', 'padding': '10px'}),
    # Hidden Divs to Store Data
    dcc.Store(id='filtered-data'),
    dcc.Store(id='search-data'),
    dcc.Store(id='selected-codes', data=[]),  # Store for selected codes
    dcc.Store(id='click-counter', data=0),  # Store to count clicks
], style={'backgroundColor': '#2c2c2c', 'padding': '0', 'margin': '0'})

# --- Callbacks ---

# Filter data based on selected year
@app.callback(
    Output('filtered-data', 'data'),
    Input('code-year-slider', 'value')
)
def filter_data(year):
    df = DATA_DICT.get(year, pd.DataFrame())
    return df.to_dict('records')

# Clientside callback to increment click counter
app.clientside_callback(
    """
    function(clickData, clickCounter) {
        if (clickData === undefined || clickData === null) {
            return clickCounter;
        }
        return clickCounter + 1;
    }
    """,
    Output('click-counter', 'data'),
    Input('code-graph', 'clickData'),
    State('click-counter', 'data')
)

# Update selected codes based on user interactions
@app.callback(
    Output('selected-codes', 'data'),
    Input('click-counter', 'data'),
    Input('show-all-trajectories-button', 'n_clicks'),
    Input('remove-all-trajectories-button', 'n_clicks'),
    State('selected-codes', 'data'),
    State('code-graph', 'clickData'),
    State('filtered-data', 'data'),
)
def update_selected_codes(click_counter, show_all_n_clicks, remove_all_n_clicks, selected_codes, clickData, filtered_data):
    triggered_id = ctx.triggered_id

    if triggered_id == 'show-all-trajectories-button':
        # Show trajectories for all codes in current view
        df = pd.DataFrame(filtered_data)
        selected_codes = df['code'].unique().tolist()
        return selected_codes
    elif triggered_id == 'remove-all-trajectories-button':
        # Remove all trajectories
        return []
    elif triggered_id == 'click-counter':
        # Handle point click
        if clickData is None:
            return selected_codes

        # Get the clicked code
        clicked_code = None
        if 'points' in clickData and len(clickData['points']) > 0:
            clicked_code = clickData['points'][0]['customdata']

        if clicked_code is None:
            return selected_codes

        # Make a copy of the list to avoid mutating state directly
        selected_codes = selected_codes.copy()

        # Toggle the clicked code in the selected codes list
        if clicked_code in selected_codes:
            selected_codes.remove(clicked_code)
        else:
            selected_codes.append(clicked_code)

        return selected_codes
    else:
        # No change
        return selected_codes

# Handle search functionality
@app.callback(
    Output('search-data', 'data'),
    Input('code-search-button', 'n_clicks'),
    State('code-search-input', 'value'),
    State('filtered-data', 'data')
)
def search_code(n_clicks, search_value, data):
    if n_clicks > 0 and search_value:
        df = pd.DataFrame(data)
        matched_code = df[df['code'] == search_value]
        if not matched_code.empty:
            return matched_code.iloc[0].to_dict()
    return None

# Update the graph based on filtered data and selected codes
@app.callback(
    Output('code-graph', 'figure'),
    Input('filtered-data', 'data'),
    Input('search-data', 'data'),
    Input('selected-codes', 'data'),
    Input('code-graph', 'relayoutData'),
    Input('code-year-slider', 'value'),
)
def update_graph(data, search_data, selected_codes, relayoutData, selected_year):
    df = pd.DataFrame(data)

    # Create the base figure using scattergl for better performance
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=df['x'],
        y=df['y'],
        mode='markers',
        name='',  # Set empty name to avoid "trace 0"
        marker=dict(color='yellow', size=6),
        customdata=df['code'],
        hovertemplate='%{customdata}: %{text}',
        text=df['name'],
        hoverinfo='text',
        showlegend=False,  # Do not show in legend
    ))

    # Add trajectories for selected codes
    if selected_codes:
        # Filter trajectories for selected codes up to the selected year
        traj_df = TRAJECTORY_DATA[
            (TRAJECTORY_DATA['code'].isin(selected_codes)) &
            (TRAJECTORY_DATA['year'] <= selected_year)
        ]

        # Ensure trajectories are sorted by 'code' and 'year'
        traj_df_sorted = traj_df.sort_values(['code', 'year'])

        # Prepare data for plotting trajectories with breaks
        x_traj = []
        y_traj = []
        for code in selected_codes:
            code_df = traj_df_sorted[traj_df_sorted['code'] == code]
            x_traj.extend(code_df['x_smooth'].tolist() + [np.nan])  # Add NaN to create a break
            y_traj.extend(code_df['y_smooth'].tolist() + [np.nan])

        fig.add_trace(go.Scattergl(
            x=x_traj,
            y=y_traj,
            mode='lines',
            name='',  # Set empty name to avoid "trace 1"
            line=dict(color='blue'),
            hoverinfo='skip',  # Disable hoverinfo for the trajectories
            showlegend=False,  # Do not show in legend
        ))

    # Handle search functionality
    if search_data:
        x = search_data['x']
        y = search_data['y']
        code = search_data['code']
        name = search_data['name']
        # Highlight the searched code
        fig.add_trace(go.Scattergl(
            x=[x],
            y=[y],
            mode='markers',
            name='',  # Set empty name to avoid "trace 2"
            marker=dict(size=15, color='red', symbol='star'),
            hovertext=code + ': ' + name,
            hoverinfo='text',
            showlegend=False,  # Do not show in legend
        ))
        # Update the figure to focus on the searched code
        fig.update_layout(
            xaxis=dict(range=[x - 0.1, x + 0.1]),
            yaxis=dict(range=[y - 0.1, y + 0.1]),
        )
    else:
        # Preserve zoom and pan if no search
        if relayoutData and 'xaxis.range[0]' in relayoutData:
            xmin = relayoutData['xaxis.range[0]']
            xmax = relayoutData['xaxis.range[1]']
            ymin = relayoutData['yaxis.range[0]']
            ymax = relayoutData['yaxis.range[1]']
            fig.update_layout(
                xaxis=dict(range=[xmin, xmax]),
                yaxis=dict(range=[ymin, ymax]),
            )
        else:
            # Set default axis ranges
            fig.update_layout(
                xaxis=dict(range=[X_MIN_CODES, X_MAX_CODES]),
                yaxis=dict(range=[Y_MIN_CODES, Y_MAX_CODES]),
            )

    # Update layout
    fig.update_layout(
        clickmode='event',  # Keep 'event' to capture clickData
        dragmode='pan',
        uirevision='constant',
        plot_bgcolor='#2c2c2c',
        paper_bgcolor='#2c2c2c',
        font_color='white',
        showlegend=False,  # Do not show legend
        margin=dict(l=0, r=0, t=0, b=0),  # Remove margins
    )
    # Update axes properties
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        showline=False,
    )
    fig.update_yaxes(
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        showline=False,
    )

    return fig