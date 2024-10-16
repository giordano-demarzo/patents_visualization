# vis_codes.py

import pandas as pd
import numpy as np
import glob
from dash import dcc, html, Input, Output, State, callback, ctx
import plotly.graph_objs as go

# --- Data Loading and Preparation ---

# Load data from CSV files
def load_data():
    data_dict = {}
    csv_files = glob.glob('data/codes_data/code_*.csv')
    print(f"Found CSV files: {csv_files}")  # For debugging
    for file in csv_files:
        year = int(file.split('code_')[1].split('.')[0])
        df = pd.read_csv(file)
        df['year'] = year  # Add a 'year' column
        data_dict[year] = df
    print(f"Data loaded for years: {list(data_dict.keys())}")  # For debugging
    return data_dict

# Load all data at startup
DATA_DICT = load_data()

# Load precomputed trajectories
TRAJECTORY_DATA = pd.read_parquet('data/precomputed_trajectories.parquet')

# Get data boundaries
def get_data_bounds():
    x_min = min(df['x'].min() for df in DATA_DICT.values())
    x_max = max(df['x'].max() for df in DATA_DICT.values())
    y_min = min(df['y'].min() for df in DATA_DICT.values())
    y_max = max(df['y'].max() for df in DATA_DICT.values())
    return x_min, x_max, y_min, y_max

# Get data boundaries at startup
X_MIN, X_MAX, Y_MIN, Y_MAX = get_data_bounds()

# Get available years
AVAILABLE_YEARS = sorted(DATA_DICT.keys())

# Simplify slider marks by selecting a subset of years
def get_slider_marks(years, step):
    marks = {}
    for i, year in enumerate(years):
        if i % step == 0:
            marks[str(year)] = str(year)
        else:
            marks[str(year)] = ''
    return marks

SLIDER_MARKS = get_slider_marks(AVAILABLE_YEARS, step=5)  # Adjust 'step' as needed

# --- Dash App Layout ---

layout = html.Div([
    # Title with dark background
    html.Div(
    [
        html.H1("Technology Space", style={'textAlign': 'center', 'color': 'white'}),
        # Home button
        html.Div(
            dcc.Link('Home', href='/', className='home-button', style={'color': 'white'}),
            style={'position': 'absolute', 'top': '15px', 'left': '15px', 'color': 'white'}
        ),
    ],
    style={'backgroundColor': '#2c2c2c', 'position': 'relative'}
    ),


    # Graph
    dcc.Graph(
        id='codes-graph',
        config={'displayModeBar': True, 'scrollZoom': True},
        style={
            'height': '75vh',
            'width': '70%',
            'display': 'inline-block',
            'padding': '60px',
            'backgroundColor': '#2c2c2c'  # Same dark grey background for the graph area
        },
        clear_on_unhover=True,  # Clear hover data when not hovering
    ),

    # Controls
    html.Div([
        # Year slider
        html.Div([
            html.Label("", style={'color': 'white', 'textAlign': 'center'}),
            dcc.Slider(
                id='codes-year-slider',
                min=AVAILABLE_YEARS[0],
                max=AVAILABLE_YEARS[-1],
                value=AVAILABLE_YEARS[-1],
                marks=SLIDER_MARKS,
                step=None
            ),
        ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'middle'}),

        # Trajectory buttons
        html.Div([
            html.Button('Show All Trajectories', id='codes-show-all-trajectories-button', n_clicks=0,
                        style={'backgroundColor': '#3a3a3a', 'color': 'white', 'border': '1px solid #555'}),
            html.Button('Remove All Trajectories', id='codes-remove-all-trajectories-button', n_clicks=0,
                        style={'backgroundColor': '#3a3a3a', 'color': 'white', 'border': '1px solid #555', 'marginLeft': '10px'}),
        ], style={'width': '40%', 'display': 'inline-block', 'textAlign': 'center', 'verticalAlign': 'middle'}),

        # Search input
        html.Div([
            dcc.Input(
                id='codes-search-input',
                type='text',
                placeholder='Search Code',
                style={'width': '200px', 'color': 'white', 'backgroundColor': '#3a3a3a', 'border': '1px solid #555'}
            ),
            html.Button('Search', id='codes-search-button', n_clicks=0,
                        style={'backgroundColor': '#3a3a3a', 'color': 'white', 'border': '1px solid #555'}),
        ], style={'width': '30%', 'display': 'inline-block', 'textAlign': 'right', 'verticalAlign': 'middle'}),
    ], style={'backgroundColor': '#2c2c2c', 'padding': '10px'}),

    # Hidden Divs to Store Data
    dcc.Store(id='codes-filtered-data'),
    dcc.Store(id='codes-search-data'),
    dcc.Store(id='codes-selected-codes', data=[]),  # Store for selected codes
    dcc.Store(id='codes-click-counter', data=0),  # Store to count clicks
], style={'height': '100vh', 'backgroundColor': '#2c2c2c', 'margin': '0', 'padding': '0'})  # Set the overall background color and remove margins

# --- Callbacks ---

# Filter data based on selected year
@callback(
    Output('codes-filtered-data', 'data'),
    Input('codes-year-slider', 'value')
)
def filter_data(year):
    year = int(year)
    df = DATA_DICT.get(year, pd.DataFrame())
    return df.to_dict('records')

# Replace clientside callback with serverside callback
@callback(
    Output('codes-click-counter', 'data'),
    Input('codes-graph', 'clickData'),
    State('codes-click-counter', 'data')
)
def update_click_counter(clickData, clickCounter):
    if clickData is None:
        return clickCounter
    return clickCounter + 1

# Update selected codes based on user interactions
@callback(
    Output('codes-selected-codes', 'data'),
    Input('codes-click-counter', 'data'),
    Input('codes-show-all-trajectories-button', 'n_clicks'),
    Input('codes-remove-all-trajectories-button', 'n_clicks'),
    State('codes-selected-codes', 'data'),
    State('codes-graph', 'clickData'),
    State('codes-filtered-data', 'data'),
)
def update_selected_codes(click_counter, show_all_n_clicks, remove_all_n_clicks, selected_codes, clickData, filtered_data):
    triggered_id = ctx.triggered_id

    if triggered_id == 'codes-show-all-trajectories-button':
        # Show trajectories for all codes in current view
        df = pd.DataFrame(filtered_data)
        selected_codes = df['code'].unique().tolist()
        return selected_codes
    elif triggered_id == 'codes-remove-all-trajectories-button':
        # Remove all trajectories
        return []
    elif triggered_id == 'codes-click-counter':
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
@callback(
    Output('codes-search-data', 'data'),
    Input('codes-search-button', 'n_clicks'),
    State('codes-search-input', 'value'),
    State('codes-filtered-data', 'data')
)
def search_code(n_clicks, search_value, data):
    if n_clicks > 0 and search_value:
        df = pd.DataFrame(data)
        matched_code = df[df['code'] == search_value]
        if not matched_code.empty:
            return matched_code.iloc[0].to_dict()
    return None

# Update the graph based on filtered data and selected codes
@callback(
    Output('codes-graph', 'figure'),
    Input('codes-filtered-data', 'data'),
    Input('codes-search-data', 'data'),
    Input('codes-selected-codes', 'data'),
    Input('codes-graph', 'relayoutData'),
    Input('codes-year-slider', 'value'),
)
def update_graph(data, search_data, selected_codes, relayoutData, selected_year):
    print(f"update_graph called with {len(data)} data points.")
    df = pd.DataFrame(data)

    # Map first letters to colors
    if not df.empty:
        df['first_letter'] = df['code'].str[0]

        # Define colors for letters A to H
        letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        colors = ['#e6194b',  # Red
                  '#3cb44b',  # Green
                  '#ffe119',  # Yellow
                  '#4363d8',  # Blue
                  '#f58231',  # Orange
                  '#911eb4',  # Purple
                  '#42d4f4',  # Cyan
                  '#f032e6']  # Magenta

        color_map = dict(zip(letters, colors))
        # Assign colors based on the first letter
        df['color'] = df['first_letter'].map(color_map)
        # For any letters outside A-H, assign a default color
        df['color'].fillna('#ffffff', inplace=True)  # White for unknown letters

    # Create the base figure using scattergl for better performance
    fig = go.Figure()
    if not df.empty:
        fig.add_trace(go.Scattergl(
            x=df['x'],
            y=df['y'],
            mode='markers',
            name='',  # Set empty name to avoid "trace 0"
            marker=dict(
                color=df['color'],
                size=6,
            ),
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
            line=dict(color='white'),  # Set trajectories to white
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
            marker=dict(size=15, color='white', symbol='circle'),  # White color for visibility
            hovertext=code + ': ' + name,
            hoverinfo='text',
            showlegend=False,  # Do not show in legend
        ))
        # Update the figure to focus on the searched code
        fig.update_layout(
            xaxis=dict(range=[x - 50, x + 50]),
            yaxis=dict(range=[y - 50, y + 50]),
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
                xaxis=dict(range=[X_MIN, X_MAX]),
                yaxis=dict(range=[Y_MIN, Y_MAX]),
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