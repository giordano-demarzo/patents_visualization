# vis_codes.py

import pandas as pd
import numpy as np
import glob
import pickle
from dash import dcc, html, Input, Output, State, callback, ctx
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import plotly.express as px  # For color sequences

# --- Data Loading and Preparation ---

# Load data from CSV files
def load_data():
    data_dict = {}
    csv_files = glob.glob('data/codes_data/code_*.csv')
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

# Load the similar codes data
with open('data/top_10_codes_2019_2023_llama8b_abstracts.pkl', 'rb') as f:
    SIMILAR_CODES_DICT = pickle.load(f)

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

# --- Define Colors and Categories ---

# Map first letters to colors and categories
letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
colors = px.colors.qualitative.Alphabet[-12:-4]  # Use a predefined color sequence

categories = {
    'A': 'HUMAN NECESSITIES',
    'B': 'PERFORMING OPERATIONS; TRANSPORTING',
    'C': 'CHEMISTRY; METALLURGY',
    'D': 'TEXTILES; PAPER',
    'E': 'FIXED CONSTRUCTIONS',
    'F': 'MECHANICAL ENGINEERING; LIGHTING; HEATING; WEAPONS; BLASTING',
    'G': 'PHYSICS',
    'H': 'ELECTRICITY'
}

color_map = dict(zip(letters, colors))

# Create legend items
legend_items = []
for letter in letters:
    category_name = categories[letter]
    color = color_map[letter]
    legend_items.append(
        html.Div([
            html.Span('■ ', style={'color': color, 'fontSize': '12px'}),
            html.Span(f"{letter}: {category_name}", style={'fontWeight': 'bold', 'fontSize': '12px'})
        ], style={'marginBottom': '1px'})
    )

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
        # Help button
        html.Div(
            html.Button("Help", id='codes-help-button', n_clicks=0, style={'backgroundColor': '#3a3a3a', 'color': 'white', 'border': 'none'}),
            style={'position': 'absolute', 'top': '15px', 'right': '15px', 'color': 'white'}
        ),
    ],
    style={'backgroundColor': '#2c2c2c', 'position': 'relative'}
    ),

    # Main content area with graph, info box, and legend
    html.Div([
        # Graph
        html.Div(
            dcc.Graph(
                id='codes-graph',
                config={'displayModeBar': False, 'scrollZoom': True},
                style={
                    'height': '65vh',
                    'width': '60%',
                    'backgroundColor': '#2c2c2c'  # Same dark grey background for the graph area
                },
                clear_on_unhover=True,  # Clear hover data when not hovering
            ),
            style={'flex': '1', 'position': 'relative'}  # Allows the graph to take up remaining space
        ),
        # Info Box
        html.Div(
            id='codes-info-box',
            style={
                'position': 'absolute',
                'top': '20px',
                'right': '100px',
                'backgroundColor': '#2c2c2c',
                'color': 'white',
                'padding': '10px',
                'border': '1px solid #555',
                'width': '300px',   # Set fixed width
                'height': '250px',  # Set fixed height
                'overflowY': 'auto',
            }
        ),
        # Legend
        html.Div(
            legend_items,
            id='codes-legend',
            style={
                'position': 'absolute',
                'bottom': '20px',
                'right': '20px',
                'backgroundColor': '#2c2c2c',
                'color': 'white',
                'fontSize': '10px',
            },
        ),
    ], style={
        'display': 'flex',
        'backgroundColor': '#2c2c2c',
        'position': 'relative',  # Necessary for absolute positioning of the legend and info box
        'height': '75vh',
        'width': '100%',
    }),

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
                step=1
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
    dcc.Store(id='clicked-code', data=None),  # Store for the clicked code

    # Modal component for displaying help information
    dbc.Modal(
        [
            dbc.ModalHeader(
                dbc.ModalTitle("Visualization Help", style={'color': 'white'}),
                style={'backgroundColor': '#2c2c2c'},
                close_button=True  # Use built-in close button
            ),
            dbc.ModalBody(
                [
                    html.H4("How to use this visualization", style={'color': 'white'}),
                    html.P(
                        "This visualization allows you to explore technological codes in a 2D space. Each point represents a code, and the position reflects similarities between codes based on their usage in patents.",
                        style={'color': 'white'}
                    ),
                    html.P(
                        "You can zoom and pan around the graph using your mouse or trackpad.",
                        style={'color': 'white'}
                    ),
                    html.P(
                        "Click on a point to view details about the code and see its most similar codes.",
                        style={'color': 'white'}
                    ),
                    html.P(
                        "Use the year slider to filter codes by the selected year.",
                        style={'color': 'white'}
                    ),
                    html.P(
                        "Use the search box to locate a specific technological code. The view will zoom into that code, and it will be highlighted.",
                        style={'color': 'white'}
                    ),
                    html.P(
                        "Click on 'Show All Trajectories' to display the trajectories of all codes over time, or 'Remove All Trajectories' to hide them.",
                        style={'color': 'white'}
                    ),
                    # Add more explanations as needed
                ],
                style={'backgroundColor': '#2c2c2c'}
            ),
        ],
        id='codes-help-modal',
        is_open=False,
    ),

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

# Callback to store the clicked code
@callback(
    Output('clicked-code', 'data'),
    Input('codes-graph', 'clickData'),
)
def update_clicked_code(clickData):
    if clickData and 'points' in clickData and len(clickData['points']) > 0:
        clicked_code = clickData['points'][0]['customdata']
        return clicked_code
    else:
        return None

# Update the graph based on filtered data and selected codes
@callback(
    Output('codes-graph', 'figure'),
    Input('codes-filtered-data', 'data'),
    Input('codes-search-data', 'data'),
    Input('codes-selected-codes', 'data'),
    Input('codes-graph', 'relayoutData'),
    Input('codes-year-slider', 'value'),
    Input('clicked-code', 'data'),
)
def update_graph(data, search_data, selected_codes, relayoutData, selected_year, clicked_code):
    import time  # Import time module for profiling

    start_time = time.time()

    df = pd.DataFrame(data)

    # Set 'code' as index for faster lookup
    if not df.empty:
        df.set_index('code', inplace=True)

        # Map first letters to colors and categories
        df['first_letter'] = df.index.str[0]

        # Assign colors based on the first letter
        df['color'] = df['first_letter'].map(color_map)
        # For any letters outside A-H, assign a default color
        df['color'].fillna('#ffffff', inplace=True)  # White for unknown letters

        # Assign categories
        df['category'] = df['first_letter'].map(categories)
        df['category'].fillna('OTHER', inplace=True)

        # Limit hover text length to improve performance and include code
        df['hover_text'] = df.index + ': ' + df['name'].str.slice(0, 100)  # Limit to 100 characters
    else:
        df = pd.DataFrame(columns=['x', 'y', 'color', 'hover_text'])  # Ensure df has necessary columns

    setup_time = time.time()

    # --- Optimization: Limit Data Sent to Client ---
    max_points = 10000  # Adjust as needed
    if len(df) > max_points:
        df_sampled = df.sample(n=max_points, random_state=42)
    else:
        df_sampled = df

    # Ensure that clicked code and similar codes are included
    if clicked_code:
        # Get the similar codes
        similar_codes_with_scores = SIMILAR_CODES_DICT.get(clicked_code, [])
        similar_codes = [code for code, _ in similar_codes_with_scores]
        codes_to_include = [clicked_code] + similar_codes

        # Get data for these codes
        codes_df = df.loc[df.index.intersection(codes_to_include)]

        # Combine sampled data and codes_df
        df_plot = pd.concat([df_sampled, codes_df]).drop_duplicates()
    else:
        df_plot = df_sampled

    data_preparation_time = time.time()

    # Create the base figure using Scattergl
    fig = go.Figure()
    if not df_plot.empty:
        fig.add_trace(go.Scattergl(
            x=df_plot['x'],
            y=df_plot['y'],
            mode='markers',
            marker=dict(
                color=df_plot['color'],
                size=4,  # Adjusted marker size
            ),
            customdata=df_plot.index,
            hovertext=df_plot['hover_text'],
            hoverinfo='text',
            showlegend=False,  # Do not show legend inside the plot
        ))

    base_plot_time = time.time()

    # Highlight the clicked code and similar codes
    if clicked_code:
        # Get data for clicked code and similar codes
        if clicked_code in df.index:
            clicked_df = df.loc[[clicked_code]]
        else:
            clicked_df = pd.DataFrame(columns=df.columns)

        similar_codes_in_data = [code for code in similar_codes if code in df.index]
        if similar_codes_in_data:
            similar_df = df.loc[similar_codes_in_data]
        else:
            similar_df = pd.DataFrame(columns=df.columns)

        # Limit hover text length
        if not similar_df.empty:
            similar_df['hover_text'] = similar_df.index + ': ' + similar_df['name'].str.slice(0, 100)

            # Add trace for similar codes
            fig.add_trace(go.Scattergl(
                x=similar_df['x'],
                y=similar_df['y'],
                mode='markers',
                marker=dict(
                    color='#ffe119',  # Hex code for yellow
                    size=8,
                    symbol='circle',
                    line=dict(color='white', width=1)
                ),
                customdata=similar_df.index,
                hovertext=similar_df['hover_text'],
                hoverinfo='text',
                hoverlabel=dict(bgcolor='#ffe119'),  # Set hover box background color
                showlegend=False,
            ))

        if not clicked_df.empty:
            clicked_df['hover_text'] = clicked_df.index + ': ' + clicked_df['name'].str.slice(0, 100)

            # Add trace for clicked code
            fig.add_trace(go.Scattergl(
                x=clicked_df['x'],
                y=clicked_df['y'],
                mode='markers',
                marker=dict(
                    color='#FF0000',  # Hex code for red
                    size=10,
                    symbol='circle',
                    line=dict(color='white', width=2)
                ),
                customdata=clicked_df.index,
                hovertext=clicked_df['hover_text'],
                hoverinfo='text',
                hoverlabel=dict(bgcolor='#FF0000'),  # Set hover box background color
                showlegend=False,
            ))

    clicked_code_time = time.time()

    # --- Optimization: Avoid Changing Axes Ranges ---
    # Preserve zoom and pan if possible
    if relayoutData and 'xaxis.range[0]' in relayoutData:
        xmin = relayoutData['xaxis.range[0]']
        xmax = relayoutData['xaxis.range[1]']
        ymin = relayoutData['yaxis.range[0]']
        ymax = relayoutData['yaxis.range[1]']
        fig.update_layout(
            xaxis=dict(range=[xmin, xmax]),
            yaxis=dict(range=[ymin, ymax], scaleanchor='x', scaleratio=1),
        )
    else:
        # Set default axis ranges with equal aspect ratio
        fig.update_layout(
            xaxis=dict(range=[X_MIN, X_MAX]),
            yaxis=dict(range=[Y_MIN, Y_MAX], scaleanchor='x', scaleratio=1),
        )

    axes_update_time = time.time()

    # Add trajectories for selected codes
    if selected_codes:
        # Limit the number of trajectories rendered
        max_trajectories = 5  # Set a limit to avoid overloading
        selected_codes_limited = selected_codes[:max_trajectories]

        # Filter trajectories for selected codes up to the selected year
        traj_df = TRAJECTORY_DATA[
            (TRAJECTORY_DATA['code'].isin(selected_codes_limited)) &
            (TRAJECTORY_DATA['year'] <= selected_year)
        ]

        # Ensure trajectories are sorted by 'code' and 'year'
        traj_df_sorted = traj_df.sort_values(['code', 'year'])

        # Prepare data for plotting trajectories with breaks
        x_traj = []
        y_traj = []
        for code in selected_codes_limited:
            code_df = traj_df_sorted[traj_df_sorted['code'] == code]
            x_traj.extend(code_df['x_smooth'].tolist() + [np.nan])  # Add NaN to create a break
            y_traj.extend(code_df['y_smooth'].tolist() + [np.nan])

        fig.add_trace(go.Scattergl(
            x=x_traj,
            y=y_traj,
            mode='lines',
            name='',  # Set empty name to avoid extra legend entry
            line=dict(color='white'),  # Set trajectories to white
            hoverinfo='skip',  # Disable hoverinfo for the trajectories
            showlegend=False,  # Do not show in legend
        ))

    trajectories_time = time.time()

    # Update layout
    fig.update_layout(
        clickmode='event',  # Keep 'event' to capture clickData
        dragmode='pan',
        uirevision='constant',  # Use 'constant' to preserve state
        plot_bgcolor='#2c2c2c',
        paper_bgcolor='#2c2c2c',
        font_color='white',
        showlegend=False,  # Do not show legend inside the plot
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
        scaleanchor='x',
        scaleratio=1,
    )

    final_time = time.time()

    # Output detailed timing information for profiling
    total_time = final_time - start_time
    setup_duration = setup_time - start_time
    data_prep_duration = data_preparation_time - setup_time
    base_plot_duration = base_plot_time - data_preparation_time
    clicked_code_duration = clicked_code_time - base_plot_time
    axes_update_duration = axes_update_time - clicked_code_time
    trajectories_duration = trajectories_time - axes_update_time
    finalization_duration = final_time - trajectories_time

    print(f"Total update_graph time: {total_time:.3f}s")
    print(f"Setup duration: {setup_duration:.3f}s")
    print(f"Data preparation duration: {data_prep_duration:.3f}s")
    print(f"Base plot duration: {base_plot_duration:.3f}s")
    print(f"Clicked code processing duration: {clicked_code_duration:.3f}s")
    print(f"Axes update duration: {axes_update_duration:.3f}s")
    print(f"Trajectories duration: {trajectories_duration:.3f}s")
    print(f"Finalization duration: {finalization_duration:.3f}s")

    return fig

# Callback to update the info box when a code is clicked
@callback(
    Output('codes-info-box', 'children'),
    Input('codes-graph', 'clickData'),
    State('codes-filtered-data', 'data'),
)
def update_info_box(clickData, filtered_data):
    if clickData and 'points' in clickData and len(clickData['points']) > 0:
        clicked_code = clickData['points'][0]['customdata']
        df = pd.DataFrame(filtered_data)
        code_info = df[df['code'] == clicked_code]
        if not code_info.empty:
            code_info = code_info.iloc[0]
            full_name = code_info.get('name_full', 'No full name available')
            similar_codes = SIMILAR_CODES_DICT.get(clicked_code, [])

            # Create the content for the info box
            content = [
                html.H4(f"{clicked_code}: {full_name}", style={'marginBottom': '10px', 'fontSize': '15px'}),
                html.H5('Most Similar Codes:', style={'marginBottom': '2px', 'fontSize': '12px'}),
                html.Ul([
                    html.Li(f"{code}: {similarity_score:.4f}", style={'marginBottom': '2px', 'fontSize': '12px'})
                    for code, similarity_score in similar_codes
                ])
            ]
            return content
        else:
            return ''
    else:
        # If no code is clicked, return an empty Div
        return 'Click on a technological code to visualize the most likely novel technology combinations involving it'

# Callback to toggle the help modal
@callback(
    Output('codes-help-modal', 'is_open'),
    Input('codes-help-button', 'n_clicks'),
    State('codes-help-modal', 'is_open'),
    prevent_initial_call=True
)
def toggle_help_modal(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open