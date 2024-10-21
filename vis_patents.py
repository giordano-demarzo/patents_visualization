# vis_patents.py

import pandas as pd
import numpy as np
import sqlite3
from dash import dcc, html, Input, Output, State, callback, callback_context
import plotly.graph_objs as go
import dash
import dash_bootstrap_components as dbc  # For modal components
import plotly.express as px  # For color sequences

# --- Database Connection and Helper Functions ---

# Database connection
def get_db_connection():
    conn = sqlite3.connect('data/patents_topic.db')
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

# Function to get all topics
def get_all_topics():
    conn = get_db_connection()
    query = "SELECT DISTINCT topic_title FROM patents"
    df = pd.read_sql_query(query, conn)
    conn.close()
    topics = df['topic_title'].sort_values().unique()
    return topics

# Get all topics at startup
ALL_TOPICS = get_all_topics()

# Define colors for topics using a qualitative color sequence with at least 20 colors
colors = px.colors.qualitative.Alphabet  # 26 colors available
topic_color_map = dict(zip(ALL_TOPICS, colors[:len(ALL_TOPICS)]))

# Create legend items
legend_items = []
for topic in ALL_TOPICS:
    color = topic_color_map[topic]
    legend_items.append(
        html.Div([
            html.Span('■ ', style={'color': color, 'fontSize': '12px'}),
            html.Span(f"{topic}", style={'fontWeight': 'bold', 'fontSize': '12px'})
        ], style={'marginBottom': '5px'})
    )

# Helper functions
def get_data_from_db(xmin, xmax, ymin, ymax, year_range, exclude_ids=[], limit=20000):
    conn = get_db_connection()
    placeholders = ','.join(['?'] * len(exclude_ids))
    params = [xmin, xmax, ymin, ymax, year_range[0], year_range[1]]
    query = """
        SELECT rowid AS id, x, y, title, topic_title FROM patents
        WHERE x BETWEEN ? AND ?
        AND y BETWEEN ? AND ?
        AND year BETWEEN ? AND ?
    """
    if exclude_ids:
        query += f" AND rowid NOT IN ({placeholders})"
        params.extend(exclude_ids)
    query += """
        LIMIT ?
    """
    params.append(limit)
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df

def search_patent_in_db(title):
    conn = get_db_connection()
    cursor = conn.cursor()
    # Modify the query to be case-insensitive using COLLATE NOCASE
    query = "SELECT rowid AS id, x, y FROM patents WHERE title = ? COLLATE NOCASE LIMIT 1;"
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
        html.H1("Patent Space", style={'textAlign': 'center', 'color': 'white'}),
        # Home button
        html.Div(
            dcc.Link('Home', href='/', className='home-button', style={'color': 'white'}),
            style={'position': 'absolute', 'top': '15px', 'left': '15px', 'color': 'white'}
        ),
    ],
    style={'backgroundColor': '#2c2c2c', 'position': 'relative'}
    ),

    # Wrapper for slider, graph, and legend
    html.Div([
        # Year slider (vertical)
        html.Div([
            html.Label("", style={'color': 'white', 'textAlign': 'center'}),
            dcc.RangeSlider(
                id='patents-year-slider',
                min=2019,
                max=2023,
                value=[2019, 2023],
                step=1,  # Ensure only integer values can be selected
                marks={year: {'label': str(year), 'style': {'color': 'white'}} for year in range(2019, 2024)},
                tooltip={"placement": "left", "always_visible": True},
                vertical=True,
                verticalHeight=400,
                allowCross=False,  # Prevent slider handles from crossing
            )
        ], style={
            'width': '5%',
            'padding': '50px',
            'display': 'flex',
            'align-items': 'center',
            'justify-content': 'center'
        }),

        # Graph
        html.Div(
            dcc.Graph(
                id='patents-graph',
                config={'displayModeBar': True, 'scrollZoom': True},
                style={
                    'height': '80vh',
                    'width': '100%',
                    'backgroundColor': '#2c2c2c',
                }
            ),
            style={'flex': '1', 'position': 'relative'}
        ),

        # Legend
        html.Div(
            legend_items,
            id='patents-legend',
            style={
                'backgroundColor': '#2c2c2c',
                'color': 'white',
                'fontSize': '10px',
                'padding': '10px',
                'border': '1px solid #555',
                'maxHeight': '80vh',
                'overflowY': 'auto',
                'width': '20%',
            },
        ),

    ], style={
        'display': 'flex',
        'backgroundColor': '#2c2c2c',
        'height': '80vh',
    }),

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
    Output('patents-selected-id', 'data'),  # Output for selected patent ID
    Input('patents-year-slider', 'value'),
    Input('patents-searched-patent-coords', 'data'),
    Input('patents-graph', 'clickData'),
    Input('patents-graph', 'relayoutData'),
    State('patents-selected-id', 'data'),
    prevent_initial_call=True
)
def update_graph(year_range, searched_coords, clickData, relayoutData, selected_patent_id):
    ctx = callback_context

    # Get the triggered property id
    triggered_prop_id = ctx.triggered[0]['prop_id']

    # Handle selection and reset appropriate variables
    if triggered_prop_id == 'patents-graph.clickData':
        if clickData and 'points' in clickData and len(clickData['points']) > 0:
            # User clicked on a point
            selected_patent_id = clickData['points'][0]['customdata']
        else:
            # Clicked on empty space
            selected_patent_id = None
    elif triggered_prop_id == 'patents-searched-patent-coords.data':
        if searched_coords:
            # New search occurred
            selected_patent_id = searched_coords['id']  # Simulate a click on the searched patent
        else:
            # Searched coords is None (search input was cleared)
            selected_patent_id = None

    # Determine current viewport bounds
    if relayoutData and ('xaxis.range[0]' in relayoutData):
        xmin = relayoutData['xaxis.range[0]']
        xmax = relayoutData['xaxis.range[1]']
        ymin = relayoutData['yaxis.range[0]']
        ymax = relayoutData['yaxis.range[1]']
    else:
        xmin, xmax = X_MIN, X_MAX
        ymin, ymax = Y_MIN, Y_MAX

    # Fetch data based on viewport and year range
    df = get_data_from_db(xmin, xmax, ymin, ymax, year_range, limit=20000)

    # Set 'is_selected' flag for the selected patent
    if selected_patent_id is not None:
        df['is_selected'] = df['id'] == selected_patent_id
    else:
        df['is_selected'] = False

    # Assign colors based on topic_title
    df['color'] = df['topic_title'].map(topic_color_map)
    # Handle any missing topics by assigning a default color
    df['color'].fillna('#808080', inplace=True)  # Grey color for unknown topics

    # Create the figure
    if not df.empty:
        # Adjust marker sizes for better interactivity
        default_size = 4
        selected_size = 12

        # Limit hover text length to improve performance
        df['hover_text'] = df['title'].str.slice(0, 100)  # Limit to 100 characters

        # Create scatter plot using go.Scattergl
        fig = go.Figure()

        fig.add_trace(go.Scattergl(
            x=df['x'],
            y=df['y'],
            mode='markers',
            marker=dict(
                size=np.where(df['is_selected'], selected_size, default_size),
                color=df['color'],
                line=dict(width=np.where(df['is_selected'], 1, 0), color='white'),
            ),
            customdata=df['id'],
            hovertext=df['hover_text'],
            hoverinfo='text',
            showlegend=False,
        ))

        # Update figure layout
        fig.update_layout(
            clickmode='event+select',
            dragmode='pan',
            uirevision='constant',
            plot_bgcolor='#2c2c2c',
            paper_bgcolor='#2c2c2c',
            font_color='white',
            title_font_color='white',
            xaxis=dict(
                range=[xmin, xmax],
                fixedrange=False,
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                showline=False,
            ),
            yaxis=dict(
                range=[ymin, ymax],
                fixedrange=False,
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                showline=False,
            ),
            margin=dict(
                l=50,
                r=50,
                t=50,
                b=50
            )
        )

        # If a search occurred, adjust the viewport to center on the searched patent
        if triggered_prop_id == 'patents-searched-patent-coords.data' and searched_coords:
            x = searched_coords['x']
            y = searched_coords['y']
            delta_x = (X_MAX - X_MIN) / 15  # Adjust delta as needed
            delta_y = (Y_MAX - Y_MIN) / 15  # Adjust delta as needed
            xmin = max(x - delta_x, X_MIN)
            xmax = min(x + delta_x, X_MAX)
            ymin = max(y - delta_y, Y_MIN)
            ymax = min(y + delta_y, Y_MAX)
            fig.update_layout(
                xaxis=dict(range=[xmin, xmax]),
                yaxis=dict(range=[ymin, ymax]),
            )

    else:
        fig = go.Figure()
        fig.update_layout(
            plot_bgcolor='#2c2c2c',
            paper_bgcolor='#2c2c2c',
            font_color='white',
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                showline=False,
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                showline=False,
            ),
        )

    return fig, selected_patent_id

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
        patent_id = point['customdata']
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
