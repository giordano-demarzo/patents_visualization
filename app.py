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

    # Store component to hold the searched patent coordinates
    dcc.Store(id='searched-patent-coords', data=None),
    
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

# Helper functions
def get_data_from_db(xmin, xmax, ymin, ymax, year_range):
    conn = get_db_connection()
    query = """
        SELECT x, y, title, abstract, codes, year FROM patents
        WHERE x BETWEEN ? AND ?
        AND y BETWEEN ? AND ?
        AND year BETWEEN ? AND ?
        ORDER BY RANDOM()
        LIMIT 200;
    """
    df = pd.read_sql_query(query, conn, params=(xmin, xmax, ymin, ymax, year_range[0], year_range[1]))
    conn.close()
    return df

def search_patent_in_db(title):
    conn = get_db_connection()
    cursor = conn.cursor()
    query = "SELECT x, y FROM patents WHERE title = ? LIMIT 1;"
    cursor.execute(query, (title,))
    result = cursor.fetchone()
    conn.close()

    if result:
        x, y = result
        return {'x': x, 'y': y}
    else:
        return None

def get_patent_details(x, y):
    conn = get_db_connection()
    query = """
        SELECT title, abstract FROM patents
        WHERE x = ? AND y = ?
        LIMIT 1;
    """
    cursor = conn.cursor()
    cursor.execute(query, (x, y))
    result = cursor.fetchone()
    conn.close()

    if result:
        title, abstract = result
        return {'title': title, 'abstract': abstract}
    else:
        return None

# Update graph based on viewport, year slider, and searched patent
@app.callback(
    Output('patent-graph', 'figure'),
    Input('patent-graph', 'relayoutData'),
    Input('year-slider', 'value'),
    Input('searched-patent-coords', 'data'),
    State('patent-graph', 'figure')
)
def update_graph(relayoutData, year_range, searched_coords, existing_fig):
    ctx = dash.callback_context

    # Determine what triggered the callback
    triggered_input = ctx.triggered[0]['prop_id'].split('.')[0]

    # Determine viewport bounds based on user interaction or existing figure
    if relayoutData and ('xaxis.range[0]' in relayoutData):
        xmin = relayoutData['xaxis.range[0]']
        xmax = relayoutData['xaxis.range[1]']
        ymin = relayoutData['yaxis.range[0]']
        ymax = relayoutData['yaxis.range[1]']
    elif existing_fig and 'layout' in existing_fig:
        xmin = existing_fig['layout']['xaxis']['range'][0]
        xmax = existing_fig['layout']['xaxis']['range'][1]
        ymin = existing_fig['layout']['yaxis']['range'][0]
        ymax = existing_fig['layout']['yaxis']['range'][1]
    else:
        xmin, xmax = -100, 100
        ymin, ymax = -100, 100

    # If a new search has occurred, center the viewport on the searched patent
    if triggered_input == 'searched-patent-coords' and searched_coords:
        x = searched_coords['x']
        y = searched_coords['y']
        delta = 5  # Adjust delta for desired zoom level
        xmin = x - delta
        xmax = x + delta
        ymin = y - delta
        ymax = y + delta

    # Fetch data within the viewport and year range
    data = get_data_from_db(xmin, xmax, ymin, ymax, year_range)

    # Initialize 'is_searched' column
    data['is_searched'] = False

    if searched_coords:
        searched_x = searched_coords['x']
        searched_y = searched_coords['y']
        # Check if searched patent is within the current data
        is_in_data = ((data['x'] == searched_x) & (data['y'] == searched_y)).any()
        if not is_in_data:
            # Fetch the searched patent and add it to the data
            conn = get_db_connection()
            query = """
                SELECT x, y, title, abstract, codes, year FROM patents
                WHERE x = ? AND y = ? LIMIT 1;
            """
            df_searched = pd.read_sql_query(query, conn, params=(searched_x, searched_y))
            conn.close()
            df_searched['is_searched'] = True
            data = pd.concat([data, df_searched], ignore_index=True)
        else:
            # Mark the searched patent in data
            data.loc[(data['x'] == searched_x) & (data['y'] == searched_y), 'is_searched'] = True

    # Create the figure
    fig = px.scatter(
        data,
        x='x',
        y='y',
        hover_data={'x': False, 'y': False, 'title': True, 'year': True, 'codes': True},
        title='Patent Visualization',
        labels={'x': 'X Coordinate', 'y': 'Y Coordinate'},
    )

    # Adjust marker properties to highlight the searched patent
    colors = np.where(data['is_searched'], 'red', 'blue')
    sizes = np.where(data['is_searched'], 12, 6)
    fig.update_traces(marker=dict(color=colors, size=sizes))

    # Update figure layout without overriding user interactions
    fig.update_layout(
        clickmode='event+select',
        dragmode='pan',
        uirevision=True  # Preserve the state of the figure unless data changes
    )

    # If we centered on the searched patent, set the axes ranges
    if triggered_input == 'searched-patent-coords' and searched_coords:
        fig.update_xaxes(range=[xmin, xmax])
        fig.update_yaxes(range=[ymin, ymax])

    return fig

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
        x, y = coords['x'], coords['y']
        return "", {'x': x, 'y': y}
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
        # No trigger, do not update anything
        return dash.no_update, dash.no_update, dash.no_update

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger_id == 'patent-graph' and clickData:
        # Open the modal with patent details
        point = clickData['points'][0]
        x = point['x']
        y = point['y']
        details = get_patent_details(x, y)
        if details:
            title = details['title']
            abstract = details['abstract']
            return True, title, html.P(abstract)
        else:
            return dash.no_update, dash.no_update, dash.no_update
    elif trigger_id == 'modal-close-button':
        # Close the modal when the close button is clicked
        return False, dash.no_update, dash.no_update
    else:
        return dash.no_update, dash.no_update, dash.no_update

if __name__ == '__main__':
    # Run the Dash app
    app.run_server(debug=False)
