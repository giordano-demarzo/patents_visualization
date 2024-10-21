# home.py

from dash import html, dcc, Input, Output, callback
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# --- Data Loading ---
# Load your data
df = joblib.load('data/tech_code_embeddings_2d_with_shap.pkl')

# --- Create the Scatter Plot ---
scatter = px.scatter(
    df,
    x='TSNE_1',
    y='TSNE_2',
    hover_data={'Tech_Code': True, 'Title': True, 'ID': True},
    labels={'TSNE_1': 'TSNE Component 1', 'TSNE_2': 'TSNE Component 2'},
    title="Embeddings of Technological Codes",
    color='Tech_Code',
    custom_data=['Tech_Code', 'Title', 'ID']
)

# Update the layout to match your app's style
scatter.update_layout(
    uirevision=True,
    plot_bgcolor='#2c2c2c',
    paper_bgcolor='#2c2c2c',
    font_color='white',
    title={'x': 0.5},  # Center the title
    legend_title_font_color='white',
)

# --- Helper Function for Color Interpolation ---
def interpolate_color(shap_value):
    if shap_value <= 0.5:
        # Interpolate between blue and white
        ratio = shap_value / 0.5
        r = int(255 * ratio)
        g = int(255 * ratio)
        b = 255
    else:
        # Interpolate between white and red
        ratio = (shap_value - 0.5) / 0.5
        r = 255
        g = int(255 * (1 - ratio))
        b = int(255 * (1 - ratio))
    return f'rgb({r}, {g}, {b})'

# --- Layout ---
layout = html.Div([
    html.H1(
        "Welcome to the Patent Visualization App",
        style={'textAlign': 'center', 'color': 'white'}
    ),
    
    # Introductory text
    html.Div([
        html.P(
            "Explore interactive visualizations of patents and technological codes. "
            "Use the navigation links below to begin.",
            style={
                'color': 'white',
                'fontSize': '18px',
                'textAlign': 'center',
                'maxWidth': '800px',
                'margin': '20px auto'
            }
        ),
    ]),
    
    # Navigation buttons
    html.Div([
        dbc.Button(
            "Patent Space",
            href='/vis_patents',
            style={
                'backgroundColor': '#3a3a3a',
                'color': 'white',
                'margin': '10px',
                'border': '2px solid #3a3a3a',
                'outline': 'none'
            }
        ),
        dbc.Button(
            "Technology Space",
            href='/vis_codes',
            style={
                'backgroundColor': '#3a3a3a',
                'color': 'white',
                'margin': '10px',
                'border': '2px solid #3a3a3a',
                'outline': 'none'
            }
        )
    ], style={'textAlign': 'center', 'marginTop': '50px'}),
    
    # Visualization
    dbc.Container([
        # Explanatory text
        dbc.Row([
            dbc.Col([
                html.P(
                    "The plot on the left is a t-SNE projection of the embeddings of a selection of technological codes. Each dot represents the embedding of a technological code based on the text of a specific patent in which it appears.\n\n"
                    "The embeddings are obtained from a model that has been fine-tuned on patent abstracts and claims, where technological codes have been added to the model as additional special tokens. That is, we fine-tune the model on strings of the form 'TECH_CODE1 TECH_CODE2 ... TECH_CODE10 [END_TECH_LIST_TOKEN] patent text...'.\n\n"
                    "Hover over a dot to see the patent's text on the right. The color of each token in the text corresponds to its importance in the model's prediction of the presence of that code in the list of technological codes for that patent. The color is determined by the SHAP value of the token, which measures the token's impact on the model's prediction. These values provide insight into which parts of the text influence the model's understanding (and embedding) of the meaning of the target technological code.\n\n"
                    "The same technology forms several clusters, suggesting that the same code can have different meanings based on the context in which it is used. For example, the code 'G06F3' (double-click on the legend to show only the corresponding points). By examining the words highlighted in different clusters, it's clear that in one cluster, the code refers to transferring data to printers, in another to transferring to and from storage devices, and in other cases to the use of a mouse. This demonstrates the power of embeddings to capture the meaning of a code in different contexts.",
                    style={'color': 'white'}
                ),
            ], width=10, style={'margin-left': 'auto', 'margin-right': 'auto'})  # Use CSS to center the column
        ], style={'marginTop': '20px', 'fontSize': '16px'}),

        dbc.Row([
            dbc.Col([
                dcc.Graph(
                    id='scatter-plot',
                    figure=scatter,
                    config={'displayModeBar': True, 'scrollZoom': True},
                    style={'height': '600px'}
                ),
            ], width=6),
            dbc.Col([
                html.Div(
                    id='text-highlight',
                    style={
                        'whiteSpace': 'pre-wrap',
                        'fontSize': '18px',
                        'color': 'white',
                        'backgroundColor': '#2c2c2c',
                        'padding': '10px',
                        'border': '1px solid #555',
                        'height': '600px',
                        'overflowY': 'auto',
                    }
                ),
            ], width=6),
        ], align="center"),
    ], fluid=True),
    
], style={'backgroundColor': '#2c2c2c', 'paddingTop': '50px'})

# --- Callbacks ---
# Update the text highlight based on hoverData
@callback(
    Output('text-highlight', 'children'),
    Input('scatter-plot', 'hoverData')
)
def update_text(hoverData):
    if hoverData is None:
        return "Hover over a dot to see the corresponding text."
    
    # Extract data from hoverData
    customdata = hoverData['points'][0]['customdata']
    code = customdata[0]
    title = customdata[1]
    id = customdata[2]
    
    # Filter the dataframe for the selected point
    row = df[(df['ID'] == id) & (df['Tech_Code'] == code)]
    if row.empty:
        return "Data not found."
    
    tokens = row['Text'].values[0]
    shap_values = row['Shap'].values[0]
    descr = row['Description'].values[0]
    title = row['Title'].values[0]
    
    # Build the highlighted text
    colored_text = [
        html.Span("Title: ", style={'font-weight': 'bold'}),
        html.Span(title),
        html.Br(),
        html.Span("Technology: ", style={'font-weight': 'bold'}),
        html.Span(descr),
        html.Br(), html.Br()
    ]
    
    for token, shap_value in zip(tokens, shap_values):
        color = interpolate_color(shap_value)
        colored_text.append(
            html.Span(
                token + ' ',
                style={
                    'backgroundColor': color,
                    'color': 'black',
                    'padding': '0px',
                    'borderRadius': '0px'
                }
            )
        )
    
    return html.Div(colored_text)
