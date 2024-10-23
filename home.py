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
            "Explore interactive visualizations of patents and technological codes. \n Understand how Large Language Models capture technologies within patents. \n Use the navigation links below to visualize the Patent Space and the Technology Space",
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
                    "The plot on the left shows a t-SNE projection of embeddings for selected technological codes. Each point represents the embedding of a technological code, derived from the text of a specific patent where the code appears. These embeddings are obtained from a model fine-tuned on patent abstracts and claims, where technological codes are treated as special tokens. The model is trained on sequences like:"
                    "\n\n"
                    "'TECH_CODE1 TECH_CODE2 ... [END_TECH_LIST_TOKEN] patent text...'."
                    "\n\n"
                    "Hover over any point to see the corresponding patent's text on the right. The color of each token in the text reflects its importance in predicting the presence of a particular technological code. These colors are based on SHAP values, which highlight the impact of each token on the model's prediction. This allows us to understand which parts of the text influence the model’s comprehension of the code. Interestingly, some codes form multiple clusters, indicating that their meaning can vary depending on context. For example, the code 'G06F3' clusters differently when used for data transfer to printers, to storage devices, or for mouse interactions. This illustrates the power of embeddings to capture nuanced meanings of technological codes in various contexts.",
                    style={'color': 'white', 'whiteSpace': 'pre-line'}
                ),
            ], width=10, style={'margin-left': 'auto', 'margin-right': 'auto'})  # Use CSS to center the column
        ], style={'marginTop': '20px', 'fontSize': '14px'}),
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
