#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 19:37:40 2024

@author: giordano
"""

# home.py

from dash import html, dcc
import dash_bootstrap_components as dbc

layout = html.Div([
    html.H1("Welcome to the Patent Visualization App", style={'textAlign': 'center', 'color': 'white'}),
    
    # Placeholder text
    html.Div(
        [
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
        ]
    ),
    
    # Placeholder image
    html.Div(
        [
            html.Img(
                src='/assets/placeholder_image.png',  # Update with your image path
                style={
                    'width': '50%',
                    'display': 'block',
                    'margin': '20px auto'
                }
            ),
        ]
    ),
    
   # Buttons with custom white margin effect
    html.Div([
        html.Div(  # Wrapper for white margin effect
            [
                dbc.Button(
                    "Patent Space", 
                    href='/vis_patents', 
                    style={
                        'backgroundColor': '#3a3a3a', 
                        'color': 'white', 
                        'margin': '10px', 
                        'border': '2px solid #3a3a3a',  # Custom border color (same as button color)
                        'outline': 'none'  # Remove blue focus outline
                    }
                ),
                dbc.Button(
                    "Technology Space", 
                    href='/vis_codes', 
                    style={
                        'backgroundColor': '#3a3a3a', 
                        'color': 'white', 
                        'margin': '10px', 
                        'border': '2px solid #3a3a3a',  # Custom border color (same as button color)
                        'outline': 'none'  # Remove blue focus outline
                    }
                )
            ],
        ),
    ], style={'textAlign': 'center', 'marginTop': '50px'}),
    
], style={'backgroundColor': '#2c2c2c', 'height': '100vh', 'paddingTop': '50px'})