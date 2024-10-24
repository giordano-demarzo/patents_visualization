# app.py

import dash
import dash_bootstrap_components as dbc

# Initialize the Dash app
app = dash.Dash(
    __name__,
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
)
server = app.server  # Expose the WSGI server for deployment