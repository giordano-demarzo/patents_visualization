# index.py

from dash import dcc, html, Input, Output
from app import app  # Import the app instance from app.py

# Import the layouts of the pages
from home import layout as home_layout
from vis_patents import layout as patents_layout
from vis_codes import layout as codes_layout

# Define the app layout
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

# Callback to render the appropriate page content
@app.callback(
    Output('page-content', 'children'),
    Input('url', 'pathname')
)
def display_page(pathname):
    if pathname == '/':
        return home_layout
    elif pathname == '/vis_patents':
        return patents_layout
    elif pathname == '/vis_codes':
        return codes_layout
    else:
        return html.H1('404: Page Not Found', style={'textAlign': 'center', 'color': 'red'})

if __name__ == '__main__':
    app.run_server(debug=True)