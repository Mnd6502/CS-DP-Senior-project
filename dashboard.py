import dash
from dash import dcc, html, Input, Output
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import plotly.express as px
from visualization import *

# Load your data
hurricane_data = pd.read_csv('selected_hurricane_data.csv')
exposure_data = pd.read_csv('New_exposures.csv')

# Unique hurricane names for the first dropdown
unique_years = sorted(hurricane_data['Season (Year)'].unique())
years_options = [{'label': year, 'value': year} for year in unique_years]
Seasonality = HurrSeasonality(hurricane_data)
Category = HurrCategory(hurricane_data)
YearTrend = YearTrend(hurricane_data)

app = dash.Dash(__name__)

app.layout = html.Div([
    html.Div([
        html.H1("Hurricane Report"),
        # Div for the first graph (Storms by Category)
        html.Div([
            html.H2("Storms by Category"),
            dcc.Graph(id='num-storms-by-category', figure=Category)
        ], style={'display': 'inline-block', 'width': '50%'}),

        # Div for the second graph (Number of Storms by Years)
        html.Div([
            html.H2("Number of Storms by Years"),
            dcc.Graph(id='num-storms-by-years', figure=YearTrend)
        ], style={'display': 'inline-block', 'width': '50%'}),
    ],style={'textAlign': 'center', 'margin': 'auto'},),
    
    html.Div([
        html.H2("Number of Storms by Month"),
        dcc.Graph(id='num-storms-by-month', figure=Seasonality,style={'display': 'block', 'margin-left': 'auto', 
            'margin-right': 'auto', 'width': '60%'}),
    ], style={'textAlign': 'center'},),
    
    
    
    # First dropdown to select the year
    dcc.Dropdown(
        id='year-dropdown',
        options=years_options,
        placeholder="Select a year",
        value=None  # No default value, force the user to choose
    ),
    
    # Second dropdown to select the hurricane name, options are dynamically generated
    dcc.Dropdown(
        id='name-dropdown',
        placeholder="Select a hurricane name",
        value=None  # No default value
    ),
    
    # Graph to display the wind and pressure subplots
    html.Div(id='output-container'),
    
    # html.Iframe(
    #     id='map',
    #     src=app.get_asset_url('Exposures_map.html'),  
    #     style={"height": "600px", "width": "100%"}
    # ),
    
    

])

@app.callback(
    Output('name-dropdown', 'options'),
    [Input('year-dropdown', 'value')]
)
def set_name_options(selected_year):
    if not selected_year:
        # No year is selected, return an empty options list
        return []
    filtered_data = hurricane_data[hurricane_data['Season (Year)'] == selected_year]
    unique_names = filtered_data['Name (N/A)'].unique()
    return [{'label': name, 'value': name} for name in unique_names]
@app.callback(
    Output('name-dropdown', 'value'),
    Input('year-dropdown', 'value')
)
def reset_name_dropdown(selected_year):
    # Resets the name dropdown to its default state when the year is changed
    return None

@app.callback(
    Output('output-container', 'children'),
    [Input('name-dropdown', 'value'),
     Input('year-dropdown', 'value')]
)
def update_output(selected_name, selected_year):
    # Check if the year is selected
    if not selected_year:
        return "Please select a year."

    # Get data for the selected year
    yearly_data = hurricane_data[hurricane_data['Season (Year)'] == selected_year]
    unique_names = yearly_data['Name (N/A)'].unique()

    # If only year is selected and not hurricane name, display summary without graph
    if not selected_name:
        hurricane_count = len(unique_names)
        hurricanes_list = ', '.join(unique_names)
        summary_string = f"In {selected_year}, there were {hurricane_count} hurricanes: {hurricanes_list}"
        return html.Div([html.P(summary_string)])
    
    # If both year and name are selected, filter data for the selected hurricane and create graph
    filtered_data = yearly_data[yearly_data['Name (N/A)'].str.upper() == selected_name.upper()].reset_index(drop=True)

    # Create the graph
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=filtered_data.index, y=filtered_data['Wind(WMO) (kt)'], name='Wind'), secondary_y=False)
    fig.add_trace(go.Scatter(x=filtered_data.index, y=filtered_data['Pres(WMO) (mb)'], name='Pressure'), secondary_y=True)
    fig.update_layout(autosize=False, width=700, height=500, title_text=f"Wind and Pressure for Hurricane {selected_name} in {selected_year}")
    fig.update_xaxes(title_text='Index of 6hr Interval')
    fig.update_yaxes(title_text="Wind (kt)", secondary_y=False)
    fig.update_yaxes(title_text="Pressure (mb)", secondary_y=True)
    

    fig2 = px.scatter_mapbox(filtered_data, lat='Latitude (deg_north)', lon='Longitude (deg_east)', size='Wind(WMO) (kt)',
                  color_continuous_scale=px.colors.cyclical.IceFire, size_max=20,zoom=3, mapbox_style="open-street-map")
    # fig2 = px.scatter_geo(filtered_data, lat='Latitude (deg_north)', lon='Longitude (deg_east)', size='Wind(WMO) (kt)',
    #               color_continuous_scale=px.colors.cyclical.IceFire, size_max=20)


    # Display the summary string for the selected name and the graph
    return html.Div([
        html.P(f"Pattern of {selected_name} in {selected_year}."),
        #dcc.Graph(figure=fig),
        html.Div([
            # Div for the first graph 
            html.Div([
                dcc.Graph(figure=fig)
            ], style={'display': 'inline-block', 'width': '50%'}),

            # Div for the second graph 
            html.Div([
                dcc.Graph(figure=fig2)
            ], style={'display': 'inline-block', 'width': '50%'}),
        ],style={'textAlign': 'center', 'margin': 'auto'},),    
    ])
    
if __name__ == '__main__':
    app.run_server(debug=True)
