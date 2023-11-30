import dash
from PIL import Image
from dash import dcc, html, Input, Output
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import plotly.express as px
from visualization import *

# Load your data
hurricane_data = pd.read_csv('selected_hurricane_data.csv')
exposure_data = pd.read_csv('New_Exposures.csv')
location_lookup_df = pd.read_csv('./data/Location_lookup.csv')

# Unique hurricane names for the first dropdown
unique_years = sorted(hurricane_data['Season (Year)'].unique())
years_options = [{'label': year, 'value': year} for year in unique_years]
Seasonality = HurrSeasonality(hurricane_data)
Category = HurrCategory(hurricane_data)
YearTrend = YearTrend(hurricane_data)
location_options = [1,2,10,11,17,25,28]

pil_img = Image.open("Image/Model 1/Figure_1.png")
pil_img2 = Image.open("Image/Model 1/Figure_2.png")
pil_img3 = Image.open("Image/Model 1/Figure_3.png")
pil_img4 = Image.open("Image/Model 1/Figure_4.png")

pil_md2_a = Image.open("Image/output.png")
pil_md2_f1 = Image.open("Image/Tune_output.png")
pil_NN = Image.open("Image/nn_output.png")
pil_NN_cfm = Image.open("Image/NN_cfm_output.png")

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
    
    html.Hr(), 

    html.Div([html.H1("Exposure Report")],style={'textAlign': 'center', 'margin': 'auto'},),
    html.Div([
        
        html.Div([
            html.H3("Pick a Location"),
            dcc.Dropdown(
                id='location-dropdown',
                options=[{'label': i, 'value': i} for i in exposure_data['Location'].unique()],
                value=exposure_data['Location'].unique()[0]
            ),]),
        html.Div([
            html.H3("Select the year"),
            dcc.Input(
                id='year-input',
                type='number',
                value=exposure_data['PolicyYear'].max(),  # Default to the most recent year
                style={'height':'30px','width': '100px'}
        ),])
    ], style={'padding': '10px', 'display': 'flex', 'justifyContent': 'space-between'}),
    
    html.Div(id='risk-metrics-display'),  # Placeholder for risk metrics

    html.Div([
        dcc.Graph(id='historical-premiums-chart', style={'margin-right': '10px'}), 
        dcc.Graph(id='historical-losses-chart', style={'margin-left': '10px'}), 
    ], style={'display': 'flex', 'justifyContent': 'center'}), 
    
    html.Div(dcc.Graph(id='historical-insured-value-chart'), style={
        'display': 'block', 
        'margin-left': 'auto',
        'margin-right': 'auto', 
        'width': '60%'
    }), 
    html.Hr(),
    html.Div([html.H1("Hurricane & Exposure Analysis Model")],style={'textAlign': 'center', 'margin': 'auto'},),
    html.Div([html.H2("Model 1: Exposure Insured Value Prediction")]),
    html.Div([
        html.Div([
            html.Div([
                html.Img(src=pil_img, style={'display':'inline-block', 'float':'left','width': '50%', 'height': '50%', 'object-fit': 'cover'})
            ]),
            html.Div([
                html.Img(src=pil_img2, style={'display':'inline-block', 'float':'right','width': '50%', 'height': '50%', 'object-fit': 'cover'})
            ]),
        ]),
        html.Div([
            html.Div([
                html.Img(src=pil_img3, style={'display':'inline-block', 'float':'left','width': '50%','height': '50%', 'object-fit': 'cover'})
            ]),
            html.Div([
                html.Img(src=pil_img4, style={'display':'inline-block', 'float':'right','width': '50%','height': '50%', 'object-fit': 'cover'})
            ]),
        ])
    ]),
    html.Hr(), 
    html.Div([html.H2("Model 2: Hurricane Classifier")]),
    html.Div([
        html.Div([
            html.Img(src=pil_md2_a, style={'display':'inline-block', 'float':'left','width': '50%', 'height': '50%'})
        ]),
        html.Div([
            html.Img(src=pil_md2_f1, style={'display':'inline-block', 'float':'right','width': '50%', 'height': '50%'})
        ]),
    ]),
    html.Hr(),
    html.Div([
        html.H3("NN Training & Testing accuracy "),
            html.Img(src=pil_NN, style={'display':'inline-block','width': '50%', 'height': '50%', 'object-fit': 'cover'})
        ], style={'textAlign': 'center', 'margin': 'auto'}),     
    html.Div([
            html.H3("NN Confusion Matrix "),
            html.Img(src=pil_NN_cfm, style={'display':'inline-block','width': '30%', 'height': '30%', 'object-fit': 'cover'})
        ], style={'textAlign': 'center', 'margin': 'auto'}),
    html.Hr(), 
    
    html.Div([html.H2("Model 3: Hurricane Path Prediction (On Going/Future)")])
    
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
    
    Loc_df = location_lookup_df[location_lookup_df['Location'].isin([1,2,10,11,17,25,28])]

    fig2 = px.scatter_mapbox(filtered_data, lat='Latitude (deg_north)', lon='Longitude (deg_east)', size='Wind(WMO) (kt)',
                  color_continuous_scale=px.colors.cyclical.IceFire, size_max=20,zoom=3, mapbox_style="open-street-map")
    fig2.add_trace(go.Scattermapbox(
        name = 'Location',
        mode = "markers",
        lat = Loc_df['Latitude'],
        lon = Loc_df['Longitude'],
        marker = {'color': "red", 
                "size": 10},
        text= Loc_df['Location'],
        hoverinfo='text'
        ))


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
    
@app.callback(
    [Output('risk-metrics-display', 'children'),
     Output('historical-premiums-chart', 'figure'),
     Output('historical-losses-chart', 'figure'),
     Output('historical-insured-value-chart', 'figure')],
    [Input('location-dropdown', 'value'),
     Input('year-input', 'value')]
)


def update_dashboard(selected_location, selected_year):
    # Filter exposure data for the selected location and the last 5 years
    
    filtered_data = exposure_data[
        (exposure_data['Location'] == selected_location) &
        (exposure_data['PolicyYear'] > selected_year - 5) &
        (exposure_data['PolicyYear'] <= selected_year)
    ]
    
    # Aggregate data for locations with multiple entries per year if necessary
    aggregated_data = filtered_data.groupby('PolicyYear').agg({
        'Premium': 'sum',
        'Losses - Non Catastrophe': 'sum',
        'Total Insured Value': 'sum'
    }).reset_index()
    
    at_risk = 'Yes' if exposure_data[exposure_data['Location'] == selected_location]['At Risk?'].eq('At Risk').any() else 'No'
    hurricane_encounters = exposure_data[(exposure_data['NumStormsEncounter'] > 0) & (exposure_data['Location'] == selected_location)
                                         & (exposure_data['PolicyYear'] <= selected_year)] ['PolicyYear'].nunique()
    Num_hurricane = exposure_data[(exposure_data['Location'] == selected_location) & (exposure_data['PolicyYear'] <= selected_year)]['NumStormsEncounter'].sum()

    location_info = location_lookup_df[location_lookup_df['Location'] == selected_location].iloc[0]
    region = location_info['Region']
    latitude = location_info['Latitude']
    longitude = location_info['Longitude']
    
    
    max_insured_value = aggregated_data['Total Insured Value'].max()
    fixed_y_axis_range = (0, max_insured_value * 4/3)

    # Create charts using Plotly
    premiums_chart = px.bar(aggregated_data, x='PolicyYear', y='Premium', title='Historical Premiums',color_discrete_sequence=['red'])
    losses_chart = px.bar(aggregated_data, x='PolicyYear', y='Losses - Non Catastrophe', title='Historical Losses',color_discrete_sequence=['red'])
    insured_value_chart = px.bar(aggregated_data, x='PolicyYear', y='Total Insured Value', 
                                title='Historical Insured Value',
                                color_discrete_sequence=['red'],
                                range_y=fixed_y_axis_range)
    

    risk_metrics_display = html.Div([
        html.Div([
            html.Table([
                html.Thead(html.Tr([html.Th("Location Information", style={'border': '1px solid black', 'textAlign': 'center', 'margin': 'auto'})])),
                html.Tbody([
                    html.Tr([html.Td("Region:", style={'border': '1px solid black'}), html.Td(region, style={'border': '1px solid black'})]),
                    html.Tr([html.Td("Latitude:", style={'border': '1px solid black'}), html.Td(latitude, style={'border': '1px solid black'})]),
                    html.Tr([html.Td("Longitude:", style={'border': '1px solid black'}), html.Td(longitude, style={'border': '1px solid black'})])
                ])
            ], style={'border-collapse': 'collapse'})
        ], style={'margin-right': '10px'}),

        html.Div([
            html.Table([
                html.Thead(html.Tr([html.Th("Risk Information", style={'border': '1px solid black','textAlign': 'center', 'margin': 'auto'})])),
                html.Tbody([
                    html.Tr([html.Td("At Risk for Hurricane Damage:", style={'border': '1px solid black'}), html.Td(at_risk, style={'border': '1px solid black'})]),
                    html.Tr([html.Td("Number of Years with Hurricane Encounters Since 1980:", style={'border': '1px solid black'}), html.Td(hurricane_encounters, style={'border': '1px solid black'})]),
                    html.Tr([html.Td("Number of Hurricanes Since 1980:", style={'border': '1px solid black'}), html.Td(Num_hurricane, style={'border': '1px solid black'})])
                ])
            ], style={'border-collapse': 'collapse'}),
        ], style={'margin-left': '10px'})
    ], style={'display': 'flex'})
  
    return risk_metrics_display, premiums_chart, losses_chart, insured_value_chart
    
if __name__ == '__main__':
    app.run_server(debug=True)
