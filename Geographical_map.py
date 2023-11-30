import folium
from folium.plugins import Search
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

def is_on_land(lat, lon):
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    point = Point(lon, lat)
    return world.geometry.contains(point).any()

def generate_map(exposures_data):
    m = folium.Map(location=[37.0902, -95.7129], zoom_start=5)

    location_group = folium.FeatureGroup(name="Locations")
    plotted_locations = set()

    for index, row in exposures_data.iterrows():
        lat, lon = row['Latitude'], row['Longitude']
        
        if is_on_land(lat, lon):
            col = 'blue'
            if 24.396308 < lat < 49.384358 and -125.000000 < lon < -66.934570:
                col = 'red'
            
            folium.Marker(
                [lat, lon],
                icon=folium.Icon(color=col),
                popup=row['Location']
            ).add_to(location_group)
            
            plotted_locations.add(row['Location'])
    location_group.add_to(m)
    search = Search(
        layer=location_group,
        geom_type="Point",
        placeholder="Search for a Location",
        collapsed=False,
        search_label='popup',
        position='topright'
    ).add_to(m)

    m.save('Exposures_map.html')
    return plotted_locations

def generate_hurricane_map(hurricane_data):

    # Create a base map
    m = folium.Map(location=[37.0902, -95.7129], zoom_start=5)  # Centered around the US


    for year, group in hurricane_data.groupby('Season (Year)'):
        feature_group = folium.FeatureGroup(name=str(year))
        for _, row in group.iterrows():
            folium.CircleMarker(
                location=[row['Latitude (deg_north)'], row['Longitude (deg_east)']],
                radius=5,
                color='blue',
                fill=True,
                fill_color='blue',
                popup=row['Name (N/A)'],
            ).add_to(feature_group)
        feature_group.add_to(m)



    # Add layer control to toggle data layers
    folium.LayerControl().add_to(m)


    # # Save the map as an HTML file

    m.save('Hurricane_map.html')

# def main():
#     exposures_data = pd.read_csv('./data/Exposures.csv')
#     plotted_locations = generate_map(exposures_data)
#     print(list(plotted_locations))
    
# main()
