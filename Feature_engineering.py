import pandas as pd
from math import radians, sin, cos, sqrt, atan2


def add_date_features(hurricane_data):
    hurricane_data['ISO_time (YYYY-MM-DD HH:MM:SS)'] = pd.to_datetime(hurricane_data['ISO_time (YYYY-MM-DD HH:MM:SS)'])
    hurricane_data['day'] = hurricane_data['ISO_time (YYYY-MM-DD HH:MM:SS)'].dt.day
    hurricane_data['month'] = hurricane_data['ISO_time (YYYY-MM-DD HH:MM:SS)'].dt.month
    hurricane_data['hour'] = hurricane_data['ISO_time (YYYY-MM-DD HH:MM:SS)'].dt.hour
    hurricane_data.rename(columns={'ISO_time (YYYY-MM-DD HH:MM:SS)': 'ISO_time'}, inplace=True)
    
    
def classify_intensity(pressure):
    if pressure == -100 or pressure == 0:
        return 'Unknown'
    elif pressure <= 940:
        return 'High'
    elif 940 < pressure <= 980:
        return 'Medium'
    else:
        return 'Low'
    
def add_intensity_level(data):  
    data['Intensity'] = data['Pres(WMO) (mb)'].apply(classify_intensity)

def get_region(row,region_mapping):
    if row['Longitude'] < region_mapping.loc[3, 'Longitude Boundary Eastern End']:
        return 'West'
    elif row['Latitude'] < region_mapping.loc[0, 'Latitude Boundary Northern End']:
        return 'South'
    elif row['Longitude'] < region_mapping.loc[2, 'Longitude Boundary Eastern End']:
        return 'Midwest'
    else:
        return 'Northeast'
    
    
def location_refine(df):

    location_dict = {}
    location_counter = 1

    for index, row in df.iterrows():
        lat_long = (row['Latitude'], row['Longitude'])
        if lat_long not in location_dict:
            # Assign the current counter value and update the counter
            location_dict[lat_long] = location_counter
            location_counter += 1
        # Assign the location value from the dictionary to the dataframe
        df.at[index, 'Location'] = location_dict[lat_long]
        
    location_list = [[location, lat, long] for (lat, long), location in location_dict.items()]
    location_df = pd.DataFrame(location_list, columns=['Location', 'Latitude', 'Longitude'])
    
    location_df.to_csv('Location_lookup.csv', index=False)

def check_at_risk(row,hurricane_data):
    lat_min = row['Latitude'] - 1
    lat_max = row['Latitude'] + 1
    long_min = row['Longitude'] - 1
    long_max = row['Longitude'] + 1
    count = len(hurricane_data[(hurricane_data['Latitude (deg_north)'] < lat_max) &
                               (hurricane_data['Latitude (deg_north)'] > lat_min) &
                               (hurricane_data['Longitude (deg_east)'] < long_max) &
                               (hurricane_data['Longitude (deg_east)'] > long_min)])
    return "At Risk" if count > 0 else "Not At Risk"

# FRIENDLY TYPE
def check_hurricane_encounters(row, hurricane_data):
    # Define the search window based on the latitude and longitude
    lat_min = row['Latitude'] - 1
    lat_max = row['Latitude'] + 1
    long_min = row['Longitude'] - 1
    long_max = row['Longitude'] + 1
    policy_year = row['PolicyYear']
    
    # Filter the hurricane data for the specific year and location range
    year_hurricanes = hurricane_data[(hurricane_data['Season (Year)'] == policy_year) &
                                     (hurricane_data['Latitude (deg_north)'] <= lat_max) &
                                     (hurricane_data['Latitude (deg_north)'] >= lat_min) &
                                     (hurricane_data['Longitude (deg_east)'] <= long_max) &
                                     (hurricane_data['Longitude (deg_east)'] >= long_min)]
    
    # Get unique hurricane encounters by Serial_Num
    unique_hurricanes = year_hurricanes['Serial_Num (N/A)'].unique()
    
    return len(unique_hurricanes)


# Define the Haversine formula
def haversine(lat1, lon1, lat2, lon2):
    # Radius of the Earth in kilometers (approximation)
    R = 6371.0

    # Convert coordinates from degrees to radians
    lat1_rad = radians(lat1)
    lon1_rad = radians(lon1)
    lat2_rad = radians(lat2)
    lon2_rad = radians(lon2)

    # Change in coordinates
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    # Haversine calculation
    a = sin(dlat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    # Distance in kilometers
    distance = R * c
    return distance

# Function to check hurricane encounters within a 1-degree radius
def check_hurricane_encounters2(row, hurricane_data):
    policy_year = row['PolicyYear']
    
    # Filter the hurricane data for the specific year
    year_hurricanes = hurricane_data[hurricane_data['Season (Year)'] == policy_year]

    # Check for each hurricane data point if it is within the 1-degree radius
    hurricanes_within_radius = []
    six_hour_count = 0
    max_wind = 0
    for _, hurricane in year_hurricanes.iterrows():
        distance = haversine(row['Latitude'], row['Longitude'], 
                             hurricane['Latitude (deg_north)'], hurricane['Longitude (deg_east)'])
        # If the hurricane is within a 1-degree radius, add its Serial_Num to the list
        if distance <= 111: # 1 degree approximated in kilometers
            hurricanes_within_radius.append(hurricane['Serial_Num (N/A)'])
            six_hour_count +=1
            max_wind = max(max_wind,hurricane['Wind(WMO) (kt)'])
    # Return the unique Serial_Num values of hurricanes within the 1-degree radius
    
    unique_count = len(set(hurricanes_within_radius))
    
    return unique_count, six_hour_count, max_wind