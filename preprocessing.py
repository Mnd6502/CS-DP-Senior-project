import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from visualization import *
from Feature_engineering import *
from Geographical_map import *
import folium
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier


def main():


    # Read data
    exposures_data = pd.read_csv('Exposures.csv')
    hurricane_data = pd.read_csv('Hurricanes.csv')

    # Initial data exploration
    # print(hurricane_data.head(5))
    # print(hurricane_data.shape)
 
    # Checking For Missing Data
    na_count_hurricane = hurricane_data.isna().sum()
    # print(na_count_hurricane)
    
    # Missing Data: Special Case
    wind_invalid_count = (hurricane_data['Wind(WMO) Percentile (%)'] == -100).sum()
    pres_invalid_count = (hurricane_data['Pres(WMO) Percentile (%)'] == -100).sum()
    pres_invalid_count = (hurricane_data['Pres(WMO) Percentile (%)'] == -100).sum()
    print(f"Invalid wind entries: {wind_invalid_count}, Invalid pres entries: {pres_invalid_count}")
    print(f"Dataset UNNAMED count: {hurricane_data['Name (N/A)'].value_counts()['UNNAMED']}")

    add_date_features(hurricane_data)
    add_intensity_level(hurricane_data)

    # Data manipulation
    new_columns = [
        'Serial_Num (N/A)', 'Season (Year)', 'Num (#)', 'Basin (BB)',
        'Sub_basin (BB)', 'Name (N/A)','ISO_time', 'day', 'month', 'hour', 
        'Nature (N/A)', 'Latitude (deg_north)', 'Longitude (deg_east)',
        'Wind(WMO) (kt)', 'Pres(WMO) (mb)', 'Center (N/A)',
        'Wind(WMO) Percentile (%)', 'Pres(WMO) Percentile (%)',
        'Track_type (N/A)','Intensity'
    ]
    hurricane_data = hurricane_data[new_columns]

    # print(hurricane_data.head(5))
    # print(hurricane_data.info())
    # print(hurricane_data.describe())
    
    
    print(f"Dataset contains data of {hurricane_data['Serial_Num (N/A)'].nunique()} individual storms from {hurricane_data['ISO_time'].dt.year.min()} to {hurricane_data['ISO_time'].dt.year.max()}.")
    
    # Visualization
    count_plot('Basin (BB)',hurricane_data)
    StormCountByCategory(hurricane_data)
    HurricanePointbyYear(hurricane_data)
    NumHurricanebyYear(hurricane_data)
    HurricanePointbyMonth(hurricane_data)
    NumHurricanebyMonth(hurricane_data)
    



    # Exposures Data
    # print(exposures_data.head(5))
    # print(exposures_data.shape)
    # print(exposures_data.info())
    # print(exposures_data.describe())
    exposures_data.drop(['PolicyResultLookup','LocationLookup'],axis = 1,inplace=True)
    # Checking for missing data
    print(exposures_data.isnull().sum())

    # Correcting Data Format
    adjusted_cols = ["Total Insured Value", "Premium","Losses - Non Catastrophe","PolicyYear"]
    for col in adjusted_cols:
        exposures_data[col] = exposures_data[col].str.replace(',', '').astype(int)

    #Feature Engineering


    exposures_data['At Risk?'] = exposures_data.apply(check_at_risk, args=(hurricane_data,), axis=1)
    
    #exposures_data['NumStormsEncounter'] = exposures_data.apply(check_hurricane_encounters2, args=(hurricane_data,), axis=1)
    exposures_data[['NumStormsEncounter', 'Num6-hourEncounters','Max_wind']] = exposures_data.apply(lambda row: check_hurricane_encounters2(row, hurricane_data), axis=1).apply(pd.Series)
    region_mapping = pd.read_csv('Region mapping.csv')
    print(exposures_data['NumStormsEncounter'].sum())
            
    exposures_data['Region'] = exposures_data.apply(get_region, args=(region_mapping,), axis=1)

    location_refine(exposures_data)



    # print(exposures_data.head(5))
    # print(exposures_data.describe())
    selected_cols =['Total Insured Value','Premium','Losses - Non Catastrophe']
    # print(exposures_data[selected_cols].corr())
    # print(exposures_data[exposures_data['Losses - Non Catastrophe'] != 0])

    # Visualization
    avg_premium = exposures_data.groupby('PolicyYear')['Premium'].mean()
    unique_years = exposures_data['PolicyYear'].unique().tolist()
    total_insured = exposures_data.groupby('PolicyYear')['Total Insured Value'].sum()
    #Scatter_Plot
    scatter_plot(unique_years, avg_premium,'Policy Year','Premium','Average Premium throughout years')
    scatter_plot(unique_years, total_insured,'Policy Year','Insured Value','Insured Value throughout years')
    Line_plot(unique_years, total_insured,'Policy Year','Insured Value','Insured Value throughout years')
    # Histogram
    plot_hist('Losses - Non Catastrophe', exposures_data)
    plot_hist('Premium',exposures_data)


    # Count plot for Region
    count_plot('Region',exposures_data)

    # Count plot for At Risk?
    count_plot('At Risk?',exposures_data)

    
    # Create a base map
    # generate_hurricane_map(hurricane_data)
    # plotted_location = generate_map(exposures_data)
    # print(plotted_location)
    


   ######################################## MODEL 1 ################################################ 
    print("---------------------------Model 1-----------------------------------------")
    X2 = hurricane_data[['Season (Year)', 'day', 'month', 'hour',
        'Wind(WMO) (kt)', 'Pres(WMO) (mb)']]
    y = hurricane_data['Nature (N/A)']

    # Normalize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X2)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Implement a simple linear regression model
    model2 = KNeighborsClassifier()
    model2.fit(X_train, y_train)

    # Predict on the test set
    
    predictions = model2.predict(X_test)
    print("Precision (Weighted):", precision_score(y_test, predictions, average='weighted'))
    print("Recall (Weighted):", recall_score(y_test, predictions, average='weighted'))
    print("F1 Score (Weighted):", f1_score(y_test, predictions, average='weighted'))
    
    new_columns = [
        'Serial_Num (N/A)','Season (Year)', 'Name (N/A)','ISO_time', 'day', 'month', 'hour', 
        'Nature (N/A)', 'Latitude (deg_north)', 'Longitude (deg_east)',
        'Wind(WMO) (kt)', 'Pres(WMO) (mb)','Wind(WMO) Percentile (%)', 'Pres(WMO) Percentile (%)','Intensity'
    ]
    
    selected_columns = hurricane_data[new_columns]
    selected_columns.to_csv('selected_hurricane_data.csv', index=False)
    
    exposures_data = exposures_data[exposures_data['Location'].isin([1,2,10,11,17,25,28])]  # In-Land Location
    print(exposures_data['NumStormsEncounter'].sum())
    exposures_data.to_csv('New_exposures.csv', index=False)
    
    
    

    
    
    

    

main()