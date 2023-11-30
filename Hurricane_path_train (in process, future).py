import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.express as px
import plotly.graph_objs as go

# Load your historical hurricane data
data = pd.read_csv('selected_hurricane_data.csv')

# Select data for a specific hurricane and season
hurricane_data = data[(data['Name (N/A)'] == 'IVAN') & (data['Season (Year)'] == 2004)].reset_index(drop=True)

# Scale the features
scaler = StandardScaler()
hurricane_data_scaled = scaler.fit_transform(hurricane_data[['Latitude (deg_north)', 'Longitude (deg_east)', 'Wind(WMO) (kt)']])

# Define the sequence parameters
n_input = 5
n_features = 3  # We're looking at latitude, longitude, and wind speed

# Split the data into training and testing sets
train_size = int(len(hurricane_data_scaled) * 0.8)
test_size = len(hurricane_data_scaled) - train_size
train_data, test_data = hurricane_data_scaled[:train_size, :], hurricane_data_scaled[train_size:, :]

# Create generators
train_generator = TimeseriesGenerator(train_data, train_data, length=n_input, batch_size=1)
test_generator = TimeseriesGenerator(test_data, test_data, length=n_input, batch_size=1)

# Define the LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_input, n_features)))
model.add(Dense(n_features))  # Output layer with 3 units for predicting latitude, longitude, and wind speed
model.compile(optimizer='adam', loss='mse')

# Fit the model and save the history
history = model.fit(train_generator, epochs=200)

# Predict using the test_generator
predictions = model.predict(test_generator)

# Inverse transform the predictions and test data
predictions_rescaled = scaler.inverse_transform(predictions)
test_data_rescaled = scaler.inverse_transform(test_data[n_input:])

# Calculate error metrics for each predicted feature
rmse_lat = np.sqrt(mean_squared_error(test_data_rescaled[:, 0], predictions_rescaled[:, 0]))
rmse_lon = np.sqrt(mean_squared_error(test_data_rescaled[:, 1], predictions_rescaled[:, 1]))
rmse_wind = np.sqrt(mean_squared_error(test_data_rescaled[:, 2], predictions_rescaled[:, 2]))

mae_lat = mean_absolute_error(test_data_rescaled[:, 0], predictions_rescaled[:, 0])
mae_lon = mean_absolute_error(test_data_rescaled[:, 1], predictions_rescaled[:, 1])
mae_wind = mean_absolute_error(test_data_rescaled[:, 2], predictions_rescaled[:, 2])

# Print the loss values and model summary
print(f"Training Loss: {history.history['loss'][-1]}")  # Last loss value from the training
print(f"RMSE Latitude: {rmse_lat}")
print(f"RMSE Longitude: {rmse_lon}")
print(f"RMSE Wind Speed: {rmse_wind}")
print(f"MAE Latitude: {mae_lat}")
print(f"MAE Longitude: {mae_lon}")
print(f"MAE Wind Speed: {mae_wind}")
print(model.summary())

# Convert the numpy arrays to pandas DataFrame
predicted_path_df = pd.DataFrame(predictions_rescaled, columns=['Latitude', 'Longitude', 'Wind'])
true_path_df = pd.DataFrame(test_data_rescaled, columns=['Latitude', 'Longitude', 'Wind'])

# Add an index to serve as a hover name in the plot, indicating the time step
predicted_path_df['Time Step'] = range(len(predicted_path_df))
true_path_df['Time Step'] = range(len(true_path_df))

# Create the scatter_geo plot for the true path
fig = px.scatter_geo(
    hurricane_data,
    lat='Latitude (deg_north)',
    lon='Longitude (deg_east)',
    size='Wind(WMO) (kt)',
    title='Hurricane Path Prediction Comparison',
    labels={'Longitude (deg_east)': 'Longitude', 'Latitude (deg_north)': 'Latitude'},
)

# Add the forecasted path to the plot
fig.add_trace(
    go.Scattergeo(
        lon=predicted_path_df['Longitude'],
        lat=predicted_path_df['Latitude'],
        mode='lines+markers',
        text=predicted_path_df.apply(lambda row: f"Lat: {row['Latitude']}<br>Lon: {row['Longitude']}<br>Wind: {row['Wind']} kt", axis=1),
        line=dict(color='orange', width=2, dash='dash'),
        marker=dict(size=10, symbol='circle', color='orange'),
        name='Forecasted Path'
    )
)

# Update layout with title and legend
fig.update_layout(
    showlegend=True,
    geo=dict(
        scope='north america',  # assuming the hurricane is in the North America region
        projection_type='natural earth'
    )
)

# Show the figure
fig.show()
