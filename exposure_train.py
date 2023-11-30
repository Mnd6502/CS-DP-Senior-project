import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder,  StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

### Simple Model

exposures_data = pd.read_csv('New_exposures.csv')

X = exposures_data[['Premium', 'Losses - Non Catastrophe']]
y = exposures_data['Total Insured Value']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lr = LinearRegression()
model = lr.fit(X_train, y_train)
predictions = model.predict(X_test)

scores = mean_squared_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)
print("-" * 40)
print(f"MSE: {mse}")
print(f"MAE: {mae}")
print(f"RMSE: {rmse}")
print(f"R2: {r2}")
print("=" * 40)
plt.scatter(y_test, predictions, alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
plt.show()

###################################



X = exposures_data.drop(['Location','Total Insured Value','At Risk?','Region', 'Num6-hourEncounters','Max_wind'], axis=1)
y = exposures_data['Total Insured Value']


# y = scaler.fit_transform(np.array(y).reshape(-1,1))
# y = y.ravel()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scores = {}
predictions_dict = {}

encoder = OneHotEncoder(drop="first")
scaler = StandardScaler()

categorical_columns = ['Multi-Story?']
numerical_columns = ['Latitude', 'Longitude', 'Losses - Non Catastrophe','Premium',  'PolicyYear', 'NumStormsEncounter']

# Create a column transformer that will apply the encoder and the scaler to the right columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', scaler, numerical_columns),
        ('cat', encoder, categorical_columns)
    ])

# Create pipelines for each regression model
pipelines = {
    "Linear Regression": Pipeline([("preprocessor", preprocessor), ("regressor", LinearRegression())]),
    "Ridge Regression": Pipeline([("preprocessor", preprocessor), ("regressor", Ridge())]),
    "Lasso Regression": Pipeline([("preprocessor", preprocessor), ("regressor", Lasso())]),
    "Random Forest": Pipeline([("preprocessor", preprocessor), ("regressor", RandomForestRegressor(random_state=42))]),
    "Gradient Boosting": Pipeline([("preprocessor", preprocessor), ("regressor", GradientBoostingRegressor(random_state=42))])
}


# Apply transformations manually to get a transformed DataFrame
preprocessor.fit(X_train)
X_train_transformed = preprocessor.transform(X_train)
X_train_transformed = pd.DataFrame(X_train_transformed, columns=(numerical_columns + list(preprocessor.named_transformers_['cat'].get_feature_names_out())))


# Calculate the correlation matrix
correlation_matrix = X_train_transformed.corr()

# Plot the correlation matrix using a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()


# Train each model and evaluate performance
scores = {}
for name, pipeline in pipelines.items():
    model = pipeline.fit(X_train, y_train)
    predictions = model.predict(X_test)
    scores[name] = mean_squared_error(y_test, predictions)
    predictions_dict[name] = predictions  # Store predictions
    
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    
    scores[name] = {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }
    print(f"Model: {name}")
    print("-" * 40)
    print(f"MSE: {mse}")
    print(f"MAE: {mae}")
    print(f"RMSE: {rmse}")
    print(f"R2: {r2}")
    print("=" * 40)


# Hyperparameter grids
param_grids = {
    "Linear Regression": {'regressor__fit_intercept': [True, False]},
    "Ridge Regression": {'regressor__alpha': [1, 10, 100, 1000]},
    "Lasso Regression": {'regressor__alpha': [0.001, 0.01, 0.1, 1, 10]},
    "Random Forest": {'regressor__n_estimators': [10, 50, 100], 'regressor__max_depth': [None, 10, 20, 30]},
    "Gradient Boosting": {'regressor__n_estimators': [50, 100, 200], 'regressor__learning_rate': [0.01, 0.1, 0.2]}
}


grid_search_objects = {}
for name, pipeline in pipelines.items():
    grid_search = GridSearchCV(estimator=pipeline,
                               param_grid=param_grids[name],
                               scoring='neg_mean_squared_error',
                               cv=5,
                               verbose=1,
                               n_jobs=-1)
    grid_search_objects[name] = grid_search

# Model training and evaluation
best_models = {}
for name, grid_search in grid_search_objects.items():
    grid_search.fit(X_train, y_train)
    best_models[name] = grid_search.best_estimator_
    
        
    
    predictions = best_models[name].predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    
    predictions_dict[f"Best {name}"] = predictions
    
    scores[f"Best {name}"] = {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }
    
    print(f"Best Model: {name}")
    print("-" * 40)
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"MSE: {mse}")
    print(f"MAE: {mae}")
    print(f"RMSE: {rmse}")
    print(f"R2: {r2}")
    print("=" * 40)
    
scores_df = pd.DataFrame(scores).T
predictions_df = pd.DataFrame(predictions_dict)

# Plot for Mean Squared Error (MSE)
plt.figure(figsize=(10, 6))
sns.barplot(x=scores_df.index, y="mse", data=scores_df)
plt.title('Mean Squared Error (MSE)')
plt.ylabel('MSE')
plt.xticks(rotation=30)
plt.show()



# Plot for Mean Absolute Error (MAE)
plt.figure(figsize=(10, 6))
sns.barplot(x=scores_df.index, y="mae", data=scores_df)
plt.title('Mean Absolute Error (MAE)')
plt.ylabel('MAE')
plt.xticks(rotation=30)
plt.show()

# Plot for Root Mean Squared Error (RMSE)
plt.figure(figsize=(10, 6))
sns.barplot(x=scores_df.index, y="rmse", data=scores_df)
plt.title('Root Mean Squared Error (RMSE)')
plt.ylabel('RMSE')
plt.xticks(rotation=30)
plt.show()

# Plot for R-Squared (R2)
plt.figure(figsize=(10, 6))
sns.barplot(x=scores_df.index, y="r2", data=scores_df)
plt.title('R-Squared')
plt.ylabel('R2')
plt.xticks(rotation=30)
plt.show()
# Visualization of predictions vs actual values for the best model
# Select the best model based on a metric (e.g., RMSE)

for column in predictions_df:
    predictions = predictions_df[column]
    # Scatter plot for actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, predictions, alpha=0.3)
    plt.title(f'Actual vs Predicted Values for {column}')
    plt.xlabel('Actual Total Insured Value')
    plt.ylabel('Predicted Total Insured Value')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
    plt.show()
    
    

