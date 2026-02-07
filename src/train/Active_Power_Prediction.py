#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 16:46:04 2024

@author: athanasiospaidoulias
"""

###----------------- PACKAGES -----------------###
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import joblib
import json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from prophet import Prophet
from sklearn.model_selection import ParameterGrid



###----------------- 1. READ DATA -----------------###
# Define the column types based on the dataset description
column_types = {
    'date': 'str',                    # Date and time (to be parsed as datetime)
    'active_power': 'float32',        # Active power in watts
    'current': 'float32',             # Electric current in amperes
    'voltage': 'float32',             # Electric tension in volts
    'apparent_power': 'float32',      # Computed apparent power in volt-amperes
    'reactive_power': 'float32',      # Computed reactive power in volt-amperes
    'power_factor': 'float32',        # Computed power factor (dimensionless)

    # OpenWeather data
    'main': 'category',               # General weather description (categorical)
    'description': 'category',        # Detailed weather description (categorical)
    'temp': 'float32',                # Temperature in °C
    'feels_like': 'float32',          # Temperature sensation in °C
    'temp_min': 'float32',            # Minimum temperature in °C
    'temp_max': 'float32',            # Maximum temperature in °C
    'pressure': 'int32',              # Atmospheric pressure in ATM
    'humidity': 'int32',              # Humidity percentage
    'speed': 'float32',               # Wind speed in km/h
    'deg': 'int32',                   # Wind direction in degrees
    'temp_t+1': 'float32',            # Forecasted temperature at t+1 in °C
    'feels_like_t+1': 'float32'       # Forecasted temperature sensation at t+1 in °C
}


# Read the data with specified dtypes and parse 'timestamp' as datetime
data = pd.read_csv('energy_weather_raw_data.csv', dtype=column_types, parse_dates=['date'])

# Drop the "temp_t+1" and "feels_like_t+1" columns as advised
data.drop(columns=['temp_t+1', 'feels_like_t+1'], inplace=True)

# Check the data types and inspect the first few rows
print(data.dtypes)
print(data.head())

###----------------- 2.DATA CLEANSING, EDA AND FEATURE ENGINEERING -----------------###

#####----------------- 2.a.Time Series line plot

data.set_index('date', inplace=True)  # Set 'date' as the index for easier time-based plotting

# Set up the figure and axes for plotting
plt.figure(figsize=(15, 12))

# Plot active power
plt.subplot(5, 1, 1)
plt.plot(data.index, data['active_power'], color='blue', linewidth=1)
plt.title('Active Power Over Time', fontsize=12)
plt.ylabel('Active Power (W)')
plt.xticks([])  # Remove x-ticks for clarity

# Plot current
plt.subplot(5, 1, 2)
plt.plot(data.index, data['current'], color='red', linewidth=1)
plt.title('Current Over Time', fontsize=12)
plt.ylabel('Current (A)')
plt.xticks([])

# Plot voltage
plt.subplot(5, 1, 3)
plt.plot(data.index, data['voltage'], color='brown', linewidth=1)
plt.title('Voltage Over Time', fontsize=12)
plt.ylabel('Voltage (V)')
plt.xticks([])

# Plot apparent power
plt.subplot(5, 1, 4)
plt.plot(data.index, data['apparent_power'], color='purple', linewidth=1)
plt.title('Apparent Power Over Time', fontsize=12)
plt.ylabel('Apparent Power (VA)')
plt.xticks([])

# Plot temperature
plt.subplot(5, 1, 5)
plt.plot(data.index, data['temp'], color='green', linewidth=1)
plt.title('Temperature Over Time', fontsize=12)
plt.ylabel('Temperature (°C)')
plt.xlabel('Date')

# Adjust layout to ensure no overlap and enhance readability
plt.tight_layout()
plt.show()

#####----------------- 2.b. Data Issues

# Check for missing values
missing_summary = data.isnull().sum()

#####----------------- 2.c. Checking for outliers

# Define a function to calculate outliers and additional statistics
def calculate_outlier_statistics(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    negative_outliers = (series < lower_bound).sum()
    positive_outliers = (series > upper_bound).sum()
    total_outliers = negative_outliers + positive_outliers
    total_count = len(series)
    negative_outlier_pct = (negative_outliers / total_count) * 100
    positive_outlier_pct = (positive_outliers / total_count) * 100
    total_outlier_pct = (total_outliers / total_count) * 100
    return {
        'Column': series.name,
        'Median': series.median(),
        'Mean': series.mean(),
        'Max Value': series.max(),
        'Min Value': series.min(),
        'Lower Threshold': lower_bound,
        'Upper Threshold': upper_bound,
        'Negative Outliers': negative_outliers,
        'Negative Outliers (%)': negative_outlier_pct,
        'Positive Outliers': positive_outliers,
        'Positive Outliers (%)': positive_outlier_pct,
        'Total Outliers': total_outliers,
        'Total Outliers (%)': total_outlier_pct,
    }

# Select numerical columns
numerical_cols = data.select_dtypes(include=['float32', 'int32']).columns

# Calculate outliers and statistics for each numerical column
outliers_summary = []

for col in numerical_cols:
    stats = calculate_outlier_statistics(data[col])
    stats['Column'] = col
    outliers_summary.append(stats)

# Convert to DataFrame for display
outliers_df = pd.DataFrame(outliers_summary)

# Display the updated table
print("Detailed Outliers Summary Table:")
print(outliers_df)

#####----------------- 2.d. Create new columns

# Assuming 'data' has a datetime index or a 'date' column in datetime format
data['year'] = data.index.year  # Extract year

# Define seasons by mapping months to seasons
def get_season(month):
    if month in [12, 1, 2]:
        return 1  # Winter
    elif month in [3, 4, 5]:
        return 2  # Spring
    elif month in [6, 7, 8]:
        return 3  # Summer
    elif month in [9, 10, 11]:
        return 4  # Autumn

# Apply the function to create the season column
data['season'] = data.index.month.map(get_season)

data['month'] = data.index.month  # Extract month (1 = January, ..., 12 = December)

# Extract weekday (1 = Monday, ..., 7 = Sunday)
data['weekday'] = data.index.weekday + 1

# Create binary weekend column (0 = weekday, 1 = weekend)
data['weekend'] = data['weekday'].apply(lambda x: 1 if x >= 6 else 0)

# Additional suggested features
data['hour'] = data.index.hour  # Hour of the day
data['is_nighttime'] = data['hour'].apply(lambda x: 1 if (x <= 8) else 0)
data['day_of_year'] = data.index.dayofyear  # Day of the year

#####----------------- 2.d. Average active power plots

# Group data and calculate the average active power for each time period
avg_active_power_per_year = data.groupby('year')['active_power'].mean()
avg_active_power_per_season = data.groupby('season')['active_power'].mean()
avg_active_power_per_month = data.groupby('month')['active_power'].mean()
avg_active_power_per_weekday = data.groupby('weekday')['active_power'].mean()

# Set up the plotting area
plt.figure(figsize=(14, 10))

# Plot average active power per year
plt.subplot(2, 2, 1)
avg_active_power_per_year.plot(marker='o', color='b')
plt.title('Average Active Power per Year')
plt.xlabel('Year')
plt.ylabel('Average Active Power (W)')

# Plot average active power per season
plt.subplot(2, 2, 2)
avg_active_power_per_season.plot(marker='o', color='g')
plt.title('Average Active Power per Season')
plt.xlabel('Season (1=Winter, 2=Spring, 3=Summer, 4=Autumn)')
plt.ylabel('Average Active Power (W)')

# Plot average active power per month
plt.subplot(2, 2, 3)
avg_active_power_per_month.plot(marker='o', color='purple')
plt.title('Average Active Power per Month')
plt.xlabel('Month')
plt.ylabel('Average Active Power (W)')

# Plot average active power per weekday
plt.subplot(2, 2, 4)
avg_active_power_per_weekday.plot(marker='o', color='orange')
plt.title('Average Active Power per Weekday')
plt.xlabel('Weekday (1=Monday, ..., 7=Sunday)')
plt.ylabel('Average Active Power (W)')

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()

# Drop the 'weekend' column
data = data.drop(columns=['weekend'])



# Calculate the average active power for each hour of the day
hourly_avg_power = data.groupby('hour')['active_power'].mean()

# Plot the average active power per hour
plt.figure(figsize=(10, 6))
plt.plot(hourly_avg_power.index, hourly_avg_power, marker='o', color='teal')
plt.title('Average Active Power per Hour of the Day')
plt.xlabel('Hour of the Day')
plt.ylabel('Average Active Power (W)')
plt.xticks(range(0, 24))  # Ensure all hours are shown on the x-axis
plt.grid(True)
plt.tight_layout()
plt.show()



# Calculate the average active power for nighttime (is_nighttime = 1) and daytime (is_nighttime = 0)
avg_power_night_day = data.groupby('is_nighttime')['active_power'].mean()

# Plotting the bar chart
plt.figure(figsize=(8, 6))
avg_power_night_day.plot(kind='bar', color=['skyblue', 'salmon'], legend=False)
plt.title('Average Active Power: Daytime vs Nighttime')
plt.xlabel('Is Nighttime (0 = Daytime, 1 = Nighttime)')
plt.ylabel('Average Active Power (W)')
plt.xticks([0, 1], labels=['Daytime', 'Nighttime'], rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


#####----------------- 2.e. ACF and PACF
# Plot ACF and PACF for active_power
plt.figure(figsize=(14, 6))

# ACF plot
plt.subplot(1, 2, 1)
plot_acf(data['active_power'].dropna(), lags=50, ax=plt.gca())
plt.title('Autocorrelation Function (ACF) for Active Power')

# PACF plot
plt.subplot(1, 2, 2)
plot_pacf(data['active_power'].dropna(), lags=50, ax=plt.gca(), method='ywm')
plt.title('Partial Autocorrelation Function (PACF) for Active Power')

plt.tight_layout()
plt.show()

# Lag 1 and Lag 2 based on PACF
data['active_power_lag_1'] = data['active_power'].shift(1)
data['active_power_lag_2'] = data['active_power'].shift(2)

# 20-minute rolling average based on ACF
data['active_power_rolling_20'] = data['active_power'].rolling(window=20).mean()
# 20-minute rolling standard deviation (new feature to capture variability)
data['active_power_rolling_std_20'] = data['active_power'].rolling(window=20).std()

data['active_power_rolling_mean_60'] = data['active_power'].rolling(window=60).mean()
data['active_power_rolling_std_60'] = data['active_power'].rolling(window=60).std()

#####----------------- 2.e. Shift t+1 and drop cols
# Shift the target variable to represent t+1
data['active_power_target'] = data['active_power'].shift(-1)

# Drop rows with NaNs
# NaNs could appear due to the shift in target or lagged/rolling features created previously
data = data.dropna()

# Drop unnecessary columns
data = data.drop(columns=['day_of_year', 'temp_min', 'temp_max', 'feels_like', 'active_power'])


#####----------------- 2.g. Corrplot
# Convert selected columns to categorical (character) type
data['year'] = data['year'].astype(str)
data['season'] = data['season'].astype(str)
data['month'] = data['month'].astype(str)
data['weekday'] = data['weekday'].astype(str)
data['hour'] = data['hour'].astype(str)

# Generate the correlation plot for numeric columns only
numeric_cols = data.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns
corr_matrix = data[numeric_cols].corr()

# Create a mask for the upper triangle
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

# Plot the new correlation heatmap with the mask applied
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", square=True, 
            cbar_kws={'shrink': .8}, linewidths=.5)
plt.title("Updated Correlation Heatmap for Remaining Numeric Features at t and Target at t+1 (Lower Triangle Only)")
plt.show()

# Drop the 'apparent_power' and 'power_factor' columns due to high multicollinearity
data = data.drop(columns=['apparent_power', 'power_factor'])


#####----------------- 2.h. Weather cols

# Get unique combinations of 'main' and 'description' columns
unique_combinations = data[['main', 'description']].drop_duplicates().reset_index(drop=True)

# Display the unique combinations table
print(unique_combinations)

# Drop the 'description' column as it provides redundant information
data = data.drop(columns=['description'])

###----------------- 3.MODELING -----------------###

#####----------------- 3.a. Convert to dummies

# Convert categorical features to dummy variables, including 'main'
data = pd.get_dummies(data, columns=['year', 'season', 'month', 'weekday', 'hour', 'main'], drop_first=True)

# Display the first few rows to confirm the changes
print(data.head())

#####----------------- 3.b. Train Test split and scaling

# Define features (X) and target (y)
X = data.drop(columns=['active_power_target'])
y = data['active_power_target']

# Split data into 80% train and 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Step 1: Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Save the scaler for later use
joblib.dump(scaler, 'scaler.joblib')

#####----------------- 3.c. Regression models evaluation

# Define the models to evaluate
models = {
    #"CatBoost": CatBoostRegressor(verbose=0),
    "RandomForest": RandomForestRegressor(),
    "GradientBoosting": GradientBoostingRegressor(),
    "XGBoost": XGBRegressor(objective='reg:squarederror', eval_metric='mae'),
    "LightGBM": LGBMRegressor(),
    "AdaBoost": AdaBoostRegressor(),
    "ElasticNet": ElasticNet(),
    "Bagging": BaggingRegressor(base_estimator=RandomForestRegressor())
}

# Create a DataFrame to store results
results = pd.DataFrame(columns=["Model", "R2", "Adjusted R2", "RMSE", "MAE"])

# Define a function to calculate adjusted R2
def adjusted_r2(r2, n, p):
    return 1 - ((1 - r2) * (n - 1) / (n - p - 1))

# Evaluate each model
for name, model in models.items():
    # Fit model on training data
    model.fit(X_train_scaled, y_train)
    
    # Predict on test data
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    adj_r2 = adjusted_r2(r2, len(y_test), X_test_scaled.shape[1])
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    # Append results to DataFrame
    results = results.append({
        "Model": name,
        "R2": r2,
        "Adjusted R2": adj_r2,
        "RMSE": rmse,
        "MAE": mae
    }, ignore_index=True)

# Sort results by Adjusted R2 and display
results = results.sort_values(by="Adjusted R2", ascending=False)
print(results)


#####----------------- 3.d. Hyperparameters optimization

# Define the refined parameter grid for faster, efficient tuning
param_grid = {
    'num_leaves': [25, 31],
    'max_depth': [5, 10, 15],
    'learning_rate': [0.05, 0.1],
    'n_estimators': [100, 200],
    'subsample': [0.6, 0.8],
    'colsample_bytree': [0.6, 0.8]
}

# Initialize the LightGBM regressor
model = LGBMRegressor()

# Set up TimeSeriesSplit for cross-validation
tscv = TimeSeriesSplit(n_splits=3)


# Initialize GridSearchCV with TimeSeriesSplit as the cross-validator
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=tscv,  # Use TimeSeriesSplit
    scoring= 'neg_mean_squared_error',
    n_jobs=-1,
    verbose=1
)

# Fit GridSearchCV on the training data
grid_search.fit(X_train_scaled, y_train)

# Retrieve the best parameters and model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_


# Calculate RMSE from the negative MSE score
best_mse = -grid_search.best_score_  # Convert to positive MSE
best_rmse = np.sqrt(best_mse)        # Take square root to get RMSE
print("Best Model Performance (RMSE):", best_rmse)

# Save the best parameters to a JSON file
with open('best_lightgbm_params.json', 'w') as file:
    json.dump(best_params, file)
    
    
# Retrain the model on the full training data with the best parameters
final_model = LGBMRegressor(**best_params)
final_model.fit(X_train_scaled, y_train)

# Evaluate on the test set
y_test_pred = final_model.predict(X_test_scaled)

rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
mae = mean_absolute_error(y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred)

print("Final Model Test RMSE:", rmse)
print("Final Model Test MAE:", mae)
print("Final Model Test R²:", r2)
    
# Save the final model to a file
joblib.dump(final_model, 'final_lightgbm_model.joblib')


#####----------------- 3.e. Real vs Predicted values

# Plot the actual vs. predicted values over time
plt.figure(figsize=(14, 7))
plt.plot(y_test.index, y_test, label='Actual Values', color='blue', alpha=0.7)
plt.plot(y_test.index, y_test_pred, label='Predicted Values', color='red', alpha=0.6)
plt.xlabel("Time")
plt.ylabel("Active Power")
plt.title("Actual vs. Predicted Active Power Over Time")
plt.legend()
plt.show()



#####----------------- 3.f.Features Importance

# Retrieve feature importances and filter out zero importance features
feature_importances = final_model.feature_importances_
features = X_train.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})

# Filter to include only features with Importance > 0
importance_df = importance_df[importance_df['Importance'] > 10].sort_values(by='Importance', ascending=False)

# Plot the feature importances with Importance > 0
plt.figure(figsize=(12, 8))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.gca().invert_yaxis()  # Invert y-axis to have the most important feature at the top
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Feature Importance for LightGBM Model (Non-Zero Importance Only)")
plt.show()


#####----------------- 3.g. Train the best model on the full dataset
scaler = joblib.load('scaler.joblib')  
X_full_scaled = scaler.transform(X)             # Transform the full feature set

# Load the best parameters from the saved JSON file
with open('best_lightgbm_params.json', 'r') as file:
    best_params = json.load(file)

# Initialize the LightGBM model with the best parameters
final_model = LGBMRegressor(**best_params)

# Train the model on the full scaled dataset
final_model.fit(X_full_scaled, y)

# Save the final model for future use
joblib.dump(final_model, 'final_lightgbm_model_full.joblib')

# Evaluate the model on the training data (optional, for verification)
y_full_pred = final_model.predict(X_full_scaled)
rmse = np.sqrt(mean_squared_error(y, y_full_pred))
mae = mean_absolute_error(y, y_full_pred)
r2 = r2_score(y, y_full_pred)

print("Final Model on Full Data - RMSE:", rmse)
print("Final Model on Full Data - MAE:", mae)
print("Final Model on Full Data - R²:", r2)



###----------------- 4.LSTM -----------------###


# Define the number of time steps (look-back period)
time_steps = 3 # Adjust as needed for your look-back period

# Find the index of 'active_power_target' in the DataFrame
target_column_index = data.columns.get_loc('active_power_target')

# Adjusted function to create sequences with correct alignment
def create_sequences_corrected(data, time_steps=1, target_idx=None):
    Xs, ys = [], []
    for i in range(time_steps, len(data)):
        Xs.append(data[i - time_steps:i, :target_idx])  # Use all columns up to the target column index
        ys.append(data[i, target_idx])  # Use the specific target column index
    return np.array(Xs), np.array(ys)

# Convert the DataFrame to a NumPy array
data_values = data.values  # Make sure 'data' has both features and 'active_power_target'

# Create sequences with the correct alignment for the target variable
X_seq, y_seq = create_sequences_corrected(data_values, time_steps=time_steps, target_idx=target_column_index)

# Split the sequences into training and test sets
split_idx = int(0.8 * len(X_seq))
X_train_seq, X_test_seq = X_seq[:split_idx], X_seq[split_idx:]
y_train_seq, y_test_seq = y_seq[:split_idx], y_seq[split_idx:]

# Check shapes to confirm correct alignment
print("Training sequence shape:", X_train_seq.shape)  # Should be (samples, timesteps, features)
print("Training target shape:", y_train_seq.shape)
print("Test sequence shape:", X_test_seq.shape)
print("Test target shape:", y_test_seq.shape)

# Define the LSTM model architecture
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model with early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(
    X_train_seq, y_train_seq,
    epochs=30,
    batch_size=64,
    validation_data=(X_test_seq, y_test_seq),
    callbacks=[early_stopping],
    verbose=1
)

# Predict on the test set
y_test_pred_lstm = model.predict(X_test_seq)

# Evaluate the model performance
rmse_lstm = np.sqrt(mean_squared_error(y_test_seq, y_test_pred_lstm))
mae_lstm = mean_absolute_error(y_test_seq, y_test_pred_lstm)
r2_lstm = r2_score(y_test_seq, y_test_pred_lstm)

print("LSTM Model Test RMSE:", rmse_lstm)
print("LSTM Model Test MAE:", mae_lstm)
print("LSTM Model Test R²:", r2_lstm)

# Plot the actual vs. predicted values over time for the LSTM model
plt.figure(figsize=(14, 7))

# Plot actual values
plt.plot(range(len(y_test_seq)), y_test_seq, label='Actual Values', color='blue', alpha=0.7)

# Plot predicted values
plt.plot(range(len(y_test_seq)), y_test_pred_lstm, label='LSTM Predicted Values', color='red', alpha=0.6)

# Add labels and title
plt.xlabel("Time (Test Samples)")
plt.ylabel("Active Power")
plt.title("Actual vs. Predicted Active Power Over Time (LSTM)")
plt.legend()
plt.show()




###----------------- 5.Prophet -----------------###

# Set up a grid for hyperparameter tuning
param_grid = {
    'seasonality_mode': ['additive'],
    'changepoint_prior_scale': [0.01, 0.05],
    'seasonality_prior_scale': [1, 5],
    'daily_seasonality': [True],  # Use daily seasonality or not
    'weekly_seasonality':[True] 
}

# Track the best model and metrics
best_rmse = float('inf')
best_params = None
best_forecast = None

# Convert scaled arrays back to DataFrames with the original feature names
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# Prepare training and test DataFrames for Prophet
prophet_df_train = pd.DataFrame({
    'ds': y_train.index,  # Time index for training
    'y': y_train.values   # Target values for training
})
for col in X_train_scaled_df.columns:
    prophet_df_train[col] = X_train_scaled_df[col].values

prophet_df_test = pd.DataFrame({
    'ds': y_test.index    # Time index for testing
})
for col in X_test_scaled_df.columns:
    prophet_df_test[col] = X_test_scaled_df[col].values

# Perform grid search
for params in ParameterGrid(param_grid):
    model = Prophet(
        seasonality_mode=params['seasonality_mode'],
        changepoint_prior_scale=params['changepoint_prior_scale'],
        seasonality_prior_scale=params['seasonality_prior_scale'],
        daily_seasonality=params['daily_seasonality'],
        weekly_seasonality=params['weekly_seasonality'],
        
    )
    
    # Add each exogenous variable as a regressor
    for col in X_train_scaled_df.columns:
        model.add_regressor(col)
    
    # Fit the model on training data
    model.fit(prophet_df_train)
    
    # Make predictions on the test set
    forecast = model.predict(prophet_df_test)
    y_test_pred_prophet = forecast['yhat'].values

    # Evaluate the model performance
    rmse_prophet = np.sqrt(mean_squared_error(y_test, y_test_pred_prophet))
    
    # Update best model if current model is better
    if rmse_prophet < best_rmse:
        best_rmse = rmse_prophet
        best_params = params
        best_forecast = y_test_pred_prophet

# Display the best parameters and performance
print("Best Parameters:", best_params)
print("Best Prophet Model Test RMSE:", best_rmse)

# Calculate final evaluation metrics for the best model
mae_prophet = mean_absolute_error(y_test, best_forecast)
r2_prophet = r2_score(y_test, best_forecast)

print("Best Prophet Model Test MAE:", mae_prophet)
print("Best Prophet Model Test R²:", r2_prophet)


plt.figure(figsize=(14, 7))
plt.plot(y_test.index, y_test, label='Actual Values', color='blue', alpha=0.7)
plt.plot(y_test.index, best_forecast, label='Best Prophet Predicted Values', color='red', alpha=0.6)
plt.xlabel("Time")
plt.ylabel("Active Power")
plt.title("Actual vs. Predicted Active Power Over Time (Best Prophet)")
plt.legend()
plt.show()