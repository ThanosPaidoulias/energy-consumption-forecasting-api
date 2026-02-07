from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
import pandas as pd
import numpy as np

# Custom transformer to handle feature engineering and apply pre-fitted transformers
class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        
        # Set index to 'date' for time-based calculations
        X.set_index('date', inplace=True)

        # Define the season mapping function
        def get_season(month):
            if month in [12, 1, 2]:
                return 1  # Winter
            elif month in [3, 4, 5]:
                return 2  # Spring
            elif month in [6, 7, 8]:
                return 3  # Summer
            elif month in [9, 10, 11]:
                return 4  # Autumn

        # Apply feature engineering
        X['year'] = X.index.year
        X['season'] = X.index.month.map(get_season)
        X['month'] = X.index.month
        X['weekday'] = X.index.weekday + 1
        X['hour'] = X.index.hour
        X['is_nighttime'] = X['hour'].apply(lambda x: 1 if (x <= 8) else 0)

        # Lag Features and Rolling Statistics
        X['active_power_lag_1'] = X['active_power'].shift(1)
        X['active_power_lag_2'] = X['active_power'].shift(2)
        X['active_power_rolling_20'] = X['active_power'].rolling(window=20).mean()
        X['active_power_rolling_std_20'] = X['active_power'].rolling(window=20).std()
        X['active_power_rolling_mean_60'] = X['active_power'].rolling(window=60).mean()
        X['active_power_rolling_std_60'] = X['active_power'].rolling(window=60).std()

        # Shift target variable by 1 step ahead
        X['active_power_target'] = X['active_power'].shift(-1)
        
        # Drop unnecessary columns and rows with NaN values
        X = X.drop(columns=['active_power', 'apparent_power', 'power_factor', 'temp_min', 'temp_max', 'feels_like', 'description']).dropna()

        # Use pd.get_dummies to create dummy variables
        X = pd.get_dummies(X, columns=['year', 'season', 'month', 'weekday', 'hour', 'main'], drop_first=True)

        # Reset the index to remove the 'date' index, then drop 'date' column before scaling
        X = X.reset_index().drop(columns=['date'])
        
        # Separate target column from the rest
        self.features = X.drop(columns=['active_power_target'])  # Keep only the feature columns
        self.target = X['active_power_target']                   # Save the target column separately
        
        return self.features

class PreFittedScalerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, scaler_path):
        self.scaler = joblib.load(scaler_path)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Apply the pre-fitted scaler to scale the features
        X_scaled = self.scaler.transform(X)
        return X_scaled

# Updated pipeline using pd.get_dummies and handling date appropriately
data_pipeline = Pipeline(steps=[
    ('feature_engineering', FeatureEngineeringTransformer()),
    ('preprocessing', PreFittedScalerTransformer('scaler.joblib'))
])

# Save the updated pipeline
joblib.dump(data_pipeline, 'data_pipeline.joblib')
