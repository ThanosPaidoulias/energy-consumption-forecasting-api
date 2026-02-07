import joblib
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import json

# Define file paths
MODEL_PATH = os.path.join('models', 'final_lightgbm_model_full.joblib')
SCALER_PATH = os.path.join('models', 'scaler.joblib')
REQUIRED_COLUMNS = [
    'date', 'active_power', 'current', 'voltage', 'reactive_power', 'apparent_power', 
    'power_factor', 'main', 'description', 'temp', 'feels_like', 'temp_min', 'temp_max', 
    'pressure', 'humidity', 'speed', 'deg'
]
EXPECTED_COLUMNS_PATH = os.path.join('models', 'expected_columns.txt')

# Load model and scaler
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    with open(EXPECTED_COLUMNS_PATH, 'r') as f:
        expected_columns = [line.strip() for line in f.readlines()]
except Exception as e:
    raise Exception(f"Error loading model or scaler: {e}")

def align_features(data: pd.DataFrame, expected_columns: list) -> pd.DataFrame:
    # Add missing columns with default value 0
    for col in expected_columns:
        if col not in data.columns:
            data[col] = 0

    # Remove any extra columns that aren't expected
    data = data[expected_columns]
    return data

# Custom Feature Engineering Transformer
class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        
        # Convert 'date' to datetime format
        if 'date' in X.columns:
            X['date'] = pd.to_datetime(X['date'])
        
        # Set 'date' as the index but drop it from the transformed dataset
        X.set_index('date', inplace=True)
        
        # Define season mapping
        def get_season(month):
            if month in [12, 1, 2]: return 1
            elif month in [3, 4, 5]: return 2
            elif month in [6, 7, 8]: return 3
            elif month in [9, 10, 11]: return 4

        # Feature engineering
        X['year'] = X.index.year
        X['season'] = X.index.month.map(get_season)
        X['month'] = X.index.month
        X['weekday'] = X.index.weekday + 1
        X['hour'] = X.index.hour
        X['is_nighttime'] = X['hour'].apply(lambda x: 1 if x <= 8 else 0)
        X['active_power_lag_1'] = X['active_power'].shift(1)
        X['active_power_lag_2'] = X['active_power'].shift(2)
        X['active_power_rolling_20'] = X['active_power'].rolling(window=20).mean()
        X['active_power_rolling_std_20'] = X['active_power'].rolling(window=20).std()
        X['active_power_rolling_mean_60'] = X['active_power'].rolling(window=60).mean()
        X['active_power_rolling_std_60'] = X['active_power'].rolling(window=60).std()
        X['active_power_target'] = X['active_power'].shift(-1)

        # Drop columns and rows with NaN values
        X = X.drop(columns=['active_power', 'apparent_power', 'power_factor', 'temp_min', 'temp_max', 'feels_like', 'description']).dropna()

        # Convert categorical columns to dummy variables
        X = pd.get_dummies(X, columns=['year', 'season', 'month', 'weekday', 'hour', 'main'], drop_first=True)
        
        # Align features to ensure compatibility with the trained model
        X = align_features(X, expected_columns)
        
        # Reset index to remove the date column from the features
        X = X.reset_index(drop=True)
        
        return X


# Create a processing pipeline using the custom feature engineering transformer and scaler
pipeline = Pipeline(steps=[
    ('feature_engineering', FeatureEngineeringTransformer()),
    ('scaler', scaler)
])

app = FastAPI()



app = FastAPI()

@app.get("/example_data")
def get_example_data():
    file_path = "data/example_data.json"
    # Check if the file exists
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Example data file not found.")
    try:
        with open(file_path, "r") as f:
            example_data = json.load(f)
        return {"data": example_data}
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail="Error reading JSON file.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class PredictionRequest(BaseModel):
    data: List[Dict]

@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        data = pd.json_normalize(request.data)
        
        # Validate required columns
        missing_cols = set(REQUIRED_COLUMNS) - set(data.columns)
        if missing_cols:
            raise HTTPException(status_code=400, detail=f"Missing columns: {missing_cols}")
        
        # Process the data with the pipeline
        processed_data = pipeline.transform(data)
        
        # Generate predictions
        predictions = model.predict(processed_data)
        
        # Return predictions as a list
        return {"predictions": predictions.tolist()}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
