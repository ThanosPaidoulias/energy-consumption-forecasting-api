# Energy Consumption Forecasting API

Production-ready machine learning system for predicting hourly active power consumption using LightGBM and weather data, deployed as a FastAPI service.

## 📊 Project Overview

This project develops a time series forecasting model to predict one-step-ahead (t+1) active power consumption in watts. By combining historical energy consumption patterns with real-time weather data from OpenWeather API, the model achieves accurate hourly predictions to support energy management and grid optimization.

**Key Achievement:** Accurate hourly energy forecasting with comprehensive feature engineering pipeline combining temporal patterns, lag features, rolling statistics, and meteorological data.

## 🎯 Business Problem

**Objective:** Predict active power consumption one hour ahead to enable:
- Demand forecasting for energy grid optimization
- Peak load management
- Renewable energy integration planning
- Cost-effective energy procurement

**Approach:** Time series regression combining:
- Historical consumption patterns (lag features, rolling statistics)
- Temporal features (hour, day, season, nighttime indicator)
- Weather conditions (temperature, humidity, pressure, wind, weather type)

## 📁 Repository Structure

```
├── src/
│   ├── train/
│   │   └── Active_Power_Prediction.py    # Complete ML pipeline (EDA, training)
│   └── api/
│       ├── predict.py                     # FastAPI prediction endpoint
│       ├── create_manipulation_pipeline.py # Data preprocessing pipeline
│       └── send_data_to_predict_API.py   # API testing script
│
├── models/
│   ├── final_lightgbm_model_full.joblib  # Trained LightGBM model
│   ├── scaler.joblib                     # Feature scaler
│   ├── data_pipeline.joblib              # Complete preprocessing pipeline
│   └── expected_columns.txt              # Model input schema (63 features)
│
├── data/
│   ├── example_data.json                 # Sample API request format
│   ├── recent_data.csv                   # Test dataset
│   └── energy_weather_raw_data.csv       # Training data (70MB - not in repo)
│
├── docs/
│   ├── Project_Documentation.pdf         # Complete technical documentation
│   └── API_Usage_Guide.md               # API endpoint usage examples
│
├── Dockerfile                            # Container configuration
├── requirements.txt                      # Python dependencies
└── README.md
```

## 🔍 Dataset Features

### Energy Consumption Variables
- **active_power** (target): Active power in watts
- **current**: Electric current in amperes
- **voltage**: Electric tension in volts
- **reactive_power**: Reactive power in volt-amperes
- **apparent_power**: Apparent power in volt-amperes
- **power_factor**: Power factor (dimensionless)

### Weather Variables (OpenWeather API)
- **temp**: Temperature in °C
- **feels_like**: Perceived temperature in °C
- **temp_min / temp_max**: Min/max temperature in °C
- **pressure**: Atmospheric pressure in ATM
- **humidity**: Humidity percentage
- **speed**: Wind speed in km/h
- **deg**: Wind direction in degrees
- **main**: General weather condition (categorical)
- **description**: Detailed weather description (categorical)

### Engineered Features (63 total)
- **Temporal:** year, season, month, weekday, hour, is_nighttime
- **Lag features:** active_power_lag_1, active_power_lag_2
- **Rolling statistics:** 20-hour and 60-hour rolling mean/std
- **One-hot encoded:** Categorical time and weather features

## 🛠️ Technologies Used

**Core ML Stack:**
- **Python 3.x**
- **LightGBM** - Gradient boosting for time series
- **Scikit-learn** - Preprocessing, scaling, model evaluation
- **Pandas & NumPy** - Data manipulation

**Additional Models Evaluated:**
- Random Forest, XGBoost, Gradient Boosting
- LSTM (TensorFlow/Keras)
- Prophet (Facebook)
- ElasticNet, AdaBoost, Bagging

**Deployment:**
- **FastAPI** - REST API framework
- **Uvicorn** - ASGI server
- **Docker** - Containerization
- **Joblib** - Model serialization

**Visualization & Analysis:**
- Matplotlib, Seaborn
- Statsmodels (ACF/PACF plots)

## 📈 Methodology

### 1. Exploratory Data Analysis
- Time series visualization of all power and weather variables
- Correlation analysis between features
- ACF/PACF analysis for lag selection
- Seasonal and hourly pattern identification

### 2. Feature Engineering

**Temporal Features:**
```python
- year, season, month, weekday, hour
- is_nighttime: Binary indicator (1 if hour <= 8, else 0)
```

**Lag Features:**
```python
- active_power_lag_1: Previous hour consumption
- active_power_lag_2: Two hours prior consumption
```

**Rolling Statistics (20-hour and 60-hour windows):**
```python
- active_power_rolling_mean_{20,60}
- active_power_rolling_std_{20,60}
```

**One-Hot Encoding:**
- Categorical time features (year, season, month, weekday, hour)
- Weather condition categories (main weather type)

### 3. Data Preprocessing Pipeline

```python
1. Parse datetime and set as index
2. Feature engineering (temporal + lag + rolling)
3. Shift target variable by 1 step (t+1 prediction)
4. Drop unnecessary columns
5. Handle NaN values from rolling/lag operations
6. One-hot encode categorical features
7. StandardScaler normalization
```

### 4. Model Training & Selection

**Time Series Cross-Validation:**
- TimeSeriesSplit for proper temporal validation
- Prevents data leakage from future to past

**Hyperparameter Tuning:**
- GridSearchCV with custom scoring metrics
- Optimized for MAE, RMSE, and R² score

**Models Compared:**
- LightGBM (selected - best performance)
- XGBoost
- Random Forest
- Gradient Boosting
- LSTM
- Prophet
- Ensemble methods

### 5. Production Deployment

**FastAPI Architecture:**
```
Client Request (JSON)
    ↓
FastAPI /predict endpoint
    ↓
Feature Engineering Transformer
    ↓
StandardScaler
    ↓
LightGBM Model
    ↓
Predictions (JSON response)
```

**Key Components:**
1. **predict.py**: Main API with endpoints
2. **data_pipeline.joblib**: Preprocessing transformations
3. **scaler.joblib**: Fitted StandardScaler
4. **final_lightgbm_model_full.joblib**: Trained model

## 📊 Model Performance

**Evaluation Metrics:**
- **MAE (Mean Absolute Error)**: Average prediction error in watts
- **RMSE (Root Mean Squared Error)**: Penalizes larger errors
- **R² Score**: Proportion of variance explained
- **Cross-Validation Score**: Time series CV performance

**Production Model:** LightGBM Regressor
- Selected after comprehensive comparison with 8+ algorithms
- Optimized hyperparameters via GridSearchCV
- Validated on time series splits

*(Detailed metrics available in `docs/Energy_Consumption_Forecasting_API_Documentation.pdf`)*

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.7+
pip
Docker (optional, for containerized deployment)
```

### Installation

#### Option 1: Local Installation

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/energy-consumption-forecasting-api.git
cd energy-consumption-forecasting-api

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### Option 2: Docker Installation

```bash
# Build Docker image
docker build -t energy-forecasting-api .

# Run container
docker run -p 8000:8000 energy-forecasting-api
```

### Running the API

**Start the FastAPI server:**

```bash
# From project root
python src/api/predict.py
```

The API will be available at: `http://localhost:8000`

**Interactive API Documentation:**
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 📡 API Usage

### Endpoint 1: Get Example Data Structure

**GET** `/example_data`

Returns sample input format for predictions.

```bash
curl http://localhost:8000/example_data
```

**Response:**
```json
{
  "data": [
    {
      "date": "2024-11-03T10:00:00",
      "active_power": 1250.5,
      "current": 5.2,
      "voltage": 240.1,
      ...
    }
  ]
}
```

### Endpoint 2: Make Predictions

**POST** `/predict`

Accepts energy and weather data, returns active power predictions.

**Request Format:**
```json
{
  "data": [
    {
      "date": "2024-11-03T10:00:00",
      "active_power": 1250.5,
      "current": 5.2,
      "voltage": 240.1,
      "reactive_power": 120.3,
      "apparent_power": 1300.0,
      "power_factor": 0.96,
      "main": "Clear",
      "description": "clear sky",
      "temp": 18.5,
      "feels_like": 17.2,
      "temp_min": 16.0,
      "temp_max": 20.0,
      "pressure": 1013,
      "humidity": 65,
      "speed": 3.5,
      "deg": 180
    }
  ]
}
```

**Response:**
```json
{
  "predictions": [1275.3]
}
```

### Testing the API

**Using the provided test script:**

```bash
python src/api/send_data_to_predict_API.py
```

This script:
1. Loads test data from `data/recent_data.csv`
2. Sends POST request to `/predict`
3. Displays predictions

**Using curl:**

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d @data/example_data.json
```

**Using Python requests:**

```python
import requests
import pandas as pd

# Load data
data = pd.read_csv('data/recent_data.csv')

# Prepare request
payload = {"data": data.to_dict(orient="records")}

# Make prediction
response = requests.post("http://localhost:8000/predict", json=payload)
predictions = response.json()
print(predictions)
```

## 🐳 Docker Deployment

**Build and run:**

```bash
# Build image
docker build -t energy-forecasting-api .

# Run container
docker run -d -p 8000:8000 --name energy-api energy-forecasting-api

# Check logs
docker logs energy-api

# Stop container
docker stop energy-api
```

**Access API:** `http://localhost:8000/docs`

## 📚 Training Your Own Model

To retrain the model with new data:

1. **Prepare data:**
   - Format: CSV with required columns (see Dataset Features)
   - Include `date`, energy variables, and weather data
   - Hourly frequency

2. **Run training script:**

```bash
python src/train/Active_Power_Prediction.py
```

This script performs:
- Data loading and cleaning
- EDA and visualization
- Feature engineering
- Model training and hyperparameter tuning
- Model evaluation
- Model serialization (saves to `models/`)

3. **Update API models:**
   - New models saved in `models/` directory
   - Restart API to load new models

## 💡 Key Insights & Findings

### Temporal Patterns
- **Nighttime effect**: Consumption significantly lower during hours 0-8
- **Hourly patterns**: Clear daily cycles with peaks during business hours
- **Seasonal variation**: Different consumption patterns across seasons

### Weather Impact
- **Temperature**: Strong correlation with energy consumption
- **Weather conditions**: Different patterns for Clear/Clouds/Rain/etc.
- **Humidity & Pressure**: Moderate influence on consumption

### Feature Importance
Top predictive features (from LightGBM):
1. Lag features (previous 1-2 hours)
2. Rolling mean statistics (20-hour, 60-hour windows)
3. Current hour of day
4. Temperature
5. Is nighttime indicator

## 🔮 Future Enhancements

**Model Improvements:**
- [ ] Implement multi-step forecasting (predict t+2, t+3, etc.)
- [ ] Add ensemble methods combining multiple models
- [ ] Explore deep learning architectures (Transformer, N-BEATS)
- [ ] Include additional external features (holidays, events)

**Data Enhancements:**
- [ ] Incorporate real-time weather forecast API
- [ ] Add historical holiday/event calendars
- [ ] Include day-ahead electricity pricing data
- [ ] Expand to multi-location forecasting

**Deployment Improvements:**
- [ ] Add model monitoring and drift detection
- [ ] Implement A/B testing framework for model versions
- [ ] Create batch prediction endpoint
- [ ] Add authentication and rate limiting
- [ ] Deploy to cloud (AWS/GCP/Azure)
- [ ] Set up CI/CD pipeline

**Visualization:**
- [ ] Build interactive dashboard (Streamlit/Dash)
- [ ] Real-time prediction visualization
- [ ] Model performance monitoring UI

## 📖 Documentation

- **`docs/Project_Documentation.pdf`**: Complete technical documentation
- **`docs/API_Usage_Guide.md`**: Detailed API usage examples
- **`/docs` endpoint**: Interactive Swagger UI documentation
- **`models/expected_columns.txt`**: Required input features

## 👤 Author

**Thanos Paidoulias**
- Data Scientist specializing in Marketing Analytics & Time Series Forecasting
- GitHub: [@https://github.com/ThanosPaidoulias]
- LinkedIn: [https://www.linkedin.com/in/thanos-paidoulias/]

## 📝 License

This project is open source and available for educational purposes.

## 🏆 Project Highlights

- ✅ Production-ready ML pipeline from EDA to deployment
- ✅ Sophisticated feature engineering (63 features from 16 raw variables)
- ✅ RESTful API with automatic documentation
- ✅ Dockerized for easy deployment
- ✅ Comprehensive preprocessing pipeline
- ✅ Time series best practices (proper CV, lag features)
- ✅ Real-world energy + weather data integration

---

⭐ If you found this project useful, please consider starring the repository!

**Note:** The training dataset (`energy_weather_raw_data.csv`, ~70MB) is not included in the repository. Contact the author for access or use your own energy consumption dataset with similar schema.
