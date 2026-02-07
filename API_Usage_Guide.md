# API Usage Guide

Complete guide for using the Energy Consumption Forecasting API.

## Table of Contents
- [Quick Start](#quick-start)
- [API Endpoints](#api-endpoints)
- [Request/Response Format](#requestresponse-format)
- [Error Handling](#error-handling)
- [Code Examples](#code-examples)
- [Troubleshooting](#troubleshooting)

## Quick Start

### 1. Start the API Server

```bash
python src/api/predict.py
```

The API runs on: `http://localhost:8000`

### 2. Access Interactive Documentation

Open your browser: `http://localhost:8000/docs`

This provides:
- Interactive API testing
- Request/response schemas
- Try-it-out functionality

## API Endpoints

### GET `/example_data`

Returns sample data structure for making predictions.

**Usage:**
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

---

### POST `/predict`

Make energy consumption predictions.

**Required Headers:**
```
Content-Type: application/json
```

**Request Body Schema:**
```json
{
  "data": [
    {
      "date": "string (ISO format)",
      "active_power": "float",
      "current": "float",
      "voltage": "float",
      "reactive_power": "float",
      "apparent_power": "float",
      "power_factor": "float",
      "main": "string (weather category)",
      "description": "string (weather details)",
      "temp": "float",
      "feels_like": "float",
      "temp_min": "float",
      "temp_max": "float",
      "pressure": "integer",
      "humidity": "integer",
      "speed": "float",
      "deg": "integer"
    }
  ]
}
```

**Response:**
```json
{
  "predictions": [1275.3, 1290.8, 1305.2]
}
```

## Request/Response Format

### Required Fields

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `date` | string | Timestamp (ISO 8601) | "2024-11-03T10:00:00" |
| `active_power` | float | Current active power (W) | 1250.5 |
| `current` | float | Electric current (A) | 5.2 |
| `voltage` | float | Voltage (V) | 240.1 |
| `reactive_power` | float | Reactive power (VA) | 120.3 |
| `apparent_power` | float | Apparent power (VA) | 1300.0 |
| `power_factor` | float | Power factor | 0.96 |
| `main` | string | Weather category | "Clear", "Clouds", "Rain" |
| `description` | string | Weather details | "clear sky" |
| `temp` | float | Temperature (°C) | 18.5 |
| `feels_like` | float | Feels like temp (°C) | 17.2 |
| `temp_min` | float | Min temperature (°C) | 16.0 |
| `temp_max` | float | Max temperature (°C) | 20.0 |
| `pressure` | integer | Pressure (ATM) | 1013 |
| `humidity` | integer | Humidity (%) | 65 |
| `speed` | float | Wind speed (km/h) | 3.5 |
| `deg` | integer | Wind direction (°) | 180 |

### Weather Categories (main field)

Valid values:
- `Clear`
- `Clouds`
- `Rain`
- `Drizzle`
- `Thunderstorm`
- `Mist`
- `Fog`
- `Haze`

## Error Handling

### Common Errors

**400 Bad Request - Missing Columns**
```json
{
  "detail": "Missing columns: {'voltage', 'current'}"
}
```
**Solution:** Ensure all required fields are present in request.

---

**500 Internal Server Error**
```json
{
  "detail": "Error message details"
}
```
**Solution:** Check logs for detailed error. Common causes:
- Invalid data types
- NaN values in rolling window features
- Insufficient historical data for lag features

---

**422 Unprocessable Entity**
```json
{
  "detail": [
    {
      "loc": ["body", "data"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```
**Solution:** Request body doesn't match expected schema.

## Code Examples

### Python (requests library)

```python
import requests
import pandas as pd

# Single prediction
data = {
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

response = requests.post("http://localhost:8000/predict", json=data)
print(response.json())
# Output: {"predictions": [1275.3]}
```

**Batch predictions from CSV:**

```python
import requests
import pandas as pd

# Load data
df = pd.read_csv('data/recent_data.csv')

# Convert to required format
payload = {"data": df.to_dict(orient="records")}

# Make predictions
response = requests.post("http://localhost:8000/predict", json=payload)

if response.status_code == 200:
    predictions = response.json()['predictions']
    df['predicted_power'] = predictions
    print(df[['date', 'active_power', 'predicted_power']])
else:
    print(f"Error: {response.status_code}")
    print(response.text)
```

### cURL

**Simple prediction:**

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

**From JSON file:**

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d @data/example_data.json
```

### JavaScript (fetch)

```javascript
const data = {
  data: [
    {
      date: "2024-11-03T10:00:00",
      active_power: 1250.5,
      current: 5.2,
      voltage: 240.1,
      reactive_power: 120.3,
      apparent_power: 1300.0,
      power_factor: 0.96,
      main: "Clear",
      description: "clear sky",
      temp: 18.5,
      feels_like: 17.2,
      temp_min: 16.0,
      temp_max: 20.0,
      pressure: 1013,
      humidity: 65,
      speed: 3.5,
      deg: 180
    }
  ]
};

fetch('http://localhost:8000/predict', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify(data)
})
  .then(response => response.json())
  .then(data => console.log('Predictions:', data.predictions))
  .catch(error => console.error('Error:', error));
```

## Troubleshooting

### Issue: "Missing columns" error

**Cause:** Request is missing required fields.

**Solution:**
1. Check `/example_data` endpoint for complete schema
2. Ensure all 16 required fields are present
3. Verify field names match exactly (case-sensitive)

---

### Issue: Predictions seem incorrect

**Cause:** Model requires sufficient historical context (lag/rolling features).

**Solution:**
- Ensure input data has at least 60 consecutive hourly records
- First ~60 predictions may be less accurate due to warm-up period
- Check that `date` field is properly formatted and sequential

---

### Issue: API returns 500 error

**Cause:** Internal processing error.

**Solution:**
1. Check API logs for detailed error message
2. Verify data types match expected schema
3. Ensure no NaN/null values in critical fields
4. Test with `/example_data` format first

---

### Issue: Docker container won't start

**Cause:** Port conflict or image build error.

**Solution:**
```bash
# Check if port 8000 is in use
lsof -i :8000

# Use different port
docker run -p 8080:8000 energy-forecasting-api

# Rebuild image
docker build --no-cache -t energy-forecasting-api .
```

## Advanced Usage

### Streaming Predictions

For real-time streaming data:

```python
import requests
import time

def stream_predictions(data_stream):
    """Make predictions on streaming data"""
    for batch in data_stream:
        payload = {"data": batch}
        response = requests.post(
            "http://localhost:8000/predict",
            json=payload,
            timeout=5
        )
        if response.status_code == 200:
            yield response.json()['predictions']
        else:
            print(f"Error: {response.status_code}")
```

### Health Check

Monitor API availability:

```python
import requests

def check_api_health():
    try:
        response = requests.get("http://localhost:8000/example_data")
        return response.status_code == 200
    except:
        return False

if check_api_health():
    print("API is running")
else:
    print("API is down")
```

## Performance Tips

1. **Batch predictions**: Send multiple records in one request for better throughput
2. **Connection pooling**: Reuse HTTP connections for multiple requests
3. **Error handling**: Implement retry logic with exponential backoff
4. **Monitoring**: Track response times and error rates

## Support

For issues or questions:
1. Check `/docs` endpoint for interactive documentation
2. Review `docs/Project_Documentation.pdf`
3. Open an issue on GitHub
4. Contact: [Your contact info]

---

**Last Updated:** February 2026
