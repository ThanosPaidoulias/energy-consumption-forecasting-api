#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import requests

# Load recent data
data_path = 'data/recent_data.csv'
data = pd.read_csv(data_path)

# Convert data to the required JSON format for FastAPI
data_json = {"data": data.to_dict(orient="records")}

# Save data as an example dataset
#data_path = 'data/example_data.json'
#data.to_json(data_path, orient="records", date_format="iso")

# API endpoint
url = "http://localhost:8000/predict"

# Send POST request
response = requests.post(url, json=data_json)

# Print the response from the API
if response.status_code == 200:
    predictions = response.json()
    print("Predictions:", predictions)
else:
    print("Error:", response.text)
