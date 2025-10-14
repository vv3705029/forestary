import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import torch
import torch.nn as nn
from datetime import datetime

# === CONFIG ===
MODEL_PATH = os.getenv("MODEL_PATH", "best_model (1).pth")
SCALER_PATH = os.getenv("SCALER_PATH", "scaler.pkl")

FEATURE_COLS = [
    'avgtemp_c', 'total_precip_mm', 'avg_humidity', 'pressure_in', 'wind_kph',
    'fwi_simple', 'temp_roll3', 'precip_roll3', 'hum_roll3', 'wind_roll3', 'dayofyear', 'hour'
]

# === MODEL DEFINITION ===
class RiskModel(nn.Module):
    def __init__(self, n_features, hidden=128, dropout=0.2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1)
        )
    def forward(self, x):
        return torch.sigmoid(self.fc(x)).squeeze(1)

# === LOAD MODEL & SCALER ===
device = torch.device("cpu")
try:
    model = RiskModel(len(FEATURE_COLS))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
except Exception as e:
    raise RuntimeError(f"Failed to load model from {MODEL_PATH}: {e}")

try:
    scaler_data = joblib.load(SCALER_PATH)
    scaler = scaler_data['scaler']
except Exception as e:
    raise RuntimeError(f"Failed to load scaler from {SCALER_PATH}: {e}")

# === FASTAPI SETUP ===
app = FastAPI(title="Wildfire Risk Prediction API", version="1.0")

# === INPUT MODEL ===
class RiskInput(BaseModel):
    avgtemp_c: float = Field(..., description="Average temperature in Celsius")
    total_precip_mm: float = Field(..., description="Total precipitation in mm")
    avg_humidity: float = Field(..., ge=0, le=100, description="Average humidity (%)")
    pressure_in: float = Field(..., description="Pressure in inches")
    wind_kph: float = Field(..., ge=0, description="Wind speed in km/h")
    acq_date: datetime = Field(..., description="Acquisition date and time in ISO format")

# === UTILITIES ===
def compute_simple_fwi(row):
    # Normalize inputs with typical ranges for better stability
    def normalize(val, min_val, max_val):
        if max_val - min_val == 0:
            return 0.5
        return (val - min_val) / (max_val - min_val)

    avgtemp_n = normalize(row['avgtemp_c'], -30, 50)
    humidity_n = normalize(row['avg_humidity'], 0, 100)
    wind_n = normalize(row['wind_kph'], 0, 100)
    precip_n = normalize(row['total_precip_mm'], 0, 100)

    fwi = 0.45 * avgtemp_n + 0.3 * (1 - humidity_n) + 0.2 * wind_n - 0.05 * precip_n
    return float(np.clip(fwi, 0, 1))

# === PREDICTION ENDPOINT ===
@app.post("/predict", summary="Predict wildfire risk score")
def predict_risk(data: RiskInput):
    try:
        # Prepare dataframe with input data
        df = pd.DataFrame([data.dict()])

        # Compute derived features
        df['fwi_simple'] = df.apply(compute_simple_fwi, axis=1)
        df['dayofyear'] = df['acq_date'].dt.dayofyear
        df['hour'] = df['acq_date'].dt.hour

        # For single record rolling means = values themselves
        df['temp_roll3'] = df['avgtemp_c']
        df['precip_roll3'] = df['total_precip_mm']
        df['hum_roll3'] = df['avg_humidity']
        df['wind_roll3'] = df['wind_kph']

        # Select and scale features
        features = df[FEATURE_COLS]
        features_scaled = scaler.transform(features).astype(np.float32)

        # Model inference
        input_tensor = torch.from_numpy(features_scaled)
        with torch.no_grad():
            pred = model(input_tensor).item()

        return {"risk_score": pred}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

@app.get("/", summary="Welcome message")
def read_root():
    return {"message": "Welcome to the Wildfire Risk Prediction API"}

# === RUN APP ===
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=4000, reload=True)
