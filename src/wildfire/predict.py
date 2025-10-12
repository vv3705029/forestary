import pandas as pd
import torch
from src.wildfire.preprocess import preprocess_data, scale_features
from src.wildfire.model import load_model

SCALER_PATH1 = "models/scaler.pkl"

def predict_wildfire(input_data):
    """
    input_data: dict with keys
    ['latitude','longitude','acq_date','avgtemp_c','total_precip_mm',
     'avg_humidity','pressure_in','wind_kph']
     
    Returns dict: latitude, longitude, probability
    """
    df = pd.DataFrame([input_data])
    
    # Preprocess and scale
    df_processed = preprocess_data(df)
    X_scaled, _ = scale_features(df_processed, scaler_path=SCALER_PATH1)
    
    # Convert to tensor
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    
    # Load model
    model = load_model(X_tensor.shape[1])
    with torch.no_grad():
        output = model(X_tensor)
        probs = torch.softmax(output, dim=1).numpy()[0]
        probability = round(float(probs[1])*100, 2)
    
    result = {
        "latitude": input_data['latitude'],
        "longitude": input_data['longitude'],
        "probability": probability
    }
    return result
