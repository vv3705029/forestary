import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

def preprocess_data(df):
    """
    Preprocess the input dataframe.
    For simplicity, we drop 'acq_date' if not used in training.
    """
    if 'acq_date' in df.columns:
        df = df.drop(['acq_date'], axis=1)
    df = df.fillna(df.mean())
    return df

def scale_features(df, scaler_path=None):
    """
    Scale dataframe features using saved scaler.
    """
    if scaler_path:
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        X_scaled = scaler.transform(df)
    else:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df)
    return X_scaled, scaler
