import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
import joblib
import json
import os

DATASET_FILE = 'Crop_recommendation.csv'

def generate_synthetic_data(num_samples=2200):
    np.random.seed(42)
    crops = ['rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas',
             'mothbeans', 'mungbean', 'blackgram', 'lentil', 'pomegranate',
             'banana', 'mango', 'grapes', 'watermelon', 'muskmelon', 'apple',
             'orange', 'papaya', 'coconut', 'cotton', 'jute', 'coffee']
    
    data = []
    for _ in range(num_samples):
        crop = np.random.choice(crops)
        N = np.random.uniform(0, 140)
        P = np.random.uniform(5, 145)
        K = np.random.uniform(5, 205)
        temperature = np.random.uniform(8, 43)
        humidity = np.random.uniform(14, 100)
        ph = np.random.uniform(3.5, 9.9)
        rainfall = np.random.uniform(20, 298)
        # Synthetic yield logic based on crop and conditions
        base_yield = np.random.uniform(2000, 8000)
        
        data.append([N, P, K, temperature, humidity, ph, rainfall, crop, base_yield])
        
    df = pd.DataFrame(data, columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'label', 'yield_estimate'])
    df.to_csv(DATASET_FILE, index=False)
    print("Created synthetic dataset.")
    return df

def train_models():
    if not os.path.exists(DATASET_FILE):
        df = generate_synthetic_data()
    else:
        df = pd.read_csv(DATASET_FILE)
        # if there's no yield_estimate in dataset, add a synthetic one for regression
        if 'yield_estimate' not in df.columns:
            df['yield_estimate'] = np.random.uniform(2000, 8000, size=len(df))
            
    X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y_crop = df['label']
    y_yield = df['yield_estimate']

    # Crop Recommendation Model
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_crop, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_c, y_train_c)
    crop_preds = clf.predict(X_test_c)
    acc = accuracy_score(y_test_c, crop_preds)

    # Yield Prediction Model
    X_train_y, X_test_y, y_train_y, y_test_y = train_test_split(X, y_yield, test_size=0.2, random_state=42)
    reg = GradientBoostingRegressor(n_estimators=100, random_state=42)
    reg.fit(X_train_y, y_train_y)
    yield_preds = reg.predict(X_test_y)
    mae = mean_absolute_error(y_test_y, yield_preds)
    r2 = r2_score(y_test_y, yield_preds)

    # Save models
    joblib.dump(clf, 'crop_model.pkl')
    joblib.dump(reg, 'yield_model.pkl')

    # Save metrics
    metrics = {
        'crop_model_accuracy': acc,
        'yield_model_mae': mae,
        'yield_model_r2': r2
    }
    with open('model_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
        
    print("Models trained and saved successfully.")
    print("Metrics:", json.dumps(metrics, indent=2))

if __name__ == "__main__":
    train_models()