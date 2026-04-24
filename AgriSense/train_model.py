import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import LabelEncoder
import joblib
import json
import os

# Download Crop Recommendation Dataset from GitHub (public Kaggle dataset)
url = "https://raw.githubusercontent.com/sujithmandala/Agricultural-Production-Analysis-andPrediction/master/Crop_recommendation.csv"
print("Downloading dataset...")
response = requests.get(url)
with open("crop_recommendation.csv", "wb") as f:
    f.write(response.content)
print("Dataset downloaded.")

# Load and preprocess
df = pd.read_csv("crop_recommendation.csv")
print(df.head())
print(f"Dataset shape: {df.shape}")

X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y_crop = df['label']

# Encode crop labels
le = LabelEncoder()
y_crop_encoded = le.fit_transform(y_crop)

# Synthetic yield target: based on rainfall, NPK balance, etc. (kg/hectare)
df['yield_estimate'] = (
    df['rainfall'] * 0.8 +
    (df['N'] + df['P'] + df['K']) / 15 +
    df['humidity'] * 0.5 +
    np.random.normal(2000, 500, len(df))  # Realistic variation ~1000-4000 kg/ha
)
y_yield = df['yield_estimate']

# Split data
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_crop_encoded, test_size=0.2, random_state=42)
X_train_y, X_test_y, y_train_y, y_test_y = train_test_split(X, y_yield, test_size=0.2, random_state=42)

# Train Crop Recommendation (RF Classifier)
print("Training Random Forest Classifier...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_c, y_train_c)
crop_pred = rf_model.predict(X_test_c)
crop_accuracy = accuracy_score(y_test_c, crop_pred)
print(f"Crop model accuracy: {crop_accuracy:.4f}")

# Train Yield Prediction (GB Regressor)
print("Training Gradient Boosting Regressor...")
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_model.fit(X_train_y, y_train_y)
yield_pred = gb_model.predict(X_test_y)
yield_r2 = r2_score(y_test_y, yield_pred)
print(f"Yield model R²: {yield_r2:.4f}")

# Save models and label encoder
os.makedirs("models", exist_ok=True)
joblib.dump(rf_model, "models/crop_model.pkl")
joblib.dump(gb_model, "models/yield_model.pkl")
joblib.dump(le, "models/label_encoder.pkl")

# Save metrics
metrics = {
    "crop_accuracy": float(crop_accuracy),
    "yield_r2": float(yield_r2),
    "num_crops": len(le.classes_),
    "crops": le.classes_.tolist()
}
with open("models/model_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("Models trained and saved! Accuracy:", crop_accuracy)
print("Metrics:", metrics)
