from rest_framework.decorators import api_view, parser_classes
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser
import numpy as np
import pandas as pd
import joblib
import json
import os
import sys
import subprocess
import requests
from django.conf import settings
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from .models import SoilRecord, Prediction
from .serializers import PredictionSerializer

MODEL_PATH = os.path.join(settings.BASE_DIR, 'crop_model.pkl')
YIELD_MODEL_PATH = os.path.join(settings.BASE_DIR, 'yield_model.pkl')

# Load models safely
try:
    crop_model = joblib.load(MODEL_PATH)
except:
    crop_model = None

try:
    yield_model = joblib.load(YIELD_MODEL_PATH)
except:
    yield_model = None

@api_view(['POST'])
def recommend_crop(request):
    try:
        data = request.data
        n = float(data.get('N', 0))
        p = float(data.get('P', 0))
        k = float(data.get('K', 0))
        ph = float(data.get('pH', 0))
        temp = float(data.get('temperature', 0))
        hum = float(data.get('humidity', 0))
        rain = float(data.get('rainfall', 0))
        loc = data.get('location', '')

        features = np.array([[n, p, k, temp, hum, ph, rain]])
        
        if crop_model:
            pred = crop_model.predict(features)[0]
            probs = crop_model.predict_proba(features)[0]
            confidence = float(max(probs)) * 100
            
            classes = crop_model.classes_
            top_3_idx = np.argsort(probs)[-3:][::-1]
            top_3_crops = [{'crop': str(classes[i]).capitalize(), 'prob': float(probs[i]) * 100} for i in top_3_idx]
        else:
            return Response({'error': 'Crop model not loaded'}, status=500)
            
        yield_est = 0
        if yield_model:
            yield_est = yield_model.predict(features)[0]

        soil_record = SoilRecord.objects.create(
            n=n, p=p, k=k, ph=ph, temperature=temp, humidity=hum, rainfall=rain, location=loc
        )
        prediction = Prediction.objects.create(
            soil_record=soil_record,
            recommended_crop=pred,
            yield_estimate=yield_est,
            confidence_score=confidence
        )

        serializer = PredictionSerializer(prediction)

        # Calculate Fertilizer Logic
        optimal_npk = {
            'rice': (80, 40, 40),
            'maize': (120, 60, 40),
            'chickpea': (20, 60, 20),
            'cotton': (120, 60, 60),
            'coffee': (100, 30, 100),
            'default': (100, 50, 50)
        }
        
        crop_name = str(pred).lower()
        opt_n, opt_p, opt_k = optimal_npk.get(crop_name, optimal_npk['default'])
        
        n_deficit = max(0, opt_n - n)
        p_deficit = max(0, opt_p - p)
        k_deficit = max(0, opt_k - k)
        
        dap_kg = p_deficit / 0.46 if p_deficit > 0 else 0
        n_from_dap = dap_kg * 0.18
        remaining_n = max(0, n_deficit - n_from_dap)
        urea_kg = remaining_n / 0.46 if remaining_n > 0 else 0
        mop_kg = k_deficit / 0.60 if k_deficit > 0 else 0
        
        fertilizer_rec = {
            'Urea': round(urea_kg, 1),
            'DAP': round(dap_kg, 1),
            'MOP': round(mop_kg, 1)
        }

        return Response({
            'recommended_crop': pred,
            'confidence_score': confidence,
            'top_3_crops': top_3_crops,
            'fertilizer': fertilizer_rec,
            'full_prediction': serializer.data
        })
    except Exception as e:
        return Response({'error': str(e)}, status=400)

@api_view(['POST'])
def predict_yield(request):
    try:
        data = request.data
        n = float(data.get('N', 0))
        p = float(data.get('P', 0))
        k = float(data.get('K', 0))
        ph = float(data.get('pH', 0))
        temp = float(data.get('temperature', 0))
        hum = float(data.get('humidity', 0))
        rain = float(data.get('rainfall', 0))

        features = np.array([[n, p, k, temp, hum, ph, rain]])
        
        if yield_model:
            yield_est = yield_model.predict(features)[0]
            return Response({
                'yield_estimate': float(yield_est)
            })
        else:
            return Response({'error': 'Yield model not loaded'}, status=500)
    except Exception as e:
        return Response({'error': str(e)}, status=400)

@api_view(['GET'])
def prediction_history(request):
    predictions = Prediction.objects.select_related('soil_record').order_by('-created_at')[:20]
    serializer = PredictionSerializer(predictions, many=True)
    return Response(serializer.data)

@api_view(['GET'])
def get_weather(request):
    lat = request.GET.get('lat')
    lon = request.GET.get('lon')
    if not lat or not lon:
        return Response({'error': 'Please provide lat and lon'}, status=400)
    
    api_key = getattr(settings, 'OPENWEATHER_API_KEY', '')
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    
    try:
        res = requests.get(url)
        data = res.json()
        if res.status_code == 200:
            return Response({
                'temperature': data['main']['temp'],
                'humidity': data['main']['humidity'],
                'rainfall': data.get('rain', {}).get('1h', 0)
            })
        return Response(data, status=res.status_code)
    except Exception as e:
        return Response({'error': str(e)}, status=500)


@api_view(['POST'])
@parser_classes([MultiPartParser])
def upload_dataset_and_retrain(request):
    """
    Accepts a Kaggle CSV dataset and forcefully retrains the global crop and yield models.
    """
    file_obj = request.FILES.get('dataset')
    if not file_obj:
        return Response({"error": "No dataset file provided."}, status=400)
    
    try:
        df = pd.read_csv(file_obj)
        
        # Standardize columns to lowercase for safe mapping
        df.columns = [c.lower().strip() for c in df.columns]
        
        # Expected crop recommendation Kaggle columns
        expected = ['n', 'p', 'k', 'temperature', 'humidity', 'ph', 'rainfall', 'label']
        for c in expected:
            if c not in df.columns:
                return Response({"error": f"Dataset missing required column: '{c}'. Found columns: {list(df.columns)}"}, status=400)
        
        # Extract features and target
        X = df[['n', 'p', 'k', 'ph', 'temperature', 'humidity', 'rainfall']]
        y_crop = df['label']
        
        # Train new crop model
        new_crop_model = RandomForestClassifier(n_estimators=100, random_state=42)
        new_crop_model.fit(X, y_crop)
        
        # Train new yield model (simulate yield if original kaggle dataset lacks it)
        if 'yield' in df.columns:
            y_yield = df['yield']
        else:
            np.random.seed(42)
            # Create synthetic yield based on standard metric distributions
            y_yield = np.random.normal(3000, 500, len(df))
            
        new_yield_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        new_yield_model.fit(X, y_yield)
        
        # Load into Global Memory Scope natively
        global crop_model, yield_model
        crop_model = new_crop_model
        yield_model = new_yield_model
        
        # Persist to disk
        joblib.dump(crop_model, 'crop_model.pkl')
        joblib.dump(yield_model, 'yield_model.pkl')
        
        return Response({
            "message": "Models successfully retrained with uploaded Kaggle dataset!",
            "samples_processed": len(df),
            "classes_identified": list(crop_model.classes_)
        })

    except Exception as e:
        return Response({"error": str(e)}, status=500)

@api_view(['POST'])
def clear_dataset(request):
    """
    Clears custom trained dataset and restores default models.
    """
    try:
        train_script = os.path.join(settings.BASE_DIR, 'train_model.py')
        subprocess.run([sys.executable, train_script], cwd=settings.BASE_DIR, check=True)
        
        global crop_model, yield_model
        crop_model = joblib.load(MODEL_PATH)
        yield_model = joblib.load(YIELD_MODEL_PATH)
        
        return Response({
            "message": "Models successfully restored to default dataset!"
        })
    except Exception as e:
        return Response({"error": str(e)}, status=500)

