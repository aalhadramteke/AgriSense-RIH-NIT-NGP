from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.http import JsonResponse
from .serializers import CropRecommendationSerializer, YieldPredictionSerializer, rf_model, gb_model, le, metrics
from .models import SoilRecord, Prediction
import numpy as np
import requests
from django.conf import settings
from django.db.models import F
from django.utils import timezone
from datetime import timedelta

@api_view(['POST'])
def recommend_crop(request):
    if rf_model is None:
        return Response({'error': 'Models not trained. Run train_model.py'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    serializer = CropRecommendationSerializer(data=request.data)
    if serializer.is_valid():
        features = np.array([[serializer.validated_data['N'], serializer.validated_data['P'], serializer.validated_data['K'],
                              serializer.validated_data['temperature'], serializer.validated_data['humidity'],
                              serializer.validated_data['ph'], serializer.validated_data['rainfall']]])
        crop_idx = rf_model.predict(features)[0]
        crop_name = le.inverse_transform([crop_idx])[0]
        confidence = rf_model.predict_proba(features)[0].max()
        
        yield_est = gb_model.predict(features)[0]
        
        pred_data = serializer.create_prediction(serializer.validated_data, yield_est, crop_name, confidence)
        
        return Response({
            'recommended_crop': crop_name,
            'confidence_score': float(confidence),
            'yield_estimate': float(yield_est),
            **pred_data
        })
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@api_view(['POST'])
def predict_yield(request):
    if gb_model is None:
        return Response({'error': 'Models not trained'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    serializer = YieldPredictionSerializer(data=request.data)
    if serializer.is_valid():
        features = np.array([[serializer.validated_data['n'], serializer.validated_data['p'], serializer.validated_data['k'],
                              serializer.validated_data['temperature'], serializer.validated_data['humidity'],
                              serializer.validated_data['ph'], serializer.validated_data['rainfall']]])
        yield_est = gb_model.predict(features)[0]
        return Response({'yield_estimate': float(yield_est)})
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@api_view(['GET'])
def prediction_history(request):
    recent = Prediction.objects.select_related('soil_record').order_by('-created_at')[:20]
    data = [{
        'id': p.id,
        'crop': p.recommended_crop,
        'yield_est': p.yield_estimate,
        'confidence': p.confidence_score,
        'n': p.soil_record.n,
        'p': p.soil_record.p,
        'k': p.soil_record.k,
        'ph': p.soil_record.ph,
        'temp': p.soil_record.temperature,
        'humidity': p.soil_record.humidity,
        'rainfall': p.soil_record.rainfall,
        'created_at': p.created_at.isoformat()
    } for p in recent]
    return Response(data)

@api_view(['GET'])
def get_weather(request):
    lat = request.GET.get('lat')
    lon = request.GET.get('lon')
    if not lat or not lon:
        return Response({'error': 'lat and lon required'}, status=400)
    
    api_key = settings.OPENWEATHER_API_KEY
    if 'your_api_key_here' in api_key:
        return Response({'error': 'Set OPENWEATHER_API_KEY in settings.py'})
    
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    resp = requests.get(url)
    if resp.status_code == 200:
        data = resp.json()
        # Estimate rainfall (not direct, use proxy or 0)
        rainfall = data.get('rain', {}).get('1h', 0) or 0
        return Response({
            'temperature': data['main']['temp'],
            'humidity': data['main']['humidity'],
            'rainfall': rainfall,
            'description': data['weather'][0]['description']
        })
    return Response({'error': 'Weather fetch failed'}, status=500)

def index(request):
    return JsonResponse({'message': 'AgriSense Dashboard - Visit / for HTML'})
