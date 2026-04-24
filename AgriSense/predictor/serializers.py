from rest_framework import serializers
from .models import SoilRecord, Prediction
import joblib
import numpy as np
from django.conf import settings
import json

MODELS_DIR = settings.MODELS_DIR

class SoilRecordSerializer(serializers.ModelSerializer):
    class Meta:
        model = SoilRecord
        fields = '__all__'

class PredictionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Prediction
        fields = '__all__'

class CropRecommendationSerializer(serializers.Serializer):
    n = serializers.FloatField()
    p = serializers.FloatField()
    k = serializers.FloatField()
    temperature = serializers.FloatField()
    humidity = serializers.FloatField()
    ph = serializers.FloatField()
    rainfall = serializers.FloatField()

    def create_prediction(self, validated_data, yield_est, crop_name, confidence):
        soil_record = SoilRecord.objects.create(**validated_data, location='')
        Prediction.objects.create(
            soil_record=soil_record,
            recommended_crop=crop_name,
            yield_estimate=yield_est,
            confidence_score=confidence
        )
        return {'crop': crop_name, 'confidence': confidence}

class YieldPredictionSerializer(CropRecommendationSerializer):
    pass

# Load models on import
try:
    rf_model = joblib.load(MODELS_DIR / 'crop_model.pkl')
    gb_model = joblib.load(MODELS_DIR / 'yield_model.pkl')
    le = joblib.load(MODELS_DIR / 'label_encoder.pkl')
    with open(MODELS_DIR / 'model_metrics.json') as f:
        metrics = json.load(f)
except FileNotFoundError:
    rf_model = gb_model = le = None
    metrics = {}
