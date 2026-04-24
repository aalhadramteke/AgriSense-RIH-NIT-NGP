from rest_framework import serializers
from .models import SoilRecord, Prediction

class SoilRecordSerializer(serializers.ModelSerializer):
    class Meta:
        model = SoilRecord
        fields = '__all__'

class PredictionSerializer(serializers.ModelSerializer):
    soil_record = SoilRecordSerializer(read_only=True)
    
    class Meta:
        model = Prediction
        fields = '__all__'
