from django.contrib import admin
from .models import SoilRecord, Prediction

@admin.register(SoilRecord)
class SoilRecordAdmin(admin.ModelAdmin):
    list_display = ('n', 'p', 'k', 'ph', 'temperature', 'created_at')

@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    list_display = ('recommended_crop', 'yield_estimate', 'confidence_score', 'created_at')
