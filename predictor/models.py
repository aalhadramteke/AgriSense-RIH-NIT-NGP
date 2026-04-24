from django.db import models

class SoilRecord(models.Model):
    n = models.FloatField()
    p = models.FloatField()
    k = models.FloatField()
    ph = models.FloatField()
    temperature = models.FloatField()
    humidity = models.FloatField()
    rainfall = models.FloatField()
    location = models.CharField(max_length=255, blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

class Prediction(models.Model):
    soil_record = models.ForeignKey(SoilRecord, on_delete=models.CASCADE)
    recommended_crop = models.CharField(max_length=100)
    yield_estimate = models.FloatField()
    confidence_score = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)
