from django.db import models
from django.utils import timezone

class SoilRecord(models.Model):
    n = models.FloatField()
    p = models.FloatField()
    k = models.FloatField()
    ph = models.FloatField()
    temperature = models.FloatField()
    humidity = models.FloatField()
    rainfall = models.FloatField()
    location = models.CharField(max_length=255, blank=True)
    created_at = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return f"Soil N:{self.n} P:{self.p} K:{self.k}"

class Prediction(models.Model):
    soil_record = models.ForeignKey(SoilRecord, on_delete=models.CASCADE)
    recommended_crop = models.CharField(max_length=100)
    yield_estimate = models.FloatField()
    confidence_score = models.FloatField()
    created_at = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return f"Pred {self.recommended_crop} yield:{self.yield_estimate:.0f}"
