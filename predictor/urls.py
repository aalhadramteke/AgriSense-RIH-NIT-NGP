from django.urls import path
from . import views

urlpatterns = [
    path('recommend/', views.recommend_crop, name='recommend_crop'),
    path('predict-yield/', views.predict_yield, name='predict_yield'),
    path('history/', views.prediction_history, name='history'),
    path('weather/', views.get_weather, name='weather'),
    path('retrain/', views.upload_dataset_and_retrain, name='retrain_models'),
    path('clear-dataset/', views.clear_dataset, name='clear_dataset'),
]
