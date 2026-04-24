from django.urls import path
from . import views
from django.views.generic import TemplateView

urlpatterns = [
    path('', TemplateView.as_view(template_name='index.html'), name='index'),
    path('api/recommend/', views.recommend_crop, name='recommend'),
    path('api/predict-yield/', views.predict_yield, name='predict_yield'),
    path('api/history/', views.prediction_history, name='history'),
    path('api/weather/', views.get_weather, name='weather'),
]
