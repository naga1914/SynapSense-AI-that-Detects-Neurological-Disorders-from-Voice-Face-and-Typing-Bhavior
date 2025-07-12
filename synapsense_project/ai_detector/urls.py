from django.urls import path
from . import views

urlpatterns = [
    path('', views.upload_detect, name='upload'),
    path('result/<int:pk>/', views.view_result, name='result'),
]