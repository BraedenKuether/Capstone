from django.urls import path
from . import views


urlpatterns = [
    path('view_run/<str:id>/', views.view_run),
    path('', views.index ),
]