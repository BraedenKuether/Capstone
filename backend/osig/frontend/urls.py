from django.urls import path
from . import views


urlpatterns = [
    path('', views.index, name='portfolio_analysis'),
    path('view_run/<str:id>/', views.view_run),
    path('', views.index ),
]