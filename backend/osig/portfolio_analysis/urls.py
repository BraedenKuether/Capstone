from django.urls import path

from . import views

urlpatterns = [
  path('', views.index, name='api/portfolio_analysis'),
  path('get_runs/<str:id>', views.get_run),
  path('create_run',views.create_run),
 ]
