from django.urls import path

from . import views

urlpatterns = [
  path('get_runs/<str:id>', views.get_run),
  path('create_run',views.create_run),
  ]
