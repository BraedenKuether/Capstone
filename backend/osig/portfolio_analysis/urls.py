from django.urls import path

from . import views

urlpatterns = [path('api/graph/',views.get_json)]
