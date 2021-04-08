from django.db import models
import datetime
from django import utils
from rest_framework import serializers

# Create your models here.

class AnalysisRun(models.Model):
  title = models.CharField(max_length=100)
  path = models.CharField(max_length=300)
  date = models.DateField(default=datetime.date.today)
  
class AnalysisRunSerializer(serializers.Serializer):
  id = serializers.IntegerField()
  title = serializers.CharField(max_length=100)
  date = serializers.DateField()