from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse
from django.template import Context, loader

def index(request):
    return render(request, 'stock_research/stock_research.html')