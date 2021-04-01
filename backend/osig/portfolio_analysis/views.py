from django.shortcuts import render

from .MLUtils import Tester as T
from .MLUtils import Portfolio as P
from .MLUtils.Trainer import *
# Create your views here.
from django.http import HttpResponse
from django.template import Context, loader
from django.http import JsonResponse

from rest_framework.response import Response
from rest_framework.decorators import api_view
from rest_framework import status
from django.views.decorators.csrf import csrf_exempt
from rest_framework.parsers import JSONParser

import pyEX as px
import json

IEX_TOKEN = "Tpk_647cd93d6c5842d6978e55c6f79b0e1a"
client = px.Client(IEX_TOKEN, version="sandbox")

user_environment = None

def index(request):
    return render(request, 'portfolio_analysis/index.html')

@api_view(['POST'])
def get_json(request):
  
  body_unicode = request.body.decode('utf-8')
  body = json.loads(body_unicode)
  print(body) 
  tickers = body['tickers'].split(',')
  jobs = body['checked']
  try:
    p = P.Portfolio(tickers,client,earnings=False)
  except SymbolError:
    f = open("error.txt","w")
    f.write("{error : \"symbolNotFound\"}")
    sys.exit(-1)

  global user_environment
  user_environment = T.Tester(p,10,60,train_func = train_net)
  
  results = {}
  for job in jobs:
    results[job] = handle(job,user_environment)
  
  
  return JsonResponse(results)

def handle(job,env):
  if job == "pred":
    return env.trainModel()
  elif job == "alphabeta":
    return env.alphabeta([.3333,.3333,.3333,])
