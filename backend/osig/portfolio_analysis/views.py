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
from rest_framework.decorators import authentication_classes 
from rest_framework.decorators import permission_classes


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
#@authentication_classes([BasicAuthentication])
#@permission_classes([IsAuthenticated])
def get_json(request):
  body_unicode = request.body.decode('utf-8')
  body = json.loads(body_unicode)
  tickers = body['tickers'].split(',')
  jobs = body['checked']
  try:
    p = P.Portfolio(tickers,client,earnings=True)
  except SymbolError:
    f = open("error.txt","w")
    f.write("{error : \"symbolNotFound\"}")
    sys.exit(-1)

  global user_environment
  user_environment = T.Tester(p,10,60,train_func = train_net)
  n = len(tickers)
  user_environement.setWeights([1/n]*len(n)) 
  results = {}
  for job in jobs:
    results[job] = handle(job,user_environment)
  
  print(results)
  return JsonResponse(results)

def handle(job,env):
  if job == "pred":
    return env.trainModel()
  
  elif job == "alphabeta":
    return env.alphabeta(env.weights)
  
  elif job == 'cumreturns':
    return env.cumualativeReturns(env.weights,withPlot=False)

  elif job == 'topbottomperf':
    return env.topbottom(env.weights)

  elif job == 'totalperf':
    return env.totalPerformance(env.weights)

  elif job == 'ytdperf':
    return env.ytdPerformance(env.weights)

  elif job == 'spytd':
    return env.spYTD()

  elif job == 'portrisk':
    return env.risk()

  elif job == 'sharperatio':
    return env.sharpe(env.weights)

  elif job == 'priceearnings':
    return env.peRatio(withPlot=False)

  elif job == 'dividendyield':
    return env.dividendYield(withPlot=False)

  elif job == 'priceshares':
    return env.psRatio(withPlot=False)

  elif job == 'plotport':
    return env.plotPortfolio()
  
  else:
    return "something is fucked"
