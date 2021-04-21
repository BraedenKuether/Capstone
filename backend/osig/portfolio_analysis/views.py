from django.shortcuts import render

from .MLUtils import Tester as T
from .MLUtils import Portfolio as P
from .MLUtils.Trainer import *
# Create your views here.
from django.http import HttpResponse
from django.template import Context, loader
from django.http import JsonResponse

from django.contrib.auth.decorators import permission_required 
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated

from rest_framework.decorators import api_view


from rest_framework import status
from django.views.decorators.csrf import csrf_exempt
from rest_framework.parsers import JSONParser
from portfolio_analysis.models import AnalysisRun, AnalysisRunSerializer
from django.core.serializers import serialize
from django.contrib.auth.decorators import login_required

import pyEX as px
import json

IEX_TOKEN = "Tpk_647cd93d6c5842d6978e55c6f79b0e1a"
client = px.Client(IEX_TOKEN, version="sandbox")

user_environment = None


def index(request):
  return render(request, 'portfolio_analysis/index.html')
   
@api_view(['GET'])
@login_required
@permission_required('portfolio_analysis.isManager',raise_exception=True)
def get_run(request,id):

  if id == 'all':
    '''
    run = AnalysisRun(title='test',path='runs/1.json')
    run.id = 1
    run2 = AnalysisRun(title='test2',path='runs/2.json')
    run2.id = 2
    serializer = AnalysisRunSerializer(run)
    serializer2 = AnalysisRunSerializer(run2)
    '''
    runs = AnalysisRun.objects.all()
    if len(runs) > 0:
      serializer = json.loads(serialize('json',runs,fields=('id','title','date')))
      data = []
      for run in serializer:
        run_formatted = run['fields']
        run_formatted['id'] = run['pk']
        data.append(run_formatted)
      print(data)
      resp = {'data': data}
    else:
      resp = {'data': []}
  else:
    with open('portfolio_analysis/runs/{}.json'.format(id), 'r') as file:
      resp = json.loads(file.read())
  return JsonResponse(resp)

@api_view(['POST'])
@permission_required('portfolio_analysis.isManager',raise_exception=True)
def create_run(request):
  body_unicode = request.body.decode('utf-8')
  body = json.loads(body_unicode)
  tickers = body['tickers'].split(',')
  title = body['title']
  try:
    p = P.Portfolio(tickers,client,earnings=True)
  except SymbolError:
    f = open("error.txt","w")
    f.write("{error : \"symbolNotFound\"}")
    sys.exit(-1)

  global user_environment
  user_environment = T.Tester(p,10,60,train_func = train_net_earnings)
  n = len(tickers)
  user_environment.setWeights([1/n]*n) 
  results = {}
  #jobs = ["pred", "alphabeta", "cumreturns", "topbottomperf", "totalperf", "ytdperf", "spytd", "portrisk", "sharperatio", "priceearnings", "dividendyield", "priceshares", "plotport"]
  jobs = ["pred", "cumreturns", "topbottomperf", "totalperf", "ytdperf", "spytd", "portrisk", "sharperatio", "priceearnings", "dividendyield", "priceshares", "plotport"]
  results['tickers'] = tickers
  for job in jobs:
    results[job] = handle(job,user_environment)
  
  run = AnalysisRun(title=title,path='')
  run.save()
  id = run.id
  path = 'portfolio_analysis/runs/{}.json'.format(id)
  run.path = path
  run.save()
  with open(run.path, 'w') as file:
    file.write(json.dumps(results))
  return HttpResponse('Run Created')

def handle(job,env):
  if job == "pred":
    return env.trainModel()
  elif job == "alphabeta":
    return env.alphabeta(env.weights)
  
  elif job == 'cumreturns':
    return env.cumulativeReturns(env.weights,withPlot=False)

  elif job == 'topbottomperf':
    return env.topbottom(env.weights)

  elif job == 'totalperf':
    return env.totalPerformance(env.weights)

  elif job == 'ytdperf':
    return env.ytdPerformance(env.weights)

  elif job == 'spytd':
    return env.spYTD()

  elif job == 'portrisk':
    return env.risk(env.weights)

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
