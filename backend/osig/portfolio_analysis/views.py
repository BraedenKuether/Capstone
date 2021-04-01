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

import pyEX as px

IEX_TOKEN = "Tpk_647cd93d6c5842d6978e55c6f79b0e1a"
client = px.Client(IEX_TOKEN, version="sandbox")


def index(request):
    return render(request, 'portfolio_analysis/index.html')

@api_view(['POST'])
def get_json(request):
  tickers = request.query_params['q'].split('$')
  try:
    p = P.Portfolio(tickers,client,earnings=False)
  except SymbolError:
    f = open("error.txt","w")
    f.write("{error : \"symbolNotFound\"}")
    sys.exit(-1)

  ts = T.Tester(p,10,60,train_func = train_net)
  
  return JsonResponse(ts.psRatio(),safe=False)

