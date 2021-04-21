from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse, HttpResponseRedirect
from django.template import Context, loader
from django.urls import reverse
import logging
#from .forms import stock_form

import pyEX as pyx
import json

logger = logging.getLogger('stock_research.views.json')

IEX_TOKEN = "Tpk_647cd93d6c5842d6978e55c6f79b0e1a"
client = pyx.Client(IEX_TOKEN, version="sandbox")

def index(request):
    return render(request, 'stock_research/base_stock_research.html')

def income_statement(request, ticker):
    form = client.incomeStatement(ticker, period='annual', last=4, format='json')
    logger.debug("-----------------------LOGGING INCOME STATEMENT JSON-----------------------")
    logger.info(json.dumps(form, sort_keys=True, indent=4))
    return render(request, 'stock_research/income_statement.html', {'form': form})

def balance_sheet(request, ticker):
    form = client.balanceSheet(ticker, period='annual', last=4, format='json')
    logger.debug("-----------------------LOGGING BALANCE SHEET JSON-----------------------")
    logger.info(json.dumps(form, sort_keys=True, indent=4))
    return render(request, 'stock_research/balance_sheet.html', {'form': form})

def cash_flows(request, ticker):
    form = client.cashFlows(ticker, period='annual', last=4, format='json')
    logger.debug("-----------------------LOGGING CASH FLOWS JSON-----------------------")
    logger.info(json.dumps(form, sort_keys=True, indent=4))
    return render(request, 'stock_research/cash_flows.html', {'form': form})

def financials(request, ticker):
    form = client.financials(ticker)
    logger.debug("-----------------------LOGGING FINANCIALS JSON-----------------------")
    logger.info(json.dumps(form, sort_keys=True, indent=4))
    return render(request, 'stock_research/financials.html', {'form': form})