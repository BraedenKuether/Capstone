from backend.osig.stock_research.forms import stock_form
from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse, HttpResponseRedirect
from django.template import Context, loader
from django.urls import reverse
import logging
from .forms import stock_form

import pyEX as pyx
import json

logger = logging.getLogger('stock_research.views.json')

IEX_TOKEN = "Tpk_647cd93d6c5842d6978e55c6f79b0e1a"
client = pyx.Client(IEX_TOKEN, version="sandbox")

def index(request):
    return render(request, 'stock_research/base_stock_research.html')

def ticker_submit(request):
    form = stock_form(request.POST)
    if form.is_valid():
        

def income_statement(request, ticker):
    data = client.incomeStatement(ticker, period='annual', last=4, format='json')
    logger.debug("-----------------------LOGGING INCOME STATEMENT JSON-----------------------")
    logger.info(json.dumps(data, sort_keys=True, indent=4))
    return render(request, 'stock_research/income_statement.html', {'data': data})

def balance_sheet(request, ticker):
    data = client.balanceSheet(ticker, period='annual', last=4, format='json')
    logger.debug("-----------------------LOGGING BALANCE SHEET JSON-----------------------")
    logger.info(json.dumps(data, sort_keys=True, indent=4))
    return render(request, 'stock_research/balance_sheet.html', {'data': data})

def cash_flows(request, ticker):
    data = client.cashFlow(ticker, period='annual', last=4, format='json')
    logger.debug("-----------------------LOGGING CASH FLOWS JSON-----------------------")
    logger.info(json.dumps(data, sort_keys=True, indent=4))
    return render(request, 'stock_research/cash_flows.html', {'data': data})

def financials(request, ticker):
    data = client.financials(ticker)
    logger.debug("-----------------------LOGGING FINANCIALS JSON-----------------------")
    logger.info(json.dumps(data, sort_keys=True, indent=4))
    return render(request, 'stock_research/financials.html', {'data': data})