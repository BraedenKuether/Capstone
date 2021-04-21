from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse, HttpResponseRedirect
from django.template import Context, loader
from django.urls import reverse
#from .forms import stock_form

import pyEX as pyx
import json

IEX_TOKEN = "Tpk_647cd93d6c5842d6978e55c6f79b0e1a"
client = pyx.Client(IEX_TOKEN, version="sandbox")

def index(request):
    return render(request, 'stock_research/base_stock_research.html')

def income_statement(request, ticker):
    form = client.incomeStatement(ticker, period='annual', last=4, format='json')
    return render(request, 'stock_research/income_statement.html', {'form': form})

def balance_sheet(request, ticker):
    form = client.balanceSheet(ticker, period='annual', last=4, format='json')
    return render(request, 'stock_research/income_statement.html', {'form': form})

def cash_flows(request, ticker):
    form = client.cashFlows(ticker, period='annual', last=4, format='json')
    return render(request, 'stock_research/income_statement.html', {'form': form})

def financials(request, ticker):
    form = client.financials(ticker)
    return render(request, 'stock_research/income_statement.html', {'form': form})