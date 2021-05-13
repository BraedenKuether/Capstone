from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse, HttpResponseRedirect, response
from django.template import Context, context, loader
from django.urls import reverse
from django.views.decorators.csrf import csrf_exempt
import logging
from .forms import stock_form

import pyEX as pyx
import json

logger = logging.getLogger('stock_research.views.json')

IEX_TOKEN = "Tpk_647cd93d6c5842d6978e55c6f79b0e1a"
client = pyx.Client(IEX_TOKEN, version="sandbox")

def index(request):
    form = stock_form()
    return render(request, 'stock_research/base_stock_research.html', {'form': form})

@csrf_exempt
def ticker_submit(request):
    if request.method == 'POST':
        form = stock_form(request.POST)
        if form.is_valid():
            ticker = form.cleaned_data['ticker']
            sec_choice = form.cleaned_data['SEC_Choice']
            if sec_choice == '1':
                data = client.incomeStatement(ticker, period='annual', last=4, format='json')
                logger.debug("-----------------------LOGGING INCOME STATEMENT JSON-----------------------")
                logger.info(json.dumps(data, sort_keys=True, indent=4))
                context = {'form':form, 'data': data}
                return render(request, 'stock_research/income_statement.html', context)
            elif sec_choice == '2':
                data = client.balanceSheet(ticker, period='annual', last=4, format='json')
                logger.debug("-----------------------LOGGING BALANCE SHEET JSON-----------------------")
                logger.info(json.dumps(data, sort_keys=True, indent=4))
                context = {'form':form, 'data': data}
                return render(request, 'stock_research/balance_sheet.html', context)
            elif sec_choice == '3':
                data = client.cashFlow(ticker, period='annual', last=4, format='json')
                logger.debug("-----------------------LOGGING CASH FLOWS JSON-----------------------")
                logger.info(json.dumps(data, sort_keys=True, indent=4))
                context = {'form':form, 'data': data}
                return render(request, 'stock_research/balance_sheet.html', context)
            elif sec_choice == '4':
                data = client.advancedStats(ticker, )
                logger.debug("-----------------------LOGGING FINANCIALS JSON-----------------------")
                logger.info(json.dumps(data, sort_keys=True, indent=4))
                context = {'form':form, 'data': data}
                return render(request, 'stock_research/financials.html', context)
    else:
        form = stock_form()
    return render(request, 'stock_research/base_stock_research.html', {'form': form})


'''
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
'''

'''
def ticker_submit(request):
    if request.method == 'POST':
        form = stock_form(request.POST)
        if form.is_valid():
            ticker = form.cleaned_data['ticker'].upper()
            sec_choice = form.cleaned_data['SEC_Choice']
            if sec_choice == "Income Statement:":
                data = client.incomeStatement(ticker, period='annual', last=4, format='json')
                logger.debug("-----------------------LOGGING INCOME STATEMENT JSON-----------------------")
                logger.info(json.dumps(data, sort_keys=True, indent=4))
                context = {'form':form, 'data': data}
                return render(request, 'stock_research/income_statement.html', context)
            elif sec_choice == "Balance Sheet":
                data = client.balanceSheet(ticker, period='annual', last=4, format='json')
                logger.debug("-----------------------LOGGING BALANCE SHEET JSON-----------------------")
                logger.info(json.dumps(data, sort_keys=True, indent=4))
                context = {'form':form, 'data': data}
                return render(request, 'stock_research/balance_sheet.html', context)
            elif sec_choice == "Cash Flows":
                data = client.cashFlow(ticker, period='annual', last=4, format='json')
                logger.debug("-----------------------LOGGING CASH FLOWS JSON-----------------------")
                logger.info(json.dumps(data, sort_keys=True, indent=4))
                context = {'form':form, 'data': data}
                return render(request, 'stock_research/balance_sheet.html', context)
            elif sec_choice == "Key Metrics":
                data = client.advanvcedStats(ticker)
                logger.debug("-----------------------LOGGING FINANCIALS JSON-----------------------")
                logger.info(json.dumps(data, sort_keys=True, indent=4))
                context = {'form':form, 'data': data}
                return render(request, 'stock_research/financials.html', context)
    else:
        form = stock_form()
    return render(request, 'stock_research/base_stock_research.html', {'form': form})
'''
