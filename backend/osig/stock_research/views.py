from django.shortcuts import render

# Create your views here.
import os
from django.http import HttpResponse, HttpResponseRedirect, response, Http404
from django.template import Context, context, loader
from django.urls import reverse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
import logging
from zipfile import ZipFile
from .forms import stock_form, excel_export
from .SECScraper import createZip

import pyEX as pyx
import json

logger = logging.getLogger('stock_research.views.json')

IEX_TOKEN = "Tpk_647cd93d6c5842d6978e55c6f79b0e1a"
client = pyx.Client(IEX_TOKEN, version="sandbox")

def index(request):
    form1 = stock_form()
    form2 = excel_export()
    return render(request, 'stock_research/base_stock_research.html', {'form1': form1, 'form2': form2})

@csrf_exempt
def ticker_submit(request):
    if request.method == 'POST':
        form1 = stock_form(request.POST)
        form2 = excel_export()
        if form1.is_valid():
            ticker = form1.cleaned_data['ticker']
            sec_choice = form1.cleaned_data['SEC_Choice']
            if sec_choice == '1':
                data = client.incomeStatement(ticker, period='annual', last=4, format='json')
                logger.debug("-----------------------LOGGING INCOME STATEMENT JSON-----------------------")
                logger.info(json.dumps(data, sort_keys=True, indent=4))
                context = {'form1':form1, 'form2': form2, 'data': data}
                return render(request, 'stock_research/income_statement.html', context)
            elif sec_choice == '2':
                data = client.balanceSheet(ticker, period='annual', last=4, format='json')
                logger.debug("-----------------------LOGGING BALANCE SHEET JSON-----------------------")
                logger.info(json.dumps(data, sort_keys=True, indent=4))
                context = {'form1':form1, 'form2': form2, 'data': data}
                return render(request, 'stock_research/balance_sheet.html', context)
            elif sec_choice == '3':
                data = client.cashFlow(ticker, period='annual', last=4, format='json')
                logger.debug("-----------------------LOGGING CASH FLOWS JSON-----------------------")
                logger.info(json.dumps(data, sort_keys=True, indent=4))
                context = {'form1':form1, 'form2': form2, 'data': data}
                return render(request, 'stock_research/balance_sheet.html', context)
            elif sec_choice == '4':
                data = client.advancedStats(ticker)
                logger.debug("-----------------------LOGGING FINANCIALS JSON-----------------------")
                logger.info(json.dumps(data, sort_keys=True, indent=4))
                context = {'form1':form1, 'form2': form2, 'data': data}
                return render(request, 'stock_research/financials.html', context)
    else:
        form1 = stock_form()
        form2 = excel_export()
    return render(request, 'stock_research/base_stock_research.html', {'form1': form1, 'form2': form2})

@csrf_exempt
def excel_workbook(request):
    if request.method == 'POST':
        form1 = stock_form()
        form2 = excel_export(request.POST)
        if form2.is_valid():
            ticker = form2.cleaned_data['ticker']
            data = client.advancedStats(ticker)
            comp1 = form2.cleaned_data['competetor1']
            comp2 = form2.cleaned_data['competetor2']
            comp3 = form2.cleaned_data['competetor3']
            comp4 = form2.cleaned_data['competetor4']
            comp5 = form2.cleaned_data['competetor5']
    else:
        form1 = stock_form()
        form2 = excel_export()

    return render(request, 'stock_research/base_stock_research.html', {'form1': form1, 'form2': form2})

@csrf_exempt
def download(request, path):
    file_path = os.path.join(settings.MEDIA_ROOT, path)
    if os.path.exists(file_path):
        with open(file_path, 'rb') as fh:
            response = HttpResponse(fh.read(), content_type="application/zip")
            response['Content-Disposition'] = 'inline; filename=' + os.path.basename(file_path)
            return response
    raise Http404
