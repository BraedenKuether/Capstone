from django.urls import path
from django.conf import settings
from django.conf.urls.static import static

from . import views

urlpatterns = [
    path('', views.index, name='stock_research'),
    path('ticker_submit', views.ticker_submit),
    path('income_statement/<str:ticker>', views.income_statement),
    path('balance_sheet/<str:ticker>', views.balance_sheet),
    path('cash_flows/<str:ticker>', views.cash_flows),
    path('financials/<str:ticker>', views.financials)
] 
#+ static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)