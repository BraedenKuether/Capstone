from django.urls import path
from django.conf import settings
from django.conf.urls.static import static

from . import views

urlpatterns = [
    path('', views.index, name='stock_research'),
    path('ticker_submit', views.ticker_submit, name='ticker_submit'),
    path('excel_workbook', views.excel_workbook, name='excel_workbook'),
] 
#+ static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)