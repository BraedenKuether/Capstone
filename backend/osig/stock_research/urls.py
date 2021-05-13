from django.urls import path
from django.conf import settings
from django.conf.urls.static import static

from . import views

urlpatterns = [
    path('', views.index, name='stock_research'),
    path('ticker_submit', views.ticker_submit, name='ticker_submit'),
] 
#+ static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)