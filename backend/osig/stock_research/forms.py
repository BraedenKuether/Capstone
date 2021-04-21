from django import forms
from django.forms.widgets import SelectDateWidget

from time import gmtime, strftime

class stock_form(forms.Form):
    current_year = int(strftime("%Y", gmtime()))
    YEAR_CHOICES = [
        (current_year),
        (current_year-1),
        (current_year-2),
        (current_year-3)
    ]

    ticker = forms.CharField(label='Stock Ticker', max_length=5)
    display_date = forms.DateField(
        widget=SelectDateWidget(
            empty_label=("Choose Year", "Choose Month", "Choose Day"),
        ),
    )