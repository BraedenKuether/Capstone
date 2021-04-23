from django import forms
from django.forms.widgets import SelectDateWidget
from django.utils.translation import ugettext_lazy as _

from time import gmtime, strftime

class stock_form(forms.Form):
    current_year = int(strftime("%Y", gmtime()))
    YEAR_CHOICES = [
        (current_year-1),
        (current_year-2),
        (current_year-3),
        (current_year-4)
    ]

    SEC_FORM_CHOICES = [
        "Income Statement",
        "Balance Sheet",
        "Cash Flows",
        "Key Metrics"
    ]

    ticker = forms.RegexField(label=_("Stock Ticker"), max_length=5,
        regex=r'^[A-Za-z0-9]+$',
        help_text=_("5 characters or fewer is required"),
        error_messages={'required': _("Entry only accepts letters"),}
    )
    display_date = forms.DateField(label=_("Year"),
        widget=SelectDateWidget(years=YEAR_CHOICES),
        help_text=_("Select a year"),
        error_messages={'required': _("Need to enter a year"),}
    )