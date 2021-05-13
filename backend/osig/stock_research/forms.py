from django import forms
from django.forms.widgets import SelectDateWidget
from django.utils.translation import ugettext_lazy as _

SEC_FORM_CHOICES = (
    ('1', 'Income Statement'),
    ('2', 'Balance Sheet'),
    ('3', 'Cash Flows'),
    ('4', 'Key Metrics'),
)

class stock_form(forms.Form):
    ticker = forms.RegexField(label=_("Stock Ticker:"), max_length=5,
        regex=r'^[A-Za-z0-9]+$',
        help_text=_("5 characters or fewer is required"),
        error_messages={'required': _("Entry only accepts letters"),}
    )
    SEC_Choice = forms.ChoiceField(choices=SEC_FORM_CHOICES, label=_("SEC/Financials Choice:"))