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
        regex=r'^[A-Za-z]+$',
        help_text=_("5 characters or fewer is required"),
        error_messages={'required': _("Entry only accepts letters"),}
    )
    SEC_Choice = forms.ChoiceField(choices=SEC_FORM_CHOICES, label=_("SEC/Financials Choice:"))

class excel_export(forms.Form):
    ticker = forms.RegexField(label=_("Stock Ticker:"), max_length=5,
        regex=r'^[A-Za-z]+$',
        help_text=_("5 characters or fewer is required"),
        error_messages={'required': _("Entry only accepts letters"),}
    )
    competetor1 = forms.RegexField(label=_("Competator 1:"), max_length=5,
        regex=r'^[A-Za-z]+$',
        help_text=_("5 characters or fewer is required"),
        error_messages={'required': _("Entry only accepts letters"),}
    )
    competetor2 = forms.RegexField(label=_("Competator 2:"), max_length=5,
        regex=r'^[A-Za-z]+$',
        help_text=_("5 characters or fewer is required"),
        error_messages={'required': _("Entry only accepts letters"),}
    )
    competetor3 = forms.RegexField(label=_("Competator 3:"), max_length=5,
        regex=r'^[A-Za-z]+$',
        help_text=_("5 characters or fewer is required"),
        error_messages={'required': _("Entry only accepts letters"),}
    )