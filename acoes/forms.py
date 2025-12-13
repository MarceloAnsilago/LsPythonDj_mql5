# acoes/forms.py
from django import forms
from .models import Asset

class AssetForm(forms.ModelForm):
    class Meta:
        model = Asset
        fields = ["ticker", "ticker_yf", "name", "is_active", "use_mt5"]  # <-- sem logo_prefix

    def clean_ticker(self):
        return (self.cleaned_data.get("ticker") or "").upper().strip()

    def clean_ticker_yf(self):
        return (self.cleaned_data.get("ticker_yf") or "").upper().strip()
