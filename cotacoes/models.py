from __future__ import annotations
from django.conf import settings
from django.db import models
from acoes.models import Asset


class QuoteDaily(models.Model):
    asset = models.ForeignKey("acoes.Asset", on_delete=models.CASCADE, related_name="quotes")
    date = models.DateField()
    close = models.FloatField()
    is_provisional = models.BooleanField(default=False)

    class Meta:
        unique_together = (("asset", "date"),)
        indexes = [
            models.Index(fields=["asset", "date"]),
        ]
        ordering = ["-date"]

    def __str__(self):
        return f"{self.asset.ticker} {self.date} = {self.close}"


class MissingQuoteLog(models.Model):
    asset = models.ForeignKey("acoes.Asset", on_delete=models.CASCADE, related_name="missing_logs")
    date = models.DateField(null=True, blank=True)  # opcional (pode logar por ativo/intervalo)
    reason = models.CharField(max_length=200)       # ex.: 'yf_error', 'no_data', 'invalid_ticker'
    detail = models.TextField(blank=True, default="")
    resolved_bool = models.BooleanField(default=False)
    resolved_by = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"[{self.asset.ticker}] {self.reason} {self.date or ''}"


# ---------------------------------------------------------------------
# 🟢 Novo modelo — cotação intradiária (tempo real via Yahoo)
# ---------------------------------------------------------------------
class QuoteLive(models.Model):
    asset = models.OneToOneField("acoes.Asset", on_delete=models.CASCADE, related_name="live_quote")
    price = models.FloatField()
    bid = models.FloatField(null=True, blank=True)
    ask = models.FloatField(null=True, blank=True)
    last = models.FloatField(null=True, blank=True)
    as_of = models.DateTimeField(null=True, blank=True)
    source = models.CharField(max_length=32, default="mt5")
    volume = models.BigIntegerField(null=True, blank=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Cotação Atual"
        verbose_name_plural = "Cotações Atuais"

    def __str__(self):
        return f"{self.asset.ticker}: {self.price:.2f}"
