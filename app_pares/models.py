from __future__ import annotations

from django.db import models
from acoes.models import Asset
from pairs.models import Pair


class PriceHistory(models.Model):
    """
    DEPRECATED: espelho redundante de preços diários.
    Preferir cotacoes.QuoteDaily como fonte única; mantido apenas para compatibilidade temporária.
    """
    asset = models.ForeignKey(
        Asset,
        on_delete=models.CASCADE,
        related_name="price_history",
    )
    date = models.DateField()
    close = models.DecimalField(max_digits=18, decimal_places=6)
    source = models.CharField(max_length=32, default="QuoteDaily")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = (("asset", "date"),)
        ordering = ["-date"]

    def __str__(self) -> str:
        return f"{self.asset.ticker} {self.date} = {self.close}"


class PairScanResult(models.Model):
    pair = models.ForeignKey(
        Pair,
        on_delete=models.CASCADE,
        related_name="scan_results",
    )
    run_date = models.DateField(db_index=True)
    best_window = models.PositiveIntegerField(null=True, blank=True)
    status = models.CharField(max_length=32)
    message = models.CharField(max_length=200, blank=True, default="")
    rows = models.JSONField(null=True, blank=True)
    best_row = models.JSONField(null=True, blank=True)
    thresholds = models.JSONField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = (("pair", "run_date"),)
        ordering = ["-run_date", "-pair_id"]

    def __str__(self) -> str:
        return f"{self.pair} [{self.run_date}]"
