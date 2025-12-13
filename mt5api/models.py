from __future__ import annotations

from django.db import models


class LiveTick(models.Model):
    """Tick intradiario recebido do MT5."""

    ticker = models.CharField(max_length=10, db_index=True)
    timestamp = models.DateTimeField(auto_now_add=True)
    bid = models.FloatField(null=True, blank=True)
    ask = models.FloatField(null=True, blank=True)
    last = models.FloatField()
    as_of = models.DateTimeField(null=True, blank=True, db_index=True)
    source = models.CharField(max_length=16, default="mt5", db_index=True)

    class Meta:
        ordering = ["-timestamp"]
        indexes = [
            models.Index(fields=["ticker", "-timestamp"]),
        ]

    def __str__(self) -> str:
        return f"{self.ticker} @ {self.timestamp:%Y-%m-%d %H:%M:%S} = {self.last}"


class DailyPrice(models.Model):
    """
    DEPRECATED: OHLC diario consolidado a partir dos ticks (MT5) ou fallback Yahoo.
    Preferir cotacoes.QuoteDaily; manter apenas para compatibilidade ate convergir para fonte unica.
    """

    ticker = models.CharField(max_length=10)
    date = models.DateField()
    open = models.FloatField()
    high = models.FloatField()
    low = models.FloatField()
    close = models.FloatField()

    class Meta:
        unique_together = ("ticker", "date")
        ordering = ["-date", "ticker"]
        indexes = [
            models.Index(fields=["ticker", "date"]),
        ]

    def __str__(self) -> str:
        return f"{self.ticker} {self.date} O={self.open} H={self.high} L={self.low} C={self.close}"


class DailyPricePivot(DailyPrice):
    """Proxy para exibir pivot no admin."""

    class Meta:
        proxy = True
        verbose_name = "DailyPrice Pivot"
        verbose_name_plural = "DailyPrice Pivot"
