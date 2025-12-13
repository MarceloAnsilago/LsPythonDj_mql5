# acoes/models.py
from django.conf import settings
from django.db import models

# Mapeamentos especiais B3 -> Yahoo; None indica ignorar (delistado/sem cotacao).
YF_TICKER_OVERRIDES: dict[str, str | None] = {
    "CPFL3": "CPFE3.SA",
    "CSN3": "CSNA3.SA",
    "CCRO3": "CCRO3",      # sem .SA funciona melhor
    "JBSS3": "JBSS3.BA",   # Yahoo usa cotacao em Buenos Aires
    "BRFS3": None,
    "CRFB3": None,
    "STBP3": None,
}


class Asset(models.Model):
    ticker = models.CharField(max_length=10, unique=True)
    ticker_yf = models.CharField(max_length=20, blank=True, default="")
    name = models.CharField(max_length=100, blank=True, default="")
    is_active = models.BooleanField(default=True)
    use_mt5 = models.BooleanField(default=False)
    logo_prefix = models.CharField(max_length=4, blank=True, default="")

    def __str__(self):
        return self.ticker

    # ---------- SOMENTE NO ASSET ----------
    def _norm_logo_prefix(self, value: str | None) -> str:
        base = (value or self.ticker or "").upper()
        letters = "".join(ch for ch in base if ch.isalpha())  # sÇü letras
        return (letters[:4] or "SEML")

    def _normalize_ticker_yf(self, raw: str | None = None) -> str:
        """
        Normaliza ticker do Yahoo aplicando mapeamentos antes da regra generica .SA.
        Se o ticker estiver marcado como delistado (override None), retorna string vazia.
        """
        base = (self.ticker or "").strip().upper()
        if not base:
            return ""

        if base in YF_TICKER_OVERRIDES:
            mapped = YF_TICKER_OVERRIDES[base]
            return (mapped or "").strip().upper()

        candidate = (raw or self.ticker_yf or base).strip().upper()
        if not candidate:
            return ""
        if "." not in candidate:
            candidate = f"{candidate}.SA"
        return candidate

    @property
    def logo_key(self) -> str:
        return self._norm_logo_prefix(self.logo_prefix)

    def save(self, *args, **kwargs):
        self.logo_prefix = self._norm_logo_prefix(self.logo_prefix)
        self.ticker_yf = self._normalize_ticker_yf(self.ticker_yf)
        return super().save(*args, **kwargs)
    # --------------------------------------


class UserAsset(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    asset = models.ForeignKey(Asset, on_delete=models.CASCADE, related_name="favorites")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=["user", "asset"], name="unique_user_asset")
        ]

    def __str__(self):
        return f"{self.user} ƒT¾ {self.asset}"
