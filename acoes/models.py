# acoes/models.py
from django.conf import settings
from django.db import models

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
        letters = "".join(ch for ch in base if ch.isalpha())  # só letras
        return (letters[:4] or "SEML")

    @property
    def logo_key(self) -> str:
        return self._norm_logo_prefix(self.logo_prefix)

    def save(self, *args, **kwargs):
        self.logo_prefix = self._norm_logo_prefix(self.logo_prefix)
        # completa YF com .SA se faltar
        if self.ticker and not self.ticker_yf:
            self.ticker_yf = self.ticker.upper()
        if self.ticker_yf and "." not in self.ticker_yf:
            self.ticker_yf = (self.ticker_yf or "").upper() + ".SA"
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
        return f"{self.user} ♥ {self.asset}"
