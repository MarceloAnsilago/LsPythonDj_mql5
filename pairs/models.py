from __future__ import annotations

from django.conf import settings
from django.core.validators import MinValueValidator
from django.db import models

from acoes.models import Asset
from .constants import (
    DEFAULT_WINDOWS,
    DEFAULT_BASE_WINDOW,
    DEFAULT_BETA_WINDOW,
    DEFAULT_ADF_MIN,
    DEFAULT_ZSCORE_ABS_MIN,
    DEFAULT_HALF_LIFE_MAX,
)


def _default_windows_string() -> str:
    return ",".join(str(w) for w in DEFAULT_WINDOWS)


class Pair(models.Model):
    left = models.ForeignKey(Asset, on_delete=models.CASCADE, related_name="pairs_left")
    right = models.ForeignKey(Asset, on_delete=models.CASCADE, related_name="pairs_right")
    base_window = models.IntegerField(default=220)
    chosen_window = models.IntegerField(null=True, blank=True)
    scan_cache_json = models.JSONField(null=True, blank=True)
    scan_cached_at = models.DateTimeField(null=True, blank=True)

    def __str__(self) -> str:
        return f"{self.left.ticker} - {self.right.ticker}"

    def _canonicalize_order(self) -> None:
        if self.left_id and self.right_id and self.left_id > self.right_id:
            self.left_id, self.right_id = self.right_id, self.left_id

    def save(self, *args, **kwargs):
        self._canonicalize_order()
        super().save(*args, **kwargs)

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=["left", "right"], name="pairs_pair_left_right_unique"),
        ]
        indexes = [
            models.Index(fields=["left", "right"]),
            models.Index(fields=["scan_cached_at"]),
        ]


class UserMetricsConfig(models.Model):
    user = models.OneToOneField(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="metrics_config",
    )
    base_window = models.PositiveIntegerField(
        default=DEFAULT_BASE_WINDOW,
        validators=[MinValueValidator(10)],
    )
    default_windows = models.CharField(
        max_length=255,
        default=_default_windows_string,
        help_text="Lista de janelas (dias) separadas por virgula.",
    )
    adf_min = models.FloatField(default=DEFAULT_ADF_MIN)
    zscore_abs_min = models.FloatField(default=DEFAULT_ZSCORE_ABS_MIN)
    beta_window = models.PositiveIntegerField(
        default=DEFAULT_BETA_WINDOW,
        validators=[MinValueValidator(1)],
    )
    half_life_max = models.FloatField(
        default=DEFAULT_HALF_LIFE_MAX or 0,
        help_text="Half-life maxima em dias. Use 0 para desativar o filtro.",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Configuracao de Métricas"
        verbose_name_plural = "Configuracoes de Métricas"

    def __str__(self) -> str:
        return f"Config métricas de {self.user}"

    @classmethod
    def default_kwargs(cls) -> dict:
        return {
            "base_window": DEFAULT_BASE_WINDOW,
            "default_windows": _default_windows_string(),
            "adf_min": DEFAULT_ADF_MIN,
            "zscore_abs_min": DEFAULT_ZSCORE_ABS_MIN,
            "beta_window": DEFAULT_BETA_WINDOW,
            "half_life_max": DEFAULT_HALF_LIFE_MAX or 0,
        }

    def windows_list(self) -> list[int]:
        values: list[int] = []
        for chunk in (self.default_windows or "").split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            try:
                num = int(chunk)
            except ValueError:
                continue
            if num > 0 and num not in values:
                values.append(num)
        if not values:
            values = list(DEFAULT_WINDOWS)
        return values

    def windows_descending(self) -> list[int]:
        return sorted(self.windows_list(), reverse=True)
