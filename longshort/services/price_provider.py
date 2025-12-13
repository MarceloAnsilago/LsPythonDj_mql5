from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import List

from acoes.models import Asset
from cotacoes.models import QuoteDaily
from django.utils import timezone
from longshort.services.quotes import ensure_daily_history


def _coerce_date(value) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return value.date()
    except Exception:
        return None


def get_daily_prices(asset: Asset, start_date=None, end_date=None, *, fallback_to_quote: bool = True) -> List[dict]:
    """
    Retorna candles diarios (ordenados por data) para o ativo.
    - Usa apenas cotacoes.QuoteDaily.
    - Se nao houver dados e fallback_to_quote=True, tenta garantir historico via ensure_daily_history.
    Sempre retorna lista de dicts: {date, open, high, low, close}.
    """
    if asset is None:
        return []

    start = _coerce_date(start_date)
    end = _coerce_date(end_date)
    today = timezone.localdate()
    cutoff_end = end
    if cutoff_end is None:
        cutoff_end = today
    if cutoff_end > today:
        cutoff_end = today

    if fallback_to_quote:
        try:
            ensure_daily_history(asset, start, cutoff_end, lookback_days=200)
        except Exception:
            pass

    qs = QuoteDaily.objects.filter(asset=asset)
    if start:
        qs = qs.filter(date__gte=start)
    if cutoff_end:
        qs = qs.filter(date__lte=cutoff_end)
    qs = qs.order_by("date")
    rows: List[dict] = []
    for row in qs:
        rows.append(
            {
                "date": row.date,
                "open": row.close,
                "high": row.close,
                "low": row.close,
                "close": row.close,
            }
        )
    return rows
