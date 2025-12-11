from __future__ import annotations

from datetime import date
from typing import List

from acoes.models import Asset
from cotacoes.models import QuoteDaily
from mt5api.models import DailyPrice
from longshort.services.quotes import ensure_min_history_for_asset


def _coerce_date(value) -> date | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value
    try:
        return value.date()
    except Exception:
        return None


def get_daily_prices(asset: Asset, start_date=None, end_date=None, *, fallback_to_quote: bool = True) -> List[dict]:
    """
    Retorna candles diarios (ordenados por data) para o ativo.
    - Se asset.use_mt5=True, usa mt5api.DailyPrice.
    - Caso contrario, usa cotacoes.QuoteDaily (replicando OHLC com close).
    - Se use_mt5=True mas nao existir DailyPrice e fallback_to_quote=True, cai para QuoteDaily,
      tentando garantir um mínimo de histórico via ensure_min_history_for_asset.
    Sempre retorna lista de dicts: {date, open, high, low, close}.
    """
    if asset is None:
        return []

    start = _coerce_date(start_date)
    end = _coerce_date(end_date)

    if getattr(asset, "use_mt5", False):
        qs = DailyPrice.objects.filter(ticker=asset.ticker)
        if start:
            qs = qs.filter(date__gte=start)
        if end:
            qs = qs.filter(date__lte=end)
        qs = qs.order_by("date")
        rows = [
            {
                "date": row.date,
                "open": row.open,
                "high": row.high,
                "low": row.low,
                "close": row.close,
            }
            for row in qs
        ]
        if rows or not fallback_to_quote:
            return rows
        # sem DailyPrice: tenta garantir histórico mínimo em QuoteDaily via MT5/Yahoo
        try:
            ensure_min_history_for_asset(asset)
        except Exception:
            pass

    qs = QuoteDaily.objects.filter(asset=asset)
    if start:
        qs = qs.filter(date__gte=start)
    if end:
        qs = qs.filter(date__lte=end)
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
