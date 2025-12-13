from __future__ import annotations

from datetime import date, datetime, timedelta
from functools import lru_cache
from typing import List
import logging

from django.conf import settings
from django.db.models import Max, Q
from acoes.models import Asset
from cotacoes.models import QuoteDaily
from django.utils import timezone
from longshort.services.quotes import ensure_daily_history, b3_holidays_between

logger = logging.getLogger(__name__)


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


@lru_cache(maxsize=128)
def _holidays_for_year(year: int) -> set[date]:
    return b3_holidays_between(date(year, 1, 1), date(year, 12, 31))


def _is_business_day(d: date) -> bool:
    return d.weekday() < 5 and d not in _holidays_for_year(d.year)


def _last_business_day(ref: date | None = None) -> date:
    """
    Retorna o ultimo dia util <= ref (ou hoje local).
    Considera fim de semana e feriados B3.
    """
    d = ref or timezone.localdate()
    while not _is_business_day(d):
        d = d - timedelta(days=1)
    return d


def get_daily_prices(
    asset: Asset,
    start_date=None,
    end_date=None,
    *,
    fallback_to_quote: bool = True,
    closed_only: bool = True,
    include_provisional: bool = False,
) -> List[dict]:
    """
    Retorna candles diarios (ordenados por data) para o ativo.
    - Usa apenas cotacoes.QuoteDaily.
    - Por padrao (closed_only=True) retorna apenas D1 fechado ate o ultimo dia util (ignora finais de semana/feriados) e ignora is_provisional.
    - Se nao houver dados e fallback_to_quote=True, tenta garantir historico via ensure_daily_history.
    Sempre retorna lista de dicts: {date, open, high, low, close}.
    """
    if asset is None:
        return []

    start = _coerce_date(start_date)
    end = _coerce_date(end_date)
    today_local = timezone.localdate()

    if closed_only:
        cutoff_end = _last_business_day(end or today_local)
    else:
        # Quando permitido intraday, usa ate a data solicitada ou hoje
        cutoff_end = end or today_local

    if start and cutoff_end and start > cutoff_end:
        return []

    if fallback_to_quote:
        try:
            ensure_daily_history(asset, start, cutoff_end, lookback_days=200)
        except Exception:
            pass

    qs = QuoteDaily.objects.filter(asset=asset)
    if closed_only:
        qs = qs.filter(date__lte=cutoff_end)
    if start:
        qs = qs.filter(date__gte=start)
    if cutoff_end and not closed_only:
        qs = qs.filter(date__lte=cutoff_end)
    if not include_provisional:
        # Inclui o cutoff_end mesmo se estiver provisório para não perder o último pregão
        qs = qs.filter(Q(is_provisional=False) | Q(date=cutoff_end))
    qs = qs.order_by("date")

    # Debug auxiliar para validar fim de semana/feriado
    if settings.DEBUG:
        try:
            max_date = qs.aggregate(Max("date"))["date__max"]
            logger.debug(
                "get_daily_prices cutoff debug: today=%s cutoff_end=%s max_date=%s closed_only=%s",
                today_local,
                cutoff_end,
                max_date,
                closed_only,
            )
        except Exception:
            pass
    rows: List[dict] = []
    for row in qs:
        # Garante que nao devolvemos dados de dia nao util mesmo que estejam no banco
        if closed_only and not _is_business_day(row.date):
            continue
        rows.append(
            {
                "date": row.date,
                "open": row.close,
                "high": row.close,
                "low": row.close,
                "close": row.close,
            }
        )
    # Garantia extra: se estamos em fim de semana/feriado e existe linha do ultimo pregao,
    # assegure que ela esteja presente mesmo que filtros acima tenham ignorado por algum motivo.
    if closed_only and cutoff_end:
        if not any(r["date"] == cutoff_end for r in rows):
            missing_qs = QuoteDaily.objects.filter(asset=asset, date=cutoff_end).order_by("date")
            for row in missing_qs:
                rows.append(
                    {
                        "date": row.date,
                        "open": row.close,
                        "high": row.close,
                        "low": row.close,
                        "close": row.close,
                    }
                )
            rows = sorted(rows, key=lambda r: r["date"])
    return rows
