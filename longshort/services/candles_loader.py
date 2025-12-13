from __future__ import annotations

from datetime import date, timedelta
from typing import Iterable

import pandas as pd
from django.db.models import Max

from acoes.models import Asset
from cotacoes.models import QuoteDaily
from longshort.services.metrics import CandleUniverse, CANDLE_LOOKBACK_PAD


def _latest_daily_date() -> date:
    latest = QuoteDaily.objects.aggregate(Max("date"))["date__max"]
    return latest or date.today()


def load_candles_for_universe(
    universe: Iterable[int] | Iterable[Asset],
    lookback_windows: int,
    *,
    window_end: date | None = None,
    pad_factor: int = CANDLE_LOOKBACK_PAD,
) -> CandleUniverse:
    """
    Carrega candles locais (QuoteDaily) para um conjunto de ativos.
    Fallback opcional para outras fontes pode ser adicionado se configurado.
    """
    asset_ids = {asset if isinstance(asset, int) else asset.id for asset in universe}
    if not asset_ids:
        return CandleUniverse({})

    if window_end is None:
        window_end = _latest_daily_date()

    lookback_days = max(1, lookback_windows * pad_factor)
    window_start = window_end - timedelta(days=lookback_days)

    qs = QuoteDaily.objects.filter(asset_id__in=asset_ids, date__gte=window_start, date__lte=window_end)
    rows: list[dict] = [
        {
            "asset_id": row.asset_id,
            "date": row.date,
            "close": row.close,
        }
        for row in qs
    ]

    if not rows:
        return CandleUniverse({})

    data = pd.DataFrame(rows)
    data["date"] = pd.to_datetime(data["date"])
    data.sort_values(["asset_id", "date"], inplace=True)
    return CandleUniverse.from_dataframe(data)
