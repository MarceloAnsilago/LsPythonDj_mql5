from __future__ import annotations

"""
MT5QuoteProvider
----------------
Camada de acesso aos preços do MT5. Implementação assumindo uma API HTTP
alimentada pelo MT5 (Opção B). Se você tiver o terminal MT5 local com a
biblioteca oficial MetaTrader5, adapte aqui facilmente (ver TODO abaixo).
"""

from datetime import date, datetime
from typing import Optional

import pandas as pd
import requests
from django.conf import settings


class MT5QuoteProvider:
    """
    Provider de cotações do MT5 via HTTP.

    Endpoints esperados (exemplos):
      - GET {MT5_HTTP_BASE}/history?ticker=PETR4&start=YYYY-MM-DD&end=YYYY-MM-DD&tf=D
        -> JSON: [{"date":"2025-01-01","close":31.2}, ...]
      - GET {MT5_HTTP_BASE}/latest?ticker=PETR4 -> JSON: {"price": 31.2}

    Ajuste os endpoints conforme seu EA/bridge. Os métodos já tratam respostas
    vazias retornando Series/None.
    """

    def __init__(self, base_url: Optional[str] = None, timeout: int = 5):
        self.base_url = base_url or getattr(settings, "MT5_HTTP_BASE", "").rstrip("/")
        self.timeout = timeout

    # TODO: Caso prefira usar MetaTrader5 (biblioteca oficial), substitua estes
    # métodos por chamadas a mt5.copy_rates_range / mt5.symbol_info_tick, etc.

    def get_history(self, ticker: str, start: date, end: date, timeframe: str = "D") -> pd.Series:
        """
        Retorna série de preços de fechamento indexada por data.
        Se nada vier da API, devolve Series vazia.
        """
        if not self.base_url:
            return pd.Series(dtype=float)
        url = f"{self.base_url}/history"
        params = {
            "ticker": ticker,
            "start": start.isoformat() if isinstance(start, (date, datetime)) else start,
            "end": end.isoformat() if isinstance(end, (date, datetime)) else end,
            "tf": timeframe,
        }
        try:
            resp = requests.get(url, params=params, timeout=self.timeout)
            if resp.status_code != 200:
                return pd.Series(dtype=float)
            data = resp.json() or []
        except Exception as exc:
            print(f"[mt5_provider] history erro {ticker}: {exc}")
            return pd.Series(dtype=float)

        if not data:
            return pd.Series(dtype=float)
        df = pd.DataFrame(data)
        if "date" not in df.columns or "close" not in df.columns:
            return pd.Series(dtype=float)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        series = pd.Series(df["close"].values, index=df["date"])
        return series

    def get_latest_price(self, ticker: str) -> Optional[float]:
        """Retorna último preço disponível ou None."""
        if not self.base_url:
            return None
        url = f"{self.base_url}/latest"
        try:
            resp = requests.get(url, params={"ticker": ticker}, timeout=self.timeout)
            if resp.status_code != 200:
                return None
            data = resp.json() or {}
            price = data.get("price")
            return float(price) if price is not None else None
        except Exception as exc:
            print(f"[mt5_provider] latest erro {ticker}: {exc}")
            return None
