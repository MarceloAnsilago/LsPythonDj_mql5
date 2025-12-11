from __future__ import annotations

from datetime import date, datetime
from typing import Optional

import pandas as pd
import yfinance as yf


class YahooQuoteProvider:
    """
    Provider simplificado de histórico diário via Yahoo Finance.

    Retorna Series indexada por date com valores de close.
    """

    def __init__(self, timeout: Optional[int] = 10):
        self.timeout = timeout

    def _coerce_date(self, value) -> Optional[date]:
        if value is None:
            return None
        if isinstance(value, date):
            return value
        if isinstance(value, datetime):
            return value.date()
        try:
            return date.fromisoformat(str(value)[:10])
        except Exception:
            return None

    def get_history(self, ticker: str, start: date, end: date) -> pd.Series:
        """
        Retorna série de preços de fechamento indexada por data.
        Se nada vier do Yahoo, devolve Series vazia.
        """
        start_dt = self._coerce_date(start)
        end_dt = self._coerce_date(end)
        if not ticker:
            return pd.Series(dtype=float)
        try:
            df = yf.download(
                ticker,
                start=start_dt,
                end=end_dt,
                interval="1d",
                progress=False,
                threads=False,
                auto_adjust=False,
            )
        except Exception:
            return pd.Series(dtype=float)

        if df is None or df.empty or "Close" not in df.columns:
            return pd.Series(dtype=float)

        df = df.dropna(subset=["Close"])
        if df.empty:
            return pd.Series(dtype=float)

        series = pd.Series(df["Close"].values, index=df.index.date)
        return series
