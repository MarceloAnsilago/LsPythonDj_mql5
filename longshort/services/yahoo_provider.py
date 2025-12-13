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
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value
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

        last_exc = None

        def _download(*, start=None, end=None, period=None):
            return yf.download(
                tickers=ticker,
                start=start,
                end=end,
                period=period,
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=False,
                group_by="column",
            )

        def _to_series(df):
            if df is None or df.empty:
                return None
            close = None
            if isinstance(df.columns, pd.MultiIndex):
                try:
                    close = df["Close"]
                except Exception:
                    return None
            else:
                if "Close" not in df.columns:
                    return None
                close = df["Close"]
            if isinstance(close, pd.DataFrame):
                if close.shape[1] == 1:
                    close = close.iloc[:, 0]
                else:
                    return None
            try:
                close = close.dropna()
            except Exception:
                return None
            if close is None or close.empty:
                return None
            try:
                idx_dates = close.index.date
            except Exception:
                return None
            return pd.Series(close.values, index=idx_dates)

        df = None
        try:
            df = _download(start=start_dt, end=end_dt)
        except Exception as exc:
            last_exc = exc
            df = None

        series = _to_series(df)
        if series is not None:
            return series

        try:
            df = _download(period="300d")
        except Exception as exc:
            last_exc = exc
            df = None

        series = _to_series(df)
        if series is not None:
            return series

        if last_exc:
            raise last_exc
        return pd.Series(dtype=float)
