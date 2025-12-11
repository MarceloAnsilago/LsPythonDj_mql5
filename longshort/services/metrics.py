# longshort/services/metrics.py
from __future__ import annotations
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal, ROUND_DOWN
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd
from django.conf import settings
from django.db.models import Max
from django.utils import timezone
from statsmodels.tsa.stattools import adfuller

# Modelos
from acoes.models import Asset
from cotacoes.models import QuoteDaily
from mt5api.models import DailyPrice
from longshort.services.price_provider import get_daily_prices

# Util: half-life para OU discreto via regressão Δs_t = α + ρ s_{t-1} + ε
def _half_life(spread: pd.Series) -> float | None:
    s = spread.dropna()
    if len(s) < 3:
        return None
    ds = s.diff().dropna()
    s_lag = s.shift(1).dropna()
    ds = ds.loc[s_lag.index]
    if len(ds) < 3:
        return None
    # OLS simples: ds = a + rho * s_{t-1}
    X = np.vstack([np.ones(len(s_lag)), s_lag.values]).T
    y = ds.values
    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        rho = beta[1]
        # evitar log de <=0
        if 1 + rho <= 0:
            return None
        hl = -np.log(2) / np.log(1 + rho)
        if np.isfinite(hl) and hl > 0:
            return float(hl)
        return None
    except Exception:
        return None

def _last_corr(returns_a: pd.Series, returns_b: pd.Series, lookback: int) -> float | None:
    a = returns_a.dropna().tail(lookback)
    b = returns_b.dropna().tail(lookback)
    idx = a.index.intersection(b.index)
    if len(idx) < max(10, int(0.6 * lookback)):  # evitar amostra muito curta
        return None
    c = np.corrcoef(a.loc[idx].values, b.loc[idx].values)
    val = c[0, 1]
    return float(val) if np.isfinite(val) else None


def _latest_daily_date() -> date:
    """Última data conhecida considerando QuoteDaily e DailyPrice."""
    qd_max = QuoteDaily.objects.aggregate(Max("date"))["date__max"]
    mt5_max = DailyPrice.objects.aggregate(Max("date"))["date__max"]
    candidates = [d for d in [qd_max, mt5_max] if d is not None]
    if not candidates:
        return timezone.localdate()
    return max(candidates)


def _df_from_provider(asset: Asset, lookback: int, window_end: date | None = None, column_name: str = "close") -> pd.DataFrame:
    """
    Busca preços diários via price_provider e devolve DataFrame com colunas [date, column_name],
    já ordenado por data e limitado ao tail(lookback).
    """
    prices = get_daily_prices(asset, end_date=window_end)
    if not prices:
        return pd.DataFrame(columns=["date", column_name])
    df = pd.DataFrame(prices)
    if "close" not in df.columns:
        return pd.DataFrame(columns=["date", column_name])
    df = df[["date", "close"]].dropna()
    df = df.sort_values("date").tail(lookback).rename(columns={"close": column_name})
    return df


@dataclass
class CandleUniverse:
    frames: dict[int, pd.DataFrame]

    @classmethod
    def from_dataframe(cls, dataframe: pd.DataFrame) -> "CandleUniverse":
        frames: dict[int, pd.DataFrame] = {}
        for asset_id, group in dataframe.groupby("asset_id", sort=False):
            sorted_group = group.sort_values("date").reset_index(drop=True)
            frames[int(asset_id)] = sorted_group
        return cls(frames)

    def tail_for(self, asset_id: int, limit: int) -> pd.DataFrame | None:
        frame = self.frames.get(asset_id)
        if frame is None or frame.empty:
            return None
        if limit <= 0 or limit >= len(frame):
            return frame
        return frame.iloc[-limit:]


CANDLE_LOOKBACK_PAD = 3

MIN_CORRELATION_THRESHOLD = getattr(settings, "PAIRS_MIN_CORRELATION_THRESHOLD", 0.5)
MIN_ZSCORE_FOR_HEAVY = getattr(settings, "PAIRS_MIN_ZSCORE_FOR_HEAVY", 1.5)


def load_candles_for_universe_from_supabase(
    universe: Iterable[int] | Iterable[Asset],
    lookback_windows: int,
    *,
    window_end: date | None = None,
    pad_factor: int = CANDLE_LOOKBACK_PAD,
) -> CandleUniverse:
    asset_ids = {asset if isinstance(asset, int) else asset.id for asset in universe}
    if not asset_ids:
        return CandleUniverse({})

    if window_end is None:
        window_end = _latest_daily_date()

    lookback_days = max(1, lookback_windows * pad_factor)
    window_start = window_end - timedelta(days=lookback_days)

    assets_map = {a.id: a for a in Asset.objects.filter(id__in=asset_ids)}
    rows: list[dict] = []
    for asset_id, asset in assets_map.items():
        prices = get_daily_prices(asset, start_date=window_start, end_date=window_end)
        for row in prices:
            rows.append(
                {
                    "asset_id": asset_id,
                    "date": row["date"],
                    "close": row["close"],
                }
            )

    data = pd.DataFrame(rows)
    if data.empty:
        return CandleUniverse({})

    data["date"] = pd.to_datetime(data["date"])
    data.sort_values(["asset_id", "date"], inplace=True)
    return CandleUniverse.from_dataframe(data)


def compute_pair_window_metrics(
    *,
    pair,
    window: int,
    candles: CandleUniverse | None = None,
) -> Dict[str, Any]:
    """
    Calcula métricas para um par (pair.left, pair.right) na janela 'window' (em dias úteis da sua base).
    Retorna dict com métricas leves (corr, n_samples etc.) e só roda os cálculos mais caros quando
    um par passar pelos filtros anteriores.
    """
    left = pair.left
    right = pair.right

    # Carrega últimos 'window' candles **alinhados por data** (inner join)
    # Estratégia: puxa um pouco mais e faz o alinhamento em pandas.
    lookback = max(window * 2, 1)
    latest_daily = _latest_daily_date()

    def _cached_asset_frame(asset_id: int, suffix: str) -> pd.DataFrame | None:
        if candles is None:
            return None
        frame = candles.tail_for(asset_id, lookback)
        if frame is None or frame.empty:
            return None
        return frame.rename(columns={"close": suffix})

    def _merge_frames(left_frame: pd.DataFrame, right_frame: pd.DataFrame) -> pd.DataFrame:
        merged = pd.merge(left_frame, right_frame, on="date", how="inner").sort_values("date")
        return merged.tail(window)

    df = None
    cached_left = _cached_asset_frame(pair.left_id, "close_l")
    cached_right = _cached_asset_frame(pair.right_id, "close_r")
    if cached_left is not None and cached_right is not None:
        df = _merge_frames(cached_left, cached_right)

    if df is None or df.empty:
        df_l = _df_from_provider(left, lookback, window_end=latest_daily, column_name="close_l")
        df_r = _df_from_provider(right, lookback, window_end=latest_daily, column_name="close_r")
        if df_l.empty or df_r.empty:
            return {"n_samples": 0}

        df = _merge_frames(df_l, df_r)

    if df.empty:
        return {
            "n_samples": 0,
            "ready_for_approval": False,
            "skip_reason": "Sem dados",
        }

    n = len(df)
    if n < 60:
        return {
            "n_samples": int(n),
            "ready_for_approval": False,
            "skip_reason": "Amostra insuficiente",
        }

    # Preços em log
    px_l = np.log(df["close_l"].astype(float))
    px_r = np.log(df["close_r"].astype(float))

    result: Dict[str, Any] = {
        "n_samples": int(n),
        "ready_for_approval": False,
        "skip_reason": None,
    }

    # Correlações de retornos (log-retornos)
    ret_l = px_l.diff()
    ret_r = px_r.diff()
    corr30 = _last_corr(ret_l, ret_r, 30)
    corr60 = _last_corr(ret_l, ret_r, 60)

    result.update(
        corr30=corr30,
        corr60=corr60,
    )

    valid_corrs = [c for c in (corr30, corr60) if c is not None]
    if not valid_corrs:
        result["skip_reason"] = "Sem correlacao confiavel"
        return result
    if all(abs(c) < MIN_CORRELATION_THRESHOLD for c in valid_corrs):
        result["skip_reason"] = "Correlacao insuficiente"
        return result

    # Estima beta via OLS: y = a + beta * x  (y=left, x=right)
    X = np.vstack([np.ones(n), px_r.values]).T
    y = px_l.values
    beta_hat = np.linalg.lstsq(X, y, rcond=None)[0][1]

    spread = px_l - beta_hat * px_r
    std = spread.std(ddof=1)
    if std == 0 or not np.isfinite(std):
        std = 1.0
    spread_z = (spread - spread.mean()) / std

    zscore_value: Optional[float] = None
    if not spread_z.empty:
        last_z = spread_z.iloc[-1]
        if np.isfinite(last_z):
            zscore_value = float(last_z)

    result.update(
        beta=float(beta_hat),
        zscore=zscore_value,
    )

    has_good_corr = any(
        corr is not None and abs(corr) >= MIN_CORRELATION_THRESHOLD
        for corr in (corr30, corr60)
    )
    heavy_ready = (
        zscore_value is not None
        and abs(zscore_value) >= MIN_ZSCORE_FOR_HEAVY
        and has_good_corr
    )

    adf_pvalue: Optional[float] = None
    half_life: Optional[float] = None

    if heavy_ready:
        try:
            adf_res = adfuller(spread.values, maxlag=1, regression="c", autolag="AIC")
            adf_pvalue = float(adf_res[1])
        except Exception:
            adf_pvalue = None
        half_life = _half_life(spread)
        result["ready_for_approval"] = True
        result["skip_reason"] = None
    else:
        if result["skip_reason"] is None:
            result["skip_reason"] = "Zscore insuficiente ou correlacao fraca"
        result["ready_for_approval"] = False
    result["adf_pvalue"] = adf_pvalue
    result["half_life"] = half_life

    return result


@dataclass(frozen=True)
class LongShortProportionResult:
    """
    Resultado do dimensionamento da operacao de long & short.
    """
    preco_short: Decimal
    preco_long: Decimal
    lote: int
    quantidade_vendida: int
    quantidade_comprada: int
    valor_vendido: Decimal
    valor_comprado: Decimal
    saldo: Decimal
    valor_minimo_para_operar: Decimal
    ponta_mais_cara: str
    capital_utilizado: Decimal
    capital_informado: Optional[Decimal] = None
    ticker_short: Optional[str] = None
    ticker_long: Optional[str] = None
    nome_short: Optional[str] = None
    nome_long: Optional[str] = None

    @property
    def proporcao(self) -> float:
        if self.quantidade_vendida <= 0:
            return 0.0
        return self.quantidade_comprada / self.quantidade_vendida

    @property
    def lotes_vendidos(self) -> int:
        return int(self.quantidade_vendida // self.lote) if self.lote else 0

    @property
    def lotes_comprados(self) -> int:
        return int(self.quantidade_comprada // self.lote) if self.lote else 0

    def _compose_label(self, ticker: Optional[str], nome: Optional[str]) -> str:
        if ticker and nome:
            return f"{ticker} ({nome})"
        return ticker or nome or "--"

    @property
    def resumo(self) -> str:
        def fmt_currency(valor: Decimal | float) -> str:
            dec_val = valor if isinstance(valor, Decimal) else Decimal(str(valor))
            quant = dec_val.quantize(Decimal("0.01"))
            formatted = f"{quant:,.2f}"
            formatted = formatted.replace(",", "X").replace(".", ",").replace("X", ".")
            return f"R$ {formatted}"

        short_label = self._compose_label(self.ticker_short, self.nome_short)
        long_label = self._compose_label(self.ticker_long, self.nome_long)
        partes: list[str] = []

        partes.append(
            f"Vende {self.quantidade_vendida} acoes de {short_label} a "
            f"{fmt_currency(self.preco_short)} (total {fmt_currency(self.valor_vendido)})."
        )

        if self.quantidade_comprada > 0:
            partes.append(
                f"Compra {self.quantidade_comprada} acoes de {long_label} a "
                f"{fmt_currency(self.preco_long)} (total {fmt_currency(self.valor_comprado)})."
            )
        else:
            partes.append(
                f"Nao ha recursos para comprar lote inteiro de {long_label}; "
                f"considere ajustar o limite."
            )

        if self.saldo > 0:
            partes.append(
                f"Recebe {fmt_currency(self.saldo)} como saldo reservado para a recompra da ponta short."
            )
        elif self.saldo < 0:
            partes.append(
                f"Precisa aportar {fmt_currency(abs(self.saldo))} para completar a ponta long."
            )
        else:
            partes.append("Operacao balanceada sem saldo remanescente.")

        partes.append(
            f"Proporcao long/short {self.proporcao:.4f} com {self.lotes_vendidos} "
            f"lotes de {self.lote} acoes."
        )

        mais_cara_label = (
            long_label if self.ponta_mais_cara == "long" else short_label
        )
        partes.append(
            f"Valor minimo recomendado para operar: {fmt_currency(self.valor_minimo_para_operar)} "
            f"({mais_cara_label})."
        )

        if self.capital_informado is not None:
            if self.capital_informado != self.capital_utilizado:
                partes.append(
                    f"Capital informado {fmt_currency(self.capital_informado)} ajustado para "
                    f"{fmt_currency(self.capital_utilizado)} no plano."
                )
            else:
                partes.append(f"Capital informado/utilizado: {fmt_currency(self.capital_utilizado)}.")
        else:
            partes.append(f"Capital utilizado no plano: {fmt_currency(self.capital_utilizado)}.")

        return " ".join(partes)

    def to_payload(self) -> Dict[str, Any]:
        return {
            "preco_short": float(self.preco_short),
            "preco_long": float(self.preco_long),
            "lote": self.lote,
            "quantidade_vendida": self.quantidade_vendida,
            "quantidade_comprada": self.quantidade_comprada,
            "valor_vendido": float(self.valor_vendido),
            "valor_comprado": float(self.valor_comprado),
            "saldo": float(self.saldo),
            "valor_minimo_para_operar": float(self.valor_minimo_para_operar),
            "ponta_mais_cara": self.ponta_mais_cara,
            "proporcao": self.proporcao,
            "lotes_vendidos": self.lotes_vendidos,
            "lotes_comprados": self.lotes_comprados,
            "capital_utilizado": float(self.capital_utilizado),
            "capital_informado": float(self.capital_informado) if self.capital_informado is not None else None,
            "ticker_short": self.ticker_short,
            "ticker_long": self.ticker_long,
            "nome_short": self.nome_short,
            "nome_long": self.nome_long,
            "resumo": self.resumo,
        }


def calcular_proporcao_long_short(
    preco_short: float,
    preco_long: float,
    limite_venda: float,
    lote: int = 100,
    *,
    ticker_short: Optional[str] = None,
    ticker_long: Optional[str] = None,
    nome_short: Optional[str] = None,
    nome_long: Optional[str] = None,
    capital_informado: Optional[float] = None,
) -> Optional[LongShortProportionResult]:
    """
    Calcula quantas acoes podem ser vendidas (short) e compradas (long) respeitando
    lotes inteiros e um limite maximo de alocacao na ponta vendida.
    Retorna um LongShortProportionResult ou None quando nenhum lote pode ser montado.

    Args:
        preco_short: Preco unitario da acao que sera vendida a descoberto.
        preco_long: Preco unitario da acao que sera comprada.
        limite_venda: Valor maximo autorizado para o lado vendido (capital utilizado).
        lote: Tamanho do lote minimo (padrao 100).
        ticker_short: Identificador da acao na ponta short (opcional).
        ticker_long: Identificador da acao na ponta long (opcional).
        nome_short: Nome/descritivo da acao na ponta short (opcional).
        nome_long: Nome/descritivo da acao na ponta long (opcional).
        capital_informado: Valor originalmente informado pelo usuario antes de ajustes (opcional).

    O resultado inclui `valor_minimo_para_operar`, calculado como o preco mais caro entre as pontas
    multiplicado por um lote completo. Esse valor pode ser exibido para o usuario como capital minimo
    recomendado antes de iniciar a operacao.
    """
    preco_short_dec = Decimal(str(preco_short))
    preco_long_dec = Decimal(str(preco_long))
    limite_venda_dec = Decimal(str(limite_venda))
    lote_dec = Decimal(lote)
    capital_informado_dec = (
        Decimal(str(capital_informado)) if capital_informado is not None else None
    )

    if preco_short_dec <= 0 or preco_long_dec <= 0 or lote <= 0:
        raise ValueError("Precos e lote precisam ser positivos.")
    if limite_venda_dec <= 0:
        return None

    ponta_mais_cara = "long" if preco_long_dec >= preco_short_dec else "short"
    valor_minimo_para_operar = max(preco_short_dec, preco_long_dec) * lote_dec

    valor_por_lote_short = preco_short_dec * lote_dec
    lotes_vendidos = int(
        (limite_venda_dec / valor_por_lote_short).to_integral_value(rounding=ROUND_DOWN)
    )
    if lotes_vendidos == 0:
        return None

    quantidade_vendida = lotes_vendidos * lote
    valor_vendido = preco_short_dec * Decimal(quantidade_vendida)

    valor_por_lote_long = preco_long_dec * lote_dec
    lotes_comprados = int(
        (valor_vendido / valor_por_lote_long).to_integral_value(rounding=ROUND_DOWN)
    )
    quantidade_comprada = lotes_comprados * lote
    valor_comprado = preco_long_dec * Decimal(quantidade_comprada)

    saldo = valor_vendido - valor_comprado

    return LongShortProportionResult(
        preco_short=preco_short_dec,
        preco_long=preco_long_dec,
        lote=lote,
        quantidade_vendida=quantidade_vendida,
        quantidade_comprada=quantidade_comprada,
        valor_vendido=valor_vendido,
        valor_comprado=valor_comprado,
        saldo=saldo,
        valor_minimo_para_operar=valor_minimo_para_operar,
        ponta_mais_cara=ponta_mais_cara,
        capital_utilizado=limite_venda_dec,
        capital_informado=capital_informado_dec,
        ticker_short=ticker_short,
        ticker_long=ticker_long,
        nome_short=nome_short,
        nome_long=nome_long,
    )


def get_zscore_series(pair, window: int) -> list[tuple[pd.Timestamp, float]]:
    """
    Retorna uma série (date, z) do Z-score do spread (log L - beta * log R)
    calculado na janela informada. O beta é estimado uma única vez via OLS
    no período e o Z-score é padronizado usando média e desvio do spread no período.
    """
    left = pair.left
    right = pair.right

    # Carrega candles a mais para garantir alinhamento e depois corta na janela
    latest_daily = _latest_daily_date()
    df_l = _df_from_provider(left, window * 2, window_end=latest_daily, column_name="close_l")
    df_r = _df_from_provider(right, window * 2, window_end=latest_daily, column_name="close_r")
    if df_l.empty or df_r.empty:
        return []

    df = (
        pd.merge(df_l, df_r, on="date", how="inner")
          .sort_values("date")
          .tail(window)
          .reset_index(drop=True)
    )
    n = len(df)
    if n < 10:  # limite mínimo para algo apresentável
        return []

    # Preços em log
    px_l = np.log(df["close_l"].astype(float))
    px_r = np.log(df["close_r"].astype(float))

    # Beta via OLS: y = a + beta * x
    X = np.vstack([np.ones(n), px_r.values]).T
    y = px_l.values
    beta_hat = np.linalg.lstsq(X, y, rcond=None)[0][1]

    # Spread e Z-score padronizado no período todo
    spread = px_l - beta_hat * px_r
    std = spread.std(ddof=1)
    if std == 0 or not np.isfinite(std):
        return []
    spread_z = (spread - spread.mean()) / std

    # Retorna lista (date, z)
    dates = pd.to_datetime(df["date"])
    series = [(d.to_pydatetime() if hasattr(d, "to_pydatetime") else d, float(z))
              for d, z in zip(dates, spread_z)]
    return series


def get_normalized_price_series(pair, window: int) -> list[tuple[pd.Timestamp, float, float]]:
    """
    Retorna uma série (date, norm_left, norm_right) com os preços normalizados
    (base 100) para o par na janela informada.
    """
    left = pair.left
    right = pair.right

    latest_daily = _latest_daily_date()
    df_l = _df_from_provider(left, window * 2, window_end=latest_daily, column_name="close_l")
    df_r = _df_from_provider(right, window * 2, window_end=latest_daily, column_name="close_r")
    if df_l.empty or df_r.empty:
        return []

    df = (
        pd.merge(df_l, df_r, on="date", how="inner")
          .sort_values("date")
          .tail(window)
          .reset_index(drop=True)
    )
    if df.empty or len(df) < 2:
        return []

    base_l = float(df["close_l"].iloc[0])
    base_r = float(df["close_r"].iloc[0])
    if base_l == 0 or base_r == 0:
        return []

    df["norm_left"] = (df["close_l"].astype(float) / base_l) * 100.0
    df["norm_right"] = (df["close_r"].astype(float) / base_r) * 100.0

    dates = pd.to_datetime(df["date"])
    series = [
        (
            d.to_pydatetime() if hasattr(d, "to_pydatetime") else d,
            float(nl),
            float(nr),
        )
        for d, nl, nr in zip(dates, df["norm_left"], df["norm_right"])
    ]
    return series


def get_moving_beta_series(pair, window: int, beta_window: int = 5) -> list[tuple[pd.Timestamp, float]]:
    """
    Calcula o beta via OLS em blocos de 'beta_window' períodos ao longo da janela.
    Retorna lista de tuplas (data_final_do_bloco, beta).
    """
    if beta_window <= 1:
        return []

    left = pair.left
    right = pair.right

    latest_daily = _latest_daily_date()
    df_l = _df_from_provider(left, window * 2, window_end=latest_daily, column_name="close_l")
    df_r = _df_from_provider(right, window * 2, window_end=latest_daily, column_name="close_r")
    if df_l.empty or df_r.empty:
        return []

    df = (
        pd.merge(df_l, df_r, on="date", how="inner")
          .sort_values("date")
          .tail(window)
          .reset_index(drop=True)
    )
    n = len(df)
    if n < beta_window:
        return []

    px_l = np.log(df["close_l"].astype(float))
    px_r = np.log(df["close_r"].astype(float))
    dates = pd.to_datetime(df["date"])

    series: list[tuple[pd.Timestamp, float]] = []
    for end_idx in range(beta_window, n + 1, beta_window):
        start_idx = end_idx - beta_window
        sub_l = px_l.iloc[start_idx:end_idx]
        sub_r = px_r.iloc[start_idx:end_idx]
        X = np.vstack([np.ones(beta_window), sub_r.values]).T
        y = sub_l.values
        try:
            beta_hat = np.linalg.lstsq(X, y, rcond=None)[0][1]
        except Exception:
            continue
        dt = dates.iloc[end_idx - 1]
        series.append((
            dt.to_pydatetime() if hasattr(dt, "to_pydatetime") else dt,
            float(beta_hat),
        ))

    return series


def get_pair_timeseries_and_metrics(
    pair,
    window: int,
    beta_window: int = 5,
) -> dict[str, Any]:
    """
    Cria um conjunto unificado de candles, métricas e séries para a análise de um par.
    """
    candles = load_candles_for_universe_from_supabase(
        universe=[pair.left_id, pair.right_id],
        lookback_windows=window,
    )

    left_frame = candles.tail_for(pair.left_id, limit=window * 2)
    right_frame = candles.tail_for(pair.right_id, limit=window * 2)

    if left_frame is None or right_frame is None or left_frame.empty or right_frame.empty:
        return {
            "metrics": {"n_samples": 0},
            "zscore_series": [],
            "normalized_series": [],
            "moving_beta_series": [],
        }

    df = (
        left_frame.rename(columns={"close": "close_l"})
        .merge(
            right_frame.rename(columns={"close": "close_r"}),
            on="date",
            how="inner",
        )
        .sort_values("date")
        .tail(window)
        .reset_index(drop=True)
    )

    n = len(df)
    if n < 2:
        return {
            "metrics": {"n_samples": n},
            "zscore_series": [],
            "normalized_series": [],
            "moving_beta_series": [],
        }

    metrics = compute_pair_window_metrics(pair=pair, window=window, candles=candles)

    px_l = np.log(df["close_l"].astype(float))
    px_r = np.log(df["close_r"].astype(float))
    X = np.vstack([np.ones(n), px_r.values]).T
    y = px_l.values
    beta_hat = np.linalg.lstsq(X, y, rcond=None)[0][1]

    spread = px_l - beta_hat * px_r
    std = spread.std(ddof=1)
    zscore_series: list[tuple[pd.Timestamp, float]] = []
    if std != 0 and np.isfinite(std):
        spread_z = (spread - spread.mean()) / std
        dates = pd.to_datetime(df["date"])
        zscore_series = [
            (
                d.to_pydatetime() if hasattr(d, "to_pydatetime") else d,
                float(z),
            )
            for d, z in zip(dates, spread_z)
            if np.isfinite(z)
        ]

    normalized_series: list[tuple[pd.Timestamp, float, float]] = []
    if n > 0:
        base_l = float(df["close_l"].iloc[0])
        base_r = float(df["close_r"].iloc[0])
        if base_l != 0 and base_r != 0 and np.isfinite(base_l) and np.isfinite(base_r):
            norm_left = (df["close_l"].astype(float) / base_l) * 100.0
            norm_right = (df["close_r"].astype(float) / base_r) * 100.0
            dates = pd.to_datetime(df["date"])
            normalized_series = [
                (
                    d.to_pydatetime() if hasattr(d, "to_pydatetime") else d,
                    float(nl),
                    float(nr),
                )
                for d, nl, nr in zip(dates, norm_left, norm_right)
            ]

    moving_beta_series: list[tuple[pd.Timestamp, float]] = []
    if beta_window > 1 and n >= beta_window:
        dates = pd.to_datetime(df["date"])
        for end_idx in range(beta_window, n + 1, beta_window):
            start_idx = end_idx - beta_window
            sub_l = px_l.iloc[start_idx:end_idx]
            sub_r = px_r.iloc[start_idx:end_idx]
            Xb = np.vstack([np.ones(beta_window), sub_r.values]).T
            yb = sub_l.values
            try:
                beta_sub = np.linalg.lstsq(Xb, yb, rcond=None)[0][1]
            except Exception:
                continue
            dt = dates.iloc[end_idx - 1]
            moving_beta_series.append(
                (
                    dt.to_pydatetime() if hasattr(dt, "to_pydatetime") else dt,
                    float(beta_sub),
                )
            )

    return {
        "metrics": metrics,
        "zscore_series": zscore_series,
        "normalized_series": normalized_series,
        "moving_beta_series": moving_beta_series,
    }
