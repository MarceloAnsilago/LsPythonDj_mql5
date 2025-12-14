from __future__ import annotations

"""
Fluxo Yahoo + MT5 live:
- Yahoo (yfinance) e a fonte unica persistida para QuoteDaily (dias < hoje).
- Intraday/tempo real segue vindo via MT5/QuoteLive sem gravar candles no banco.
"""

import time
from datetime import date, datetime, timedelta
from typing import Callable, Iterable, Optional

import pandas as pd
from django.conf import settings
from django.core.cache import cache
from django.db.models import Max, Min
from django.db.utils import OperationalError
from django.utils import timezone

from acoes.models import Asset
from cotacoes.models import MissingQuoteLog, QuoteDaily, QuoteLive
from longshort.services.mt5_provider import MT5QuoteProvider
from longshort.services.yahoo_provider import YahooQuoteProvider

# -----------------------
# Progresso (callback)
# -----------------------
# assinatura: (ticker, idx1, total, status, rows_inserted)
ProgressCB = Optional[Callable[[str, int, int, str, int], None]]


INCREMENTAL_LOOKBACK_DAYS = 5  # dias de folga ao baixar de forma incremental
BULK_BATCH_SIZE = 1000  # flush para nao acumular objetos em memoria
MIN_HISTORY_BARS = 200  # minimo de barras diarias desejado por ativo
DEFAULT_TOLERANCE = 1e-6
MAX_PROVIDER_RETRIES = 3
PROVIDER_RETRY_DELAY = 0.1


class MissingReason:
    YAHOO_TICKER_MISSING = "yahoo_ticker_missing"
    YAHOO_HISTORY_ERROR = "yahoo_history_error"
    YAHOO_HISTORY_EMPTY = "yahoo_history_empty"
    YAHOO_ERROR = "yahoo_error"


def _safe_float(value: object) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _ticker_for_yahoo(asset: Asset) -> str:
    """
    Retorna simbolo para Yahoo usando normalizacao do modelo (mapeamentos antes do .SA).
    """
    raw = getattr(asset, "ticker_yf", "") or getattr(asset, "ticker", "")
    sym = (raw or "").strip().upper()
    if hasattr(asset, "_normalize_ticker_yf"):
        # Respeita o mapeamento/override do modelo (ja aplica .SA quando necessario)
        sym = (asset._normalize_ticker_yf(sym) or "").strip().upper()
        return sym
    if not sym:
        return ""
    # Fallback generico: se nao houver sufixo e houver digito (B3), forca .SA
    if any(ch in sym for ch in ".-="):
        return sym
    if any(ch.isdigit() for ch in sym):
        return f"{sym}.SA"
    return sym


def _today_local() -> date:
    try:
        return timezone.localdate()
    except Exception:
        return date.today()


def _coerce_dt_value(dt_value) -> date | None:
    if dt_value is None:
        return None
    if isinstance(dt_value, datetime):
        return dt_value.date()
    if isinstance(dt_value, date):
        return dt_value
    try:
        if hasattr(dt_value, "to_pydatetime"):
            return dt_value.to_pydatetime().date()
    except Exception:
        pass
    if hasattr(dt_value, "date"):
        try:
            return dt_value.date()
        except Exception:
            return None
    try:
        return date.fromisoformat(str(dt_value)[:10])
    except Exception:
        return None


def _log_missing(asset: Asset, reason: str, detail: str) -> None:
    if asset is None:
        return
    try:
        MissingQuoteLog.objects.create(asset=asset, reason=reason, detail=(detail or "")[:500])
    except Exception:
        pass


def _call_progress(progress_cb: ProgressCB, ticker: str, idx: int, total: int, status: str, rows: int) -> None:
    if progress_cb is None:
        return
    try:
        progress_cb(ticker, idx, total, status, rows)
    except Exception:
        pass


def _call_with_retry(fn: Callable[[], object], *, retries: int = MAX_PROVIDER_RETRIES, delay: float = PROVIDER_RETRY_DELAY):
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
    if last_exc:
        raise last_exc


def _normalize_series_for_ingest(series, *, today: date, allow_today: bool) -> dict[date, float]:
    if series is None or not hasattr(series, "items"):
        return {}
    try:
        if hasattr(series, "dropna"):
            series = series.dropna()
    except Exception:
        pass

    rows: dict[date, float] = {}
    for dt_value, px in series.items():
        close = _safe_float(px)
        if close is None:
            continue
        the_date = _coerce_dt_value(dt_value)
        if the_date is None:
            continue
        if the_date > today:
            continue
        if not allow_today and the_date >= today:
            continue
        rows[the_date] = close
    return rows


def ingest_daily_series(
    asset: Asset,
    series,
    *,
    today: date | None = None,
    mode: str = "update_if_changed",  # insert_only | upsert | update_if_changed
    allow_today: bool = False,
    is_provisional: bool = False,
    batch_size: int = BULK_BATCH_SIZE,
    tolerance: float = DEFAULT_TOLERANCE,
) -> tuple[int, int]:
    """
    Grava serie (date -> close) em QuoteDaily em modo batelada.
    Retorna (criadas, atualizadas).
    """
    if asset is None:
        return 0, 0

    today = today or _today_local()
    rows_map = _normalize_series_for_ingest(series, today=today, allow_today=allow_today)
    if not rows_map:
        return 0, 0

    target_dates = list(rows_map.keys())
    existing_qs = QuoteDaily.objects.filter(asset=asset, date__in=target_dates).only("id", "date", "close", "is_provisional")
    existing_by_date = {row.date: row for row in existing_qs}

    to_create: list[QuoteDaily] = []
    to_update: list[QuoteDaily] = []

    for the_date, close in rows_map.items():
        current = existing_by_date.get(the_date)
        if current is None:
            to_create.append(QuoteDaily(asset=asset, date=the_date, close=close, is_provisional=is_provisional))
            continue

        if mode == "insert_only":
            continue

        if is_provisional and not current.is_provisional:
            # nao sobrescreve dado definitivo com provisorio
            continue

        needs_update = mode == "upsert"
        if not needs_update and abs(current.close - close) > tolerance:
            needs_update = True
        if not is_provisional and current.is_provisional:
            needs_update = True

        if needs_update:
            current.close = close
            current.is_provisional = is_provisional
            to_update.append(current)

    created = 0
    updated = 0

    if to_create:
        QuoteDaily.objects.bulk_create(to_create, ignore_conflicts=True, batch_size=batch_size)
        created = len(to_create)

    if to_update:
        QuoteDaily.objects.bulk_update(to_update, ["close", "is_provisional"], batch_size=batch_size)
        updated = len(to_update)

    return created, updated


def _fetch_history_with_retry(
    provider,
    *,
    asset: Asset,
    reason_on_error: str,
    detail_label: str,
    retries: int = MAX_PROVIDER_RETRIES,
    delay: float = PROVIDER_RETRY_DELAY,
    **kwargs,
):
    try:
        return _call_with_retry(lambda: provider.get_history(**kwargs), retries=retries, delay=delay)
    except Exception as exc:
        _log_missing(asset, reason_on_error, f"Erro ao obter historico {detail_label}: {exc}")
        return None


def ensure_daily_history(asset: Asset, start=None, end=None, lookback_days: int = 200) -> int:
    """
    Garante historico D1 em QuoteDaily via Yahoo.
    - Baixa apenas datas < hoje
    - Limita o backfill a 'lookback_days' para tras
    - Recomeca do ultimo dia salvo + 1
    Retorna linhas gravadas (criadas + atualizadas).
    """
    if asset is None:
        return 0
    today = _today_local()
    end_dt = today - timedelta(days=1)

    last_dt = QuoteDaily.objects.filter(asset=asset, date__lt=today).aggregate(Max("date"))["date__max"]
    start_dt = _coerce_dt_value(start)
    if last_dt:
        start_dt = last_dt + timedelta(days=1)
    elif start_dt is None:
        start_dt = end_dt - timedelta(days=lookback_days - 1)
    min_start = end_dt - timedelta(days=lookback_days - 1)
    if start_dt is None or start_dt < min_start:
        start_dt = min_start
    if start_dt > end_dt:
        return 0

    ticker_yf = _ticker_for_yahoo(asset)
    if not ticker_yf:
        _log_missing(asset, MissingReason.YAHOO_TICKER_MISSING, "Ticker Yahoo ausente/delistado (ensure_daily_history)")
        return 0

    yahoo = YahooQuoteProvider()
    series = _fetch_history_with_retry(
        yahoo,
        asset=asset,
        reason_on_error=MissingReason.YAHOO_HISTORY_ERROR,
        detail_label="Yahoo",
        ticker=ticker_yf,
        start=start_dt,
        end=end_dt,
    )

    if series is None or getattr(series, "empty", True):
        _log_missing(asset, MissingReason.YAHOO_HISTORY_EMPTY, "Nenhum dado retornado pelo Yahoo (ensure_daily_history)")
        return 0

    created, updated = ingest_daily_series(asset, series, today=today, mode="update_if_changed")
    return created + updated


def ensure_min_history_for_asset(asset: Asset, min_bars: int = MIN_HISTORY_BARS) -> int:
    """
    Garante historico minimo fechado (data < hoje) para o ativo via Yahoo.
    Retorna o numero de linhas gravadas (criadas + atualizadas).
    """
    cache_key = f"min_history_seed_asset_{getattr(asset, 'id', 'none')}"
    if cache.get(cache_key):
        return 0
    if asset is None:
        return 0

    ticker = (getattr(asset, "ticker", "") or "").strip().upper()
    today = _today_local()

    existing = QuoteDaily.objects.filter(asset=asset, date__lt=today).count()
    if existing >= min_bars:
        cache.set(cache_key, True, timeout=3600)
        return 0

    lookback_days = max(400, min_bars * 2)
    start_dt = today - timedelta(days=lookback_days)
    end_dt = today - timedelta(days=1)

    yahoo_ticker = _ticker_for_yahoo(asset)
    if not yahoo_ticker:
        _log_missing(asset, MissingReason.YAHOO_TICKER_MISSING, "Ticker Yahoo ausente/delistado (mapeamento)")
        cache.set(cache_key, True, timeout=3600)
        return 0

    yahoo_series = _fetch_history_with_retry(
        YahooQuoteProvider(),
        asset=asset,
        reason_on_error=MissingReason.YAHOO_HISTORY_ERROR,
        detail_label="Yahoo",
        ticker=yahoo_ticker,
        start=start_dt,
        end=end_dt,
    )
    if yahoo_series is None or getattr(yahoo_series, "empty", True):
        _log_missing(asset, MissingReason.YAHOO_HISTORY_EMPTY, "Nenhum dado retornado pelo Yahoo (ensure_min_history_for_asset)")
        return 0

    created, updated = ingest_daily_series(asset, yahoo_series, today=today, mode="update_if_changed")
    total_written = created + updated

    if existing + created >= min_bars:
        cache.set(cache_key, True, timeout=3600)

    return total_written


def ensure_min_history_for_all_assets(min_bars: int = MIN_HISTORY_BARS) -> tuple[int, int]:
    """
    Garante historico minimo para todos os ativos.
    Retorna (ativos_afetados, linhas_gravadas).
    """
    qs = Asset.objects.filter(is_active=True).order_by("id")

    total_written = 0
    assets_touched = 0

    for asset in qs:
        written = ensure_min_history_for_asset(asset, min_bars=min_bars)
        if written > 0:
            assets_touched += 1
            total_written += written

    return assets_touched, total_written


# ============================================================
# Atualizacao diaria (D1 fechado)
# ============================================================
def bulk_update_quotes(
    assets: Iterable,
    period: str = "2y",
    interval: str = "1d",
    progress_cb: ProgressCB = None,
    use_stooq: bool = False,  # mantido na assinatura para compat, ignorado
) -> tuple[int, int]:
    """
    Atualiza cotacoes diarias fechadas apenas via Yahoo (dias < hoje).
    Retorna (n_ativos_modificados, n_linhas_gravadas).
    """
    yahoo = YahooQuoteProvider()
    assets = list(assets)
    total_assets = len(assets)
    today = _today_local()
    _call_progress(progress_cb, "start", 0, total_assets, "starting", 0)

    total_rows = 0
    assets_with_writes = 0

    def _period_to_days(label: str) -> int:
        text_label = (label or "").lower()
        if text_label.endswith("y"):
            try:
                return int(float(text_label[:-1]) * 365)
            except Exception:
                return 730
        if text_label.endswith("mo"):
            try:
                return int(float(text_label[:-2]) * 30)
            except Exception:
                return 180
        return 730

    lookback_days = _period_to_days(period)

    for idx, asset in enumerate(assets, start=1):
        ticker = (getattr(asset, "ticker", "") or "").strip().upper()
        if not ticker:
            _call_progress(progress_cb, "", idx, total_assets, "skip_invalid", 0)
            continue

        _call_progress(progress_cb, ticker, idx, total_assets, "processing", 0)

        last_dt = QuoteDaily.objects.filter(asset=asset, date__lt=today).aggregate(Max("date"))["date__max"]
        start_dt = (last_dt - timedelta(days=INCREMENTAL_LOOKBACK_DAYS)) if last_dt else (today - timedelta(days=lookback_days))
        end_dt = today - timedelta(days=1)
        if start_dt > end_dt:
            _call_progress(progress_cb, ticker, idx, total_assets, "no_data", 0)
            continue

        rows_written = 0

        yahoo_ticker = _ticker_for_yahoo(asset)
        if not yahoo_ticker:
            _log_missing(asset, MissingReason.YAHOO_TICKER_MISSING, "Ticker Yahoo ausente/delistado (mapeamento)")
            _call_progress(progress_cb, ticker, idx, total_assets, "no_data", 0)
            continue

        series = _fetch_history_with_retry(
            yahoo,
            asset=asset,
            reason_on_error=MissingReason.YAHOO_HISTORY_ERROR,
            detail_label="Yahoo",
            ticker=yahoo_ticker,
            start=start_dt,
            end=end_dt,
        )
        if series is None or getattr(series, "empty", True):
            _log_missing(asset, MissingReason.YAHOO_HISTORY_EMPTY, f"Nenhum dado retornado pelo Yahoo para {ticker}")
            _call_progress(progress_cb, ticker, idx, total_assets, "no_data", 0)
            continue
        created, updated = ingest_daily_series(asset, series, today=today, mode="update_if_changed")
        rows_written += created + updated

        if rows_written > 0:
            total_rows += rows_written
            assets_with_writes += 1
            _call_progress(progress_cb, ticker, idx, total_assets, "ok", rows_written)
        else:
            _call_progress(progress_cb, ticker, idx, total_assets, "no_data", 0)

    _call_progress(progress_cb, "done", total_assets, total_assets, "done", total_rows)

    return assets_with_writes, total_rows



# ============================================================
# Preco em "tempo real" (via MT5)
# ============================================================
def apply_live_quote(
    asset,
    *,
    bid: float | None = None,
    ask: float | None = None,
    last: float | None = None,
    price: float | None = None,
    as_of=None,
    source: str = "mt5",
):
    """
    Atualiza apenas a cotacao ao vivo (QuoteLive) com dados externos (ex.: MT5)
    e retorna o preco salvo. Nao grava mais em QuoteDaily aqui.
    """
    px = _safe_float(price) or _safe_float(last)
    if px is None and bid is not None and ask is not None:
        try:
            px = (float(bid) + float(ask)) / 2.0
        except Exception:
            px = None
    if px is None:
        return None

    when = as_of
    if when is None:
        when = timezone.now()
    else:
        try:
            if timezone.is_naive(when):
                when = timezone.make_aware(when, timezone.utc)
            else:
                when = as_of
        except Exception:
            when = timezone.now()

    defaults = {
        "price": px,
        "bid": _safe_float(bid),
        "ask": _safe_float(ask),
        "last": _safe_float(last) or px,
        "as_of": when,
        "source": source or "mt5",
    }
    for _ in range(3):
        try:
            QuoteLive.objects.update_or_create(asset=asset, defaults=defaults)
            break
        except OperationalError:
            time.sleep(0.05)

    return px


def persist_today_from_live_quotes(assets=None, target_date: date | None = None) -> tuple[int, int]:
    """
    Converte QuoteLive em QuoteDaily apenas para datas ja fechadas.
    Intraday (data >= hoje) e ignorado.
    """
    if assets is None:
        try:
            assets = Asset.objects.filter(is_active=True)
        except Exception:
            assets = Asset.objects.all()

    today = _today_local()
    target = target_date or today
    if target >= today:
        return 0, 0

    assets_count = 0
    rows_count = 0

    for asset in assets:
        live = QuoteLive.objects.filter(asset=asset).first()
        if not live:
            continue

        close = _safe_float(getattr(live, "price", None)) or _safe_float(getattr(live, "last", None))
        if close is None:
            continue

        created, updated = ingest_daily_series(
            asset,
            {target: close},
            today=today,
            allow_today=False,
            is_provisional=False,
            mode="update_if_changed",
        )
        if created + updated > 0:
            assets_count += 1
            rows_count += created + updated

    return assets_count, rows_count


def fetch_latest_price(ticker: str) -> Optional[float]:
    """
    Retorna o ultimo preco disponivel do MT5 (ticker B3).
    """
    provider = MT5QuoteProvider()
    ticker_norm = (ticker or "").strip().upper()
    if not ticker_norm:
        return None
    try:
        return _call_with_retry(lambda: provider.get_latest_price(ticker_norm))
    except Exception as exc:
        print(f"[mt5] erro latest price {ticker_norm}: {exc}")
        return None


def update_live_quotes(assets: Iterable, progress_cb: ProgressCB = None) -> tuple[int, int]:
    """
    Atualiza (ou cria) cotacoes em tempo real (tabela QuoteLive) via MT5.
    """
    if not getattr(settings, "USE_MT5_LIVE", True):
        total = len(list(assets))
        _call_progress(progress_cb, "mt5_live_disabled", total, total, "disabled", 0)
        return 0, total

    assets = list(assets)
    total = len(assets)
    updated = 0

    for idx, asset in enumerate(assets, start=1):
        if not getattr(asset, "use_mt5", False):
            _call_progress(progress_cb, "", idx, total, "skip_non_mt5", 0)
            continue
        ticker = getattr(asset, "ticker", "").strip().upper()
        if not ticker:
            continue

        _call_progress(progress_cb, ticker, idx, total, "processing_live", 0)

        px = fetch_latest_price(ticker)
        if px is None:
            _call_progress(progress_cb, ticker, idx, total, "no_data", 0)
            continue

        QuoteLive.objects.update_or_create(asset=asset, defaults={"price": px, "last": px, "source": "mt5"})
        updated += 1

        _call_progress(progress_cb, ticker, idx, total, "ok", 1)

    _call_progress(progress_cb, "done", total, total, "done", updated)

    return updated, total


# ============================================================
# Utilitario opcional (teste rapido de um ativo)
# ============================================================
def update_single_asset(ticker_b3: str, period: str = "2y", interval: str = "1d") -> tuple[int, int]:
    """
    Atualiza um unico ticker (string) sem precisar montar queryset.
    Util para depuracao pontual no shell.
    """
    from acoes.models import Asset

    asset = Asset.objects.filter(ticker=ticker_b3.upper()).first()
    if not asset:
        raise ValueError(f"Ativo {ticker_b3} nao encontrado")

    def _p(t, i, tot, st, rows):  # progress minimalista
        print(f"[{i}/{tot}] {t} -> {st} ({rows})")

    return bulk_update_quotes([asset], period=period, interval=interval, progress_cb=_p, use_stooq=False)


# ============================================================
# Scanner de buracos (mantido para compat; ignora MT5/use_mt5)
# ============================================================
def find_missing_dates_for_asset(
    asset,
    *,
    since_months: int | None = None,
    until: date | None = None,
) -> list[date]:
    if getattr(asset, "use_mt5", False):
        return []
    qs = QuoteDaily.objects.filter(asset=asset)

    bounds = qs.aggregate(min_dt=Min("date"), max_dt=Max("date"))
    min_dt, _ = bounds["min_dt"], bounds["max_dt"]
    if not min_dt:
        return []

    if until is None:
        until = date.today()

    start = min_dt
    if since_months:
        approx_days = int(since_months * 30.44)
        start = max(min_dt, until - timedelta(days=approx_days))

    expected = set(_business_days(start, until))

    ticker = getattr(asset, "ticker", "").upper()
    expected -= _ignored_days_for_ticker(ticker)

    existing = set(qs.values_list("date", flat=True))
    missing = sorted(expected - existing)
    return missing


def try_fill_missing_for_asset(
    asset,
    missing_dates: list[date],
    *,
    use_stooq: bool = False,
) -> tuple[int, list[date]]:
    # Sem Yahoo/Stooq: apenas informa que nao ha mecanismo automatico.
    return 0, sorted(missing_dates)


def scan_all_assets_and_fix(
    *,
    use_stooq: bool = False,
    since_months: int | None = None,
    tickers: list[str] | None = None,
):
    """
    Varre ativos, tenta corrigir buracos e retorna lista serializavel:
    [{ticker, missing_before, fixed, remaining:[YYYY-MM-DD,...]}]
    - Ignora ativos com use_mt5=True (tratados via MT5/QuoteDaily).
    """
    try:
        qs = Asset.objects.filter(is_active=True)
    except Exception:
        qs = Asset.objects.all()

    qs = qs.filter(use_mt5=False)

    if tickers:
        qs = qs.filter(ticker__in=[t.upper() for t in tickers])

    results = []
    for asset in qs.order_by("ticker"):
        missing = find_missing_dates_for_asset(asset, since_months=since_months)
        fixed, remaining = try_fill_missing_for_asset(asset, missing, use_stooq=use_stooq)
        results.append({
            "ticker": getattr(asset, "ticker", ""),
            "missing_before": int(len(missing)),
            "fixed": int(fixed),
            "remaining": [d.isoformat() for d in remaining],
        })
    return results


def _date_to_unix(d: date) -> int:
    # Mantido apenas para compat (ainda pode ser usado em templates)
    return int(datetime(d.year, d.month, d.day).timestamp())


def try_fetch_single_date(asset, d: date, *, use_stooq: bool = True) -> bool:
    """Sem Yahoo/Stooq, nao ha preenchimento automatico para datas isoladas."""
    if getattr(asset, "use_mt5", False):
        return False
    return False


# -------------------------------------------------------------------
# Feriados/uteis (mantidos do codigo original)
# -------------------------------------------------------------------
# ---------- FERIADOS B3 DINAMICOS (qualquer ano) ----------
def _easter_date(year: int) -> date:
    # Meeus/Jones/Butcher
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def _b3_holidays_for_year(year: int) -> set[date]:
    pascoa = _easter_date(year)
    carnaval_seg = pascoa - timedelta(days=48)
    carnaval_ter = pascoa - timedelta(days=47)
    sexta_santa = pascoa - timedelta(days=2)
    corpus_christi = pascoa + timedelta(days=60)
    fixed = {
        date(year, 1, 1),
        date(year, 4, 21),
        date(year, 5, 1),
        date(year, 9, 7),
        date(year, 10, 12),
        date(year, 11, 2),
        date(year, 11, 15),
        date(year, 11, 20),
        date(year, 12, 25),
        date(year, 12, 24),
        date(year, 12, 31),
    }
    mobile = {carnaval_seg, carnaval_ter, sexta_santa, corpus_christi}
    return fixed | mobile


def b3_holidays_between(start: date, end: date) -> set[date]:
    years = range(start.year, end.year + 1)
    out: set[date] = set()
    for y in years:
        out |= _b3_holidays_for_year(y)
    return {d for d in out if start <= d <= end}


def _business_days(start: date, end: date) -> list[date]:
    """Dias de negociacao: seg-sex excluindo feriados B3 (qualquer ano)."""
    if start > end:
        return []
    rng = pd.date_range(start, end, freq="B")
    holidays = b3_holidays_between(start, end)
    return [d.date() for d in rng if d.date() not in holidays]


IGNORED_RANGES: dict[str, list[tuple[date, date]]] = {
    # "BRFS3": [(date(2025, 9, 23), date(2025, 10, 2))],
}


def _ignored_days_for_ticker(ticker: str) -> set[date]:
    out: set[date] = set()
    for (ini, fim) in IGNORED_RANGES.get(ticker.upper(), []):
        if ini and fim and ini <= fim:
            for d in pd.date_range(ini, fim, freq="D"):
                out.add(d.date())
    return out
