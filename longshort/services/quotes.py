# longshort/services/quotes.py
from __future__ import annotations

"""
Plano de refatoração (MT5 como única fonte):
- Removemos completamente yfinance/Yahoo; todas as cotações vêm do MT5.
- Para histórico diário, usamos MT5QuoteProvider.get_history e gravamos em QuoteDaily.
- Para preço "ao vivo", usamos MT5QuoteProvider.get_latest_price e gravamos em QuoteLive.
- Flags antigas USE_YAHOO_HISTORY foram eliminadas; USE_MT5_LIVE passa a significar
  apenas se devemos tentar buscar preço ao vivo ou não (padrão: True).
- Fallbacks antigos (Stooq) e funções de Yahoo foram removidos.
"""

from datetime import date, datetime, timedelta
from typing import Iterable, Optional, Callable
import time

import pandas as pd
from django.conf import settings
from django.core.cache import cache
from django.db import transaction
from django.db.models import Max, Min
from django.utils import timezone
from django.db.utils import OperationalError

from acoes.models import Asset
from cotacoes.models import QuoteDaily, MissingQuoteLog, QuoteLive
from longshort.services.mt5_provider import MT5QuoteProvider
from longshort.services.yahoo_provider import YahooQuoteProvider

# -----------------------
# Progresso (callback)
# -----------------------
# assinatura: (ticker, idx1, total, status, rows_inserted)
ProgressCB = Optional[Callable[[str, int, int, str, int], None]]


INCREMENTAL_LOOKBACK_DAYS = 5  # dias de folga ao baixar de forma incremental
BULK_BATCH_SIZE = 1000  # flush para nao acumular objetos em memoria
MIN_HISTORY_BARS = 200  # mínimo de barras diárias desejado por ativo


def _safe_float(value: object) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _ticker_for_yahoo(asset: Asset) -> str:
    """
    Retorna símbolo para Yahoo (usa ticker_yf se existir; senão adiciona .SA).
    """
    raw = getattr(asset, "ticker_yf", "") or getattr(asset, "ticker", "")
    sym = (raw or "").strip().upper()
    if not sym:
        return ""
    if "." not in sym:
        sym = f"{sym}.SA"
    return sym


def _update_daily_quote_close_if_changed(asset, quote_date, new_close) -> bool:
    new_value = _safe_float(new_close)
    if new_value is None:
        return False
    queryset = QuoteDaily.objects.filter(asset=asset, date=quote_date)
    row = queryset.first()
    if row is None:
        return False

    if abs(row.close - new_value) <= 1e-6:
        # Mesmo preço: apenas marca como definitivo se era provisório
        if row.is_provisional:
            queryset.update(is_provisional=False)
        return False

    queryset.update(close=new_value, is_provisional=False)
    return True


def ensure_today_placeholder(asset) -> None:
    """
    Se hoje for dia útil e ainda não existir QuoteDaily para hoje,
    cria uma linha provisória copiando o último fechamento conhecido.
    """
    today = timezone.localdate()
    if today not in _business_days(today, today):
        return

    qs = QuoteDaily.objects.filter(asset=asset)
    if qs.filter(date=today).exists():
        return

    last = qs.order_by("-date").first()
    if not last:
        return

    with transaction.atomic():
        QuoteDaily.objects.get_or_create(
            asset=asset,
            date=today,
            defaults={"close": last.close, "is_provisional": True},
        )


def ensure_min_history_for_asset(asset: Asset, min_bars: int = MIN_HISTORY_BARS) -> int:
    """
    Garante que o ativo tenha pelo menos `min_bars` barras diárias em QuoteDaily.
    Tenta MT5 primeiro; se vier vazio ou insuficiente, cai para Yahoo para completar.
    Retorna o número de linhas inseridas.
    """
    cache_key = f"min_history_seed_asset_{getattr(asset, 'id', 'none')}"
    if cache.get(cache_key):
        return 0

    existing = QuoteDaily.objects.filter(asset=asset).count()
    if existing >= min_bars:
        return 0

    today = timezone.localdate()
    lookback_days = max(400, min_bars * 2)
    start_dt = today - timedelta(days=lookback_days)
    end_dt = today + timedelta(days=1)

    def _insert_series(series, reason_tag: str) -> int:
        added = 0
        if series is None or not hasattr(series, "items") or getattr(series, "empty", False):
            return 0
        for dt, px in series.items():
            if px is None:
                continue
            try:
                the_date = getattr(dt, "date", lambda: dt)()
                # não sobrescreve o dia atual se já existe linha (MT5 cuida do dia)
                if the_date == today and QuoteDaily.objects.filter(asset=asset, date=today).exists():
                    continue
                obj, created = QuoteDaily.objects.get_or_create(
                    asset=asset,
                    date=the_date,
                    defaults={"close": float(px), "is_provisional": False},
                )
                if created:
                    added += 1
            except Exception:
                continue
        if added > 0 and reason_tag:
            try:
                MissingQuoteLog.objects.create(
                    asset=asset,
                    reason=reason_tag,
                    detail=f"Backfill inserido: {added} linhas",
                )
            except Exception:
                pass
        return added

    inserted = 0

    # 1) MT5 primeiro
    try:
        mt5_series = MT5QuoteProvider().get_history(
            ticker=getattr(asset, "ticker", "").strip().upper(),
            start=start_dt,
            end=end_dt,
            timeframe="D",
        )
    except Exception as exc:
        mt5_series = None
        try:
            MissingQuoteLog.objects.create(
                asset=asset,
                reason="mt5_history_error",
                detail=f"Erro ao obter historico MT5: {exc}",
            )
        except Exception:
            pass
    inserted += _insert_series(mt5_series, reason_tag="")

    total_after_mt5 = QuoteDaily.objects.filter(asset=asset).count()
    if total_after_mt5 >= min_bars:
        return inserted

    # 2) Fallback Yahoo para completar
    yahoo_ticker = _ticker_for_yahoo(asset)
    try:
        yahoo_series = YahooQuoteProvider().get_history(
            ticker=yahoo_ticker,
            start=start_dt,
            end=today,  # evita sobrescrever o dia atual
        )
    except Exception as exc:
        yahoo_series = None
        try:
            MissingQuoteLog.objects.create(
                asset=asset,
                reason="yahoo_fallback_error",
                detail=f"Erro ao obter historico Yahoo: {exc}",
            )
        except Exception:
            pass

    if yahoo_series is None or getattr(yahoo_series, "empty", True):
        try:
            MissingQuoteLog.objects.create(
                asset=asset,
                reason="yahoo_fallback_empty",
                detail="Nenhum dado retornado pelo Yahoo (ensure_min_history_for_asset)",
            )
        except Exception:
            pass
        cache.set(cache_key, True, timeout=3600)  # evita repetir imediatamente
        return inserted

    inserted += _insert_series(yahoo_series, reason_tag="yahoo_fallback")
    cache.set(cache_key, True, timeout=3600)  # evita repetição excessiva
    return inserted


def ensure_min_history_for_all_assets(min_bars: int = MIN_HISTORY_BARS) -> tuple[int, int]:
    """
    Garante histórico mínimo para todos os ativos.
    Retorna (ativos_afetados, linhas_inseridas).
    """
    qs = Asset.objects.filter(is_active=True).order_by("id")

    total_inserted = 0
    assets_touched = 0

    for asset in qs:
        inserted = ensure_min_history_for_asset(asset, min_bars=min_bars)
        if inserted > 0:
            assets_touched += 1
            total_inserted += inserted

    return assets_touched, total_inserted


# ============================================================
# Atualização diária (MT5 como fonte única)
# ============================================================
def bulk_update_quotes(
    assets: Iterable,
    period: str = "2y",
    interval: str = "1d",
    progress_cb: ProgressCB = None,
    use_stooq: bool = False,  # mantido na assinatura para compat, ignorado
) -> tuple[int, int]:
    """
    Atualiza cotações por ATIVO usando MT5QuoteProvider como fonte de histórico.
    Retorna: (n_ativos_com_insercao, n_linhas_inseridas)
    """
    provider = MT5QuoteProvider()
    assets = list(assets)
    total_assets = len(assets)
    today = timezone.localdate()
    if progress_cb:
        progress_cb("start", 0, total_assets, "starting", 0)

    bulk_objs: list[QuoteDaily] = []
    total_rows = 0
    assets_with_inserts = 0

    def _flush_bulk():
        if not bulk_objs:
            return
        QuoteDaily.objects.bulk_create(
            bulk_objs,
            ignore_conflicts=True,
            batch_size=BULK_BATCH_SIZE,
        )
        bulk_objs.clear()

    # Converte período para dias aproximados (2y -> 730, 6mo -> 180, etc.)
    def _period_to_days(label: str) -> int:
        text = (label or "").lower()
        if text.endswith("y"):
            try:
                return int(float(text[:-1]) * 365)
            except Exception:
                return 730
        if text.endswith("mo"):
            try:
                return int(float(text[:-2]) * 30)
            except Exception:
                return 180
        return 730

    lookback_days = _period_to_days(period)

    for idx, asset in enumerate(assets, start=1):
        ticker = getattr(asset, "ticker", "").strip().upper()
        if not ticker:
            if progress_cb:
                progress_cb("", idx, total_assets, "skip_invalid", 0)
            continue

        if progress_cb:
            progress_cb(ticker, idx, total_assets, "processing", 0)

        # garante placeholder provisório para hoje (se negociável e ainda não existir)
        ensure_today_placeholder(asset)

        last_dt = QuoteDaily.objects.filter(asset=asset).aggregate(Max("date"))["date__max"]
        start_dt = (last_dt - timedelta(days=INCREMENTAL_LOOKBACK_DAYS)) if last_dt else (today - timedelta(days=lookback_days))
        end_dt = today + timedelta(days=1)

        inserted_for_asset = 0

        try:
            series = provider.get_history(ticker=ticker, start=start_dt, end=end_dt, timeframe="D")
        except Exception as exc:
            print(f"[mt5] erro ao obter historico {ticker}: {exc}")
            series = None

        if series is not None and isinstance(series, pd.Series) and not series.empty:
            series = series.dropna()
            if last_dt:
                series = series[series.index > last_dt]
            for dt, px in series.items():
                try:
                    bulk_objs.append(QuoteDaily(asset=asset, date=dt, close=float(px)))
                    inserted_for_asset += 1
                    if len(bulk_objs) >= BULK_BATCH_SIZE:
                        _flush_bulk()
                except Exception:
                    pass

        if inserted_for_asset > 0:
            total_rows += inserted_for_asset
            assets_with_inserts += 1
            if progress_cb:
                progress_cb(ticker, idx, total_assets, "ok", inserted_for_asset)
        else:
            # nenhuma fonte trouxe dado algum -> loga
            try:
                MissingQuoteLog.objects.create(
                    asset=asset,
                    reason="no_data",
                    detail=f"Nenhum dado retornado pelo MT5 para {ticker}",
                )
            except Exception:
                pass
            if progress_cb:
                progress_cb(ticker, idx, total_assets, "no_data", 0)

    _flush_bulk()

    if progress_cb:
        progress_cb("done", total_assets, total_assets, "done", total_rows)

    return assets_with_inserts, total_rows


# ============================================================
# Preço em "tempo real" (via MT5)
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
    Atualiza apenas a cotação ao vivo (QuoteLive) com dados externos (ex.: MT5)
    e retorna o preço salvo. NÃO grava mais em QuoteDaily aqui.
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


def persist_today_from_live_quotes(assets=None) -> tuple[int, int]:
    """
    Lê as cotações em QuoteLive e grava/atualiza QuoteDaily para a data de hoje.

    - Se 'assets' for None, usa Asset ativos.
    - Para cada asset com QuoteLive:
        * date = timezone.localdate()
        * close = live.price (ou live.last como fallback)
        * cria/atualiza QuoteDaily(asset, date, close, is_provisional=False)
    - Retorna (n_ativos_atualizados, n_linhas_afetadas).
    """
    if assets is None:
        try:
            assets = Asset.objects.filter(is_active=True)
        except Exception:
            assets = Asset.objects.all()

    today = timezone.localdate()
    assets_count = 0
    rows_count = 0

    for asset in assets:
        live = QuoteLive.objects.filter(asset=asset).first()
        if not live:
            continue

        close = _safe_float(getattr(live, "price", None)) or _safe_float(getattr(live, "last", None))
        if close is None:
            continue

        QuoteDaily.objects.update_or_create(
            asset=asset,
            date=today,
            defaults={"close": close, "is_provisional": False},
        )
        assets_count += 1
        rows_count += 1

    return assets_count, rows_count


def fetch_latest_price(ticker: str) -> Optional[float]:
    """
    Retorna o último preço disponível do MT5 (ticker B3).
    """
    provider = MT5QuoteProvider()
    try:
        return provider.get_latest_price(ticker)
    except Exception as exc:
        print(f"[mt5] erro latest price {ticker}: {exc}")
        return None


def update_live_quotes(assets: Iterable, progress_cb: ProgressCB = None) -> tuple[int, int]:
    """
    Atualiza (ou cria) cotações em tempo real (tabela QuoteLive) via MT5.
    """
    if not getattr(settings, "USE_MT5_LIVE", True):
        total = len(list(assets))
        if progress_cb:
            progress_cb("mt5_live_disabled", total, total, "disabled", 0)
        return 0, total

    assets = list(assets)
    total = len(assets)
    updated = 0

    for idx, asset in enumerate(assets, start=1):
        ticker = getattr(asset, "ticker", "").strip().upper()
        if not ticker:
            continue

        if progress_cb:
            progress_cb(ticker, idx, total, "processing_live", 0)

        px = fetch_latest_price(ticker)
        if px is None:
            if progress_cb:
                progress_cb(ticker, idx, total, "no_data", 0)
            continue

        QuoteLive.objects.update_or_create(asset=asset, defaults={"price": px, "last": px, "source": "mt5"})
        updated += 1

        if progress_cb:
            progress_cb(ticker, idx, total, "ok", 1)

    if progress_cb:
        progress_cb("done", total, total, "done", updated)

    return updated, total


# ============================================================
# Utilitário opcional (teste rápido de um ativo)
# ============================================================
def update_single_asset(ticker_b3: str, period: str = "2y", interval: str = "1d") -> tuple[int, int]:
    """
    Atualiza um único ticker (string) sem precisar montar queryset.
    Útil para depuração pontual no shell.
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
from django.db import IntegrityError


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
    # Sem Yahoo/Stooq: apenas informa que nao ha mecanismo automático.
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
    - Ignora ativos com use_mt5=True (tratados via MT5/DailyPrice).
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
    """Sem Yahoo/Stooq, não há preenchimento automático para datas isoladas."""
    if getattr(asset, "use_mt5", False):
        return False
    return False


# -------------------------------------------------------------------
# Feridos/úteis (mantidos do código original)
# -------------------------------------------------------------------
from django.utils.timezone import make_naive

# ---------- FERIADOS B3 DINÂMICOS (qualquer ano) ----------
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
    """Dias de negociação: seg-sex excluindo feriados B3 (qualquer ano)."""
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
