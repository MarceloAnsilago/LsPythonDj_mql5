# longshort/services/quotes.py
from __future__ import annotations

import io
from typing import Iterable, Optional, Callable

import pandas as pd
import requests
import yfinance as yf
from django.conf import settings
from django.db import transaction
from django.db.models import Max
from django.utils import timezone

from cotacoes.models import QuoteDaily, MissingQuoteLog

# -----------------------
# Progresso (callback)
# -----------------------
# assinatura: (ticker, idx1, total, status, rows_inserted)
ProgressCB = Optional[Callable[[str, int, int, str, int], None]]


# ============================================================
# 🔵 Helpers de fonte de dados
# ============================================================
def _yf_symbol(ticker_b3: str) -> str:
    """
    Retorna o símbolo correto para o Yahoo Finance,
    garantindo apenas um sufixo '.SA'.
    """
    t = (ticker_b3 or "").strip().upper()
    return t if t.endswith(".SA") else f"{t}.SA"


def _yf_close_series(df: Optional[pd.DataFrame]) -> Optional[pd.Series]:
    """
    Normaliza DataFrame do yfinance e devolve uma Series com 'Close' (ou 'Adj Close'):
      - Trata MultiIndex em que o 1º nível são campos ('Close','Open',...) e o 2º nível é o ticker
      - Ou achata para colunas simples e procura 'Close'/'Adj Close'
      - Converte o índice para date (datetime.date)
    """
    if df is None or not isinstance(df, pd.DataFrame) or getattr(df, "empty", True):
        return None

    if isinstance(df.columns, pd.MultiIndex):
        # Caso clássico do yfinance: 1º nível = ('Close','Open',...), 2º nível = tickers
        level0 = df.columns.get_level_values(0)
        # Prioridade: 'Close', depois 'Adj Close'
        if "Close" in set(level0):
            sub = df["Close"]
        elif "Adj Close" in set(level0):
            sub = df["Adj Close"]
        else:
            # fallback: achata pegando último nível (ticker), e depois tenta 'Close'
            flat = df.copy()
            flat.columns = flat.columns.get_level_values(-1)
            if "Close" in flat.columns:
                sub = flat["Close"]
            elif "Adj Close" in flat.columns:
                sub = flat["Adj Close"]
            else:
                return None

        # sub pode ser DataFrame (vários tickers) ou Series (um ticker só)
        if isinstance(sub, pd.DataFrame):
            # pega a primeira coluna (único ticker no seu caso)
            if sub.shape[1] == 0:
                return None
            s = sub.iloc[:, 0].dropna()
        else:
            s = sub.dropna()

        if s.empty:
            return None

        s.index = pd.to_datetime(s.index).date
        return s

    # Colunas simples (não-MultiIndex)
    cols = list(df.columns)
    col = "Close" if "Close" in cols else ("Adj Close" if "Adj Close" in cols else None)
    if col is None:
        return None

    s = df[col].dropna().copy()
    if s.empty:
        return None
    s.index = pd.to_datetime(s.index).date
    return s


CLOSE_UPDATE_TOLERANCE = 1e-6
INCREMENTAL_LOOKBACK_DAYS = 5  # dias de folga ao baixar de forma incremental
BULK_BATCH_SIZE = 1000  # flush para nao acumular objetos em memoria


def _safe_float(value: object) -> float | None:
    if getattr(settings, "USE_MT5_LIVE", False):
        return None
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _series_value_for_date(series: Optional[pd.Series], target_date) -> float | None:
    if series is None or target_date is None:
        return None
    val = series.get(target_date)
    if pd.isna(val):
        return None
    return val


def _update_daily_quote_close_if_changed(asset, quote_date, new_close) -> bool:
    if getattr(asset, "use_mt5", False):
        return False
    new_value = _safe_float(new_close)
    if new_value is None:
        return False
    queryset = QuoteDaily.objects.filter(asset=asset, date=quote_date)
    row = queryset.first()
    if row is None:
        return False

    if abs(row.close - new_value) <= CLOSE_UPDATE_TOLERANCE:
        # Mesmo preço: apenas marca como definitivo se era provisório
        if row.is_provisional:
            queryset.update(is_provisional=False)
        return False

    queryset.update(close=new_value, is_provisional=False)
    return True


def ensure_today_placeholder(asset) -> None:
    if getattr(asset, "use_mt5", False):
        return
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


def fetch_stooq_df(ticker: str) -> Optional[pd.DataFrame]:
    """
    Retorna DataFrame diário do Stooq para ticker B3 (ex: 'PETR4') ou None.
    Nota: Stooq pode ficar lento/indisponível. Timeout curto para não travar.
    """
    try:
        t = f"{ticker.lower()}.sa"
        url = f"https://stooq.com/q/d/l/?s={t}&i=d"
        r = requests.get(url, timeout=4)  # timeout curto para não travar shell/servidor
        if r.status_code == 200 and "Date,Open,High,Low,Close,Volume" in r.text:
            df = pd.read_csv(io.StringIO(r.text), parse_dates=["Date"])
            df.set_index("Date", inplace=True)
            return df
    except Exception as e:
        print(f"[stooq] erro {ticker}: {e}")
    return None


# ============================================================
# 🟢 Atualização diária (Yahoo principal, Stooq opcional)
# ============================================================
def bulk_update_quotes(
    assets: Iterable,
    period: str = "2y",
    interval: str = "1d",
    progress_cb: ProgressCB = None,
    use_stooq: bool = False,  # desligado por padrão para não travar
) -> tuple[int, int]:
    """
    Atualiza cotações por ATIVO:
      1) Yahoo Finance (principal)
      2) (Opcional) Stooq como fallback se nada inserido e use_stooq=True
      3) Só loga MissingQuote quando NENHUMA fonte trouxe dado algum
      4) Em caso de 'up_to_date' (sem novas datas), NÃO loga

    Retorna: (n_ativos_com_insercao, n_linhas_inseridas)
    """
    if not getattr(settings, "USE_YAHOO_HISTORY", False):
        if progress_cb:
            progress_cb("disabled", 0, 0, "yahoo_disabled", 0)
        return 0, 0

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

    for idx, asset in enumerate(assets, start=1):
        ticker = getattr(asset, "ticker", "").strip().upper()
        if not ticker:
            if progress_cb:
                progress_cb("", idx, total_assets, "skip_invalid", 0)
            continue
        if getattr(asset, "use_mt5", False):
            # Ativo controlado pelo MT5: não tenta atualizar via Yahoo/Stooq.
            if progress_cb:
                progress_cb(ticker, idx, total_assets, "mt5_mode", 0)
            continue

        if progress_cb:
            progress_cb(ticker, idx, total_assets, "processing", 0)

        # garante placeholder provisório para hoje (se negociável e ainda não existir)
        ensure_today_placeholder(asset)

        # última data gravada para filtrar incrementalmente
        last_dt = QuoteDaily.objects.filter(asset=asset).aggregate(Max("date"))["date__max"]

        inserted_for_asset = 0
        had_any_source_data = False

        # ---- 1) YAHOO (principal) ----
        try:
            yf_kwargs = dict(
                tickers=_yf_symbol(ticker),
                interval=interval,
                auto_adjust=False,   # mantem compat com seu pipeline
                progress=False,
                threads=False,
                group_by="column",   # ajuda a padronizar colunas
            )
            if last_dt:
                start_dt = last_dt - timedelta(days=INCREMENTAL_LOOKBACK_DAYS)
                yf_kwargs["start"] = start_dt.isoformat()
                yf_kwargs["end"] = (today + timedelta(days=1)).isoformat()
            else:
                yf_kwargs["period"] = period

            df_yf = yf.download(**yf_kwargs)
            s_close = _yf_close_series(df_yf)
            if s_close is not None:
                had_any_source_data = True
                if last_dt == today:
                    today_price = _series_value_for_date(s_close, today)
                    if today_price is not None:
                        _update_daily_quote_close_if_changed(asset, today, today_price)
                s = s_close
                if last_dt:
                    s = s[s.index > last_dt]
                if not s.empty:
                    for dt, px in s.items():
                        if pd.isna(px):
                            continue
                        try:
                            bulk_objs.append(QuoteDaily(asset=asset, date=dt, close=float(px)))
                            inserted_for_asset += 1
                            if len(bulk_objs) >= BULK_BATCH_SIZE:
                                _flush_bulk()
                        except Exception:
                            # ignora erro pontual ao criar o objeto
                            pass
        except Exception as e:
            print(f"[yfinance] erro {ticker}: {e}")

        # ---- 2) STQOOQ (fallback opcional) ----
        if inserted_for_asset == 0 and use_stooq:
            try:
                df_stq = fetch_stooq_df(ticker)
                if isinstance(df_stq, pd.DataFrame) and not getattr(df_stq, "empty", True):
                    had_any_source_data = True
                    s = df_stq["Close"].dropna().copy()
                    s.index = pd.to_datetime(s.index).date
                    if last_dt == today:
                        today_price = _series_value_for_date(s, today)
                        if today_price is not None:
                            _update_daily_quote_close_if_changed(asset, today, today_price)
                    if last_dt:
                        s = s[s.index > last_dt]
                    if not s.empty:
                        for dt, px in s.items():
                            try:
                                bulk_objs.append(QuoteDaily(asset=asset, date=dt, close=float(px)))
                                inserted_for_asset += 1
                                if len(bulk_objs) >= BULK_BATCH_SIZE:
                                    _flush_bulk()
                            except Exception:
                                pass
            except Exception as e:
                print(f"[stooq] exceção {ticker}: {e}")

        # ---- 3) Contabiliza / Progresso / Logs ----
        if inserted_for_asset > 0:
            total_rows += inserted_for_asset
            assets_with_inserts += 1
            if progress_cb:
                progress_cb(ticker, idx, total_assets, "ok", inserted_for_asset)
        else:
            if had_any_source_data:
                # havia dados, mas todos já estavam gravados -> up_to_date
                if progress_cb:
                    progress_cb(ticker, idx, total_assets, "up_to_date", 0)
            else:
                # nenhuma fonte trouxe dado algum -> logar (não bloqueia)
                try:
                    MissingQuoteLog.objects.create(
                        asset=asset,
                        reason="no_data",
                        detail=f"Nenhum dado retornado pelo Yahoo/Stooq para {ticker}",
                    )
                except Exception:
                    pass
                if progress_cb:
                    progress_cb(ticker, idx, total_assets, "no_data", 0)

    # ---- 4) Persistência em lote ----
    _flush_bulk()

    if progress_cb:
        progress_cb("done", total_assets, total_assets, "done", total_rows)

    return assets_with_inserts, total_rows


# ============================================================
# 🟣 Preço em "tempo real" (5m, ~15 min de delay típico no Yahoo)
# ============================================================
from cotacoes.models import QuoteLive


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
    Atualiza QuoteLive com dados externos (ex.: MT5) e retorna o preÇõ salvo.
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
    QuoteLive.objects.update_or_create(asset=asset, defaults=defaults)
    return px


def fetch_latest_price(ticker: str) -> Optional[float]:
    """
    Retorna o último preço (quase em tempo real) do Yahoo Finance.
    Intervalo de 5m, atraso típico de ~15 minutos.
    """
    if getattr(settings, "USE_MT5_LIVE", False) or not getattr(settings, "USE_YAHOO_HISTORY", False):
        return None
    try:
        sym = _yf_symbol(ticker)
        df = yf.download(
            tickers=sym,
            period="1d",
            interval="5m",
            progress=False,
            threads=False,
        )
        s_close = _yf_close_series(df)
        if s_close is not None and not s_close.empty:
            return float(s_close.iloc[-1])
    except Exception as e:
        print(f"[live] erro {ticker}: {e}")
    return None


def update_live_quotes(assets: Iterable, progress_cb: ProgressCB = None) -> tuple[int, int]:
    """
    Atualiza (ou cria) cotações em tempo real (tabela QuoteLive).
    """
    if getattr(settings, "USE_MT5_LIVE", False):
        total = len(list(assets))
        if progress_cb:
            progress_cb("mt5_live_mode", total, total, "mt5_only", 0)
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

        QuoteLive.objects.update_or_create(asset=asset, defaults={"price": px})
        updated += 1

        if progress_cb:
            progress_cb(ticker, idx, total, "ok", 1)

    if progress_cb:
        progress_cb("done", total, total, "done", updated)

    return updated, total


# ============================================================
# 🧪 Utilitário opcional (teste rápido de um ativo)
# ============================================================
def update_single_asset(ticker_b3: str, period: str = "2y", interval: str = "1d") -> tuple[int, int]:
    """
    Atualiza um único ticker (string) sem precisar montar queryset.
    Útil para depuração pontual no shell.
    """
    if not getattr(settings, "USE_YAHOO_HISTORY", False):
        return 0, 0
    from acoes.models import Asset
    asset = Asset.objects.filter(ticker=ticker_b3.upper()).first()
    if not asset:
        raise ValueError(f"Ativo {ticker_b3} não encontrado")

    def _p(t, i, tot, st, rows):  # progress minimalista
        print(f"[{i}/{tot}] {t} -> {st} ({rows})")

    return bulk_update_quotes([asset], period=period, interval=interval, progress_cb=_p, use_stooq=False)

# ============================================================
# 🔎 Scanner de buracos (faltantes) e tentativa de correção
# ============================================================

from datetime import date, timedelta
import pandas as pd
from django.db.models import Min, Max
from acoes.models import Asset
from datetime import datetime
from django.utils.timezone import make_naive
from django.db import IntegrityError

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
        date(year, 1, 1),    # Confraternização
        date(year, 4, 21),   # Tiradentes
        date(year, 5, 1),    # Dia do Trabalho
        date(year, 9, 7),    # Independência
        date(year,10,12),    # N. Sra. Aparecida
        date(year,11,2),     # Finados
        date(year,11,15),    # Proclamação da República
        date(year,11,20),    # Consciência Negra (B3 tem fechado)
        date(year,12,25),    # Natal
        date(year,12,24),    # Véspera (B3 não abre)
        date(year,12,31),    # Véspera (B3 não abre)
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
    """Dias de negociação: seg–sex excluindo feriados B3 (qualquer ano)."""
    if start > end:
        return []
    rng = pd.date_range(start, end, freq="B")
    holidays = b3_holidays_between(start, end)
    return [d.date() for d in rng if d.date() not in holidays]

# ---------- INTERVALOS A IGNORAR (halts/eventos por ticker) ----------
# preencha conforme necessário; exemplo para BRFS3:
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

# ---------- SCANNER ----------
def find_missing_dates_for_asset(
    asset,
    *,
    since_months: int | None = None,
    until: date | None = None,
) -> list[date]:
    if getattr(asset, "use_mt5", False):
        return []
    """
    Datas faltantes (dias de negociação) em QuoteDaily para o ativo.
    - Se since_months for fornecido, limita a janela aos últimos N meses.
    - until padrão: hoje.
    - Ignora feriados B3 e intervalos por ticker (IGNORED_RANGES).
    """
    qs = QuoteDaily.objects.filter(asset=asset)

    # Sem nenhum dado: não tratamos como "buraco" (inicialização pelo botão principal)
    bounds = qs.aggregate(min_dt=Min("date"), max_dt=Max("date"))
    min_dt, _ = bounds["min_dt"], bounds["max_dt"]
    if not min_dt:
        return []

    if until is None:
        until = date.today()

    start = min_dt
    if since_months:
        # Janela deslizante: últimos N meses
        approx_days = int(since_months * 30.44)
        start = max(min_dt, until - timedelta(days=approx_days))

    expected = set(_business_days(start, until))

    # remove dias ignorados específicos do ticker
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
    if getattr(asset, "use_mt5", False):
        return 0, []
    """
    Tenta preencher datas faltantes para um ativo.
    Baixa um bloco (Yahoo; Stooq opcional) e insere apenas as faltantes.
    Retorna: (n_inseridos, restantes).
    """
    if not getattr(settings, "USE_YAHOO_HISTORY", False):
        return 0, sorted(missing_dates)

    if not missing_dates:
        return 0, []

    lo = min(missing_dates) - timedelta(days=2)
    hi = max(missing_dates) + timedelta(days=2)

    remaining = set(missing_dates)
    to_insert: list[QuoteDaily] = []
    inserted = 0

    # ---- Yahoo ----
    try:
        df = yf.download(
            tickers=_yf_symbol(getattr(asset, "ticker", "").upper()),
            start=lo.isoformat(),
            end=(hi + timedelta(days=1)).isoformat(),  # end exclusivo
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
            group_by="column",
        )
        s_close = _yf_close_series(df)
        if s_close is not None and not s_close.empty:
            for dt, px in s_close.items():
                if dt in remaining and pd.notna(px):
                    try:
                        to_insert.append(QuoteDaily(asset=asset, date=dt, close=float(px)))
                        remaining.discard(dt)
                    except Exception:
                        remaining.discard(dt)
    except Exception as e:
        print(f"[faltantes][yahoo] {asset} erro: {e}")

    # ---- Stooq (opcional, só se nada entrou via Yahoo) ----
    if not to_insert and use_stooq and remaining:
        try:
            df_stq = fetch_stooq_df(getattr(asset, "ticker", ""))
            if isinstance(df_stq, pd.DataFrame) and not getattr(df_stq, "empty", True):
                s = df_stq["Close"].dropna().copy()
                s.index = pd.to_datetime(s.index).date
                for dt in list(remaining):
                    if dt in s.index:
                        try:
                            to_insert.append(QuoteDaily(asset=asset, date=dt, close=float(s.loc[dt])))
                            remaining.discard(dt)
                        except Exception:
                            remaining.discard(dt)
        except Exception as e:
            print(f"[faltantes][stooq] {asset} erro: {e}")

    # ---- Persistência em lote ----
    if to_insert:
        try:
            QuoteDaily.objects.bulk_create(to_insert, ignore_conflicts=True, batch_size=1000)
            inserted = len(to_insert)
        except Exception as e:
            # fallback caso algum banco não suporte ignore_conflicts
            print(f"[faltantes][bulk_create] erro: {e}")
            inserted = 0

    # ---- Logs simpáticos ----
    try:
        if inserted > 0:
            MissingQuoteLog.objects.create(
                asset=asset,
                reason="gap_fix",
                detail=f"Preenchidos {inserted} buraco(s) pelo scanner.",
                resolved_bool=True,
            )
        if remaining:
            # mostra só as 12 primeiras pra não poluir
            tail = sorted(remaining)[:12]
            MissingQuoteLog.objects.create(
                asset=asset,
                reason="gap_remaining",
                detail=f"Restando {len(remaining)} data(s): {tail}{' ...' if len(remaining) > 12 else ''}",
                resolved_bool=False,
            )
    except Exception:
        pass

    return inserted, sorted(remaining)

def scan_all_assets_and_fix(
    *,
    use_stooq: bool = False,
    since_months: int | None = None,
    tickers: list[str] | None = None,
):
    """
    Varre ativos, tenta corrigir buracos e retorna lista serializavel:
    [{ticker, missing_before, fixed, remaining:[YYYY-MM-DD,...]}]
    - since_months: limitar janela (p.ex. 18 = ultimos 18 meses).
    - tickers: filtrar um subconjunto (strings, ex.: ["BRFS3","PETR4"]).
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
    # Yahoo usa epoch (UTC) nos parâmetros period1/period2
    return int(datetime(d.year, d.month, d.day).timestamp())

def try_fetch_single_date(asset, d: date, *, use_stooq: bool = True) -> bool:
    """Tenta inserir apenas a data d para o ativo."""
    if getattr(asset, "use_mt5", False):
        return False
    if not getattr(settings, "USE_YAHOO_HISTORY", False):
        return False
    # 1) Yahoo
    try:
        df = yf.download(
            tickers=_yf_symbol(getattr(asset, "ticker", "").upper()),
            start=d.isoformat(),
            end=(d + timedelta(days=1)).isoformat(),
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
            group_by="column",
        )
        s_close = _yf_close_series(df)
        if s_close is not None and not s_close.empty and d in s_close.index:
            px = float(s_close.loc[d])
            QuoteDaily.objects.create(asset=asset, date=d, close=px)
            return True
    except IntegrityError:
        return True
    except Exception as e:
        print(f"[fetch_one][yahoo] {asset} {d} erro: {e}")

    # 2) Stooq (fallback)
    if use_stooq:
        try:
            df_stq = fetch_stooq_df(getattr(asset, "ticker", ""))
            if isinstance(df_stq, pd.DataFrame) and "Close" in df_stq.columns:
                s = df_stq["Close"].dropna().copy()
                s.index = pd.to_datetime(s.index).date
                if d in s.index:
                    QuoteDaily.objects.create(asset=asset, date=d, close=float(s.loc[d]))
                    return True
        except IntegrityError:
            return True
        except Exception as e:
            print(f"[fetch_one][stooq] {asset} {d} erro: {e}")

    return False
