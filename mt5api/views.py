from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Tuple

import logging
from django.conf import settings
from django.core.cache import cache
from django.http import HttpResponse, JsonResponse
from django.utils import timezone
from django.utils.dateparse import parse_datetime, parse_date
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from acoes.models import Asset
from cotacoes.models import QuoteLive
from longshort.services.price_provider import get_daily_prices
from longshort.services.quotes import apply_live_quote
from .models import LiveTick
from longshort.services.metrics import compute_pair_window_metrics, calcular_proporcao_long_short
from pairs.constants import DEFAULT_BASE_WINDOW, DEFAULT_ZSCORE_ABS_MIN, DEFAULT_HALF_LIFE_MAX
from pairs.models import Pair

logger = logging.getLogger(__name__)


def _client_ip(request) -> str | None:
    fly_ip = (request.META.get("HTTP_FLY_CLIENT_IP") or "").strip()
    if fly_ip:
        return fly_ip
    forwarded = request.META.get("HTTP_X_FORWARDED_FOR") or request.headers.get("X-Forwarded-For")
    if forwarded:
        parts = [p.strip() for p in str(forwarded).split(",") if p.strip()]
        for part in parts:
            if part.lower() != "unknown":
                return part
    return request.META.get("REMOTE_ADDR")


def _error_response(message: str, *, status: int = 400, code: str = "ERROR") -> JsonResponse:
    return JsonResponse({"ok": False, "error_code": code, "detail": message}, status=status)


def _auth_error(message: str, *, status: int = 401, code: str = "UNAUTHORIZED") -> JsonResponse:
    return _error_response(message, status=status, code=code)


def _ensure_api_auth(request):
    expected_key = getattr(settings, "MT5_API_KEY", None)
    provided_key = request.headers.get("X-API-KEY") or request.META.get("HTTP_X_API_KEY")
    if not settings.DEBUG and (not expected_key or not str(expected_key).strip()):
        return _auth_error("MT5_API_KEY nao configurada (DEBUG=False).", status=401, code="MISSING_API_KEY")
    if expected_key and str(expected_key).strip():
        if not provided_key or provided_key.strip() != expected_key.strip():
            return _auth_error("API key invalida.", status=401, code="UNAUTHORIZED")
    allowed_ips = getattr(settings, "MT5_ALLOWED_IPS", []) or []
    if allowed_ips:
        ip = _client_ip(request)
        if not ip:
            return _auth_error("IP nao autorizado.", status=403, code="FORBIDDEN_IP")
        if ip not in allowed_ips:
            return _auth_error("IP nao autorizado.", status=403, code="FORBIDDEN_IP")
    return None


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_tick_as_of(payload: dict) -> timezone.datetime:
    """
    Extrai timestamp do payload MT5:
    - time_msc: epoch em milissegundos
    - time: epoch em segundos
    - as_of/datetime: string ISO ou datetime
    Fallback: timezone.now()
    Retorna sempre timezone-aware (UTC).
    """
    now = timezone.now()
    if not isinstance(payload, dict):
        return now

    raw_time_msc = payload.get("time_msc")
    if raw_time_msc is not None:
        try:
            ts = float(raw_time_msc) / 1000.0
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        except Exception:
            pass

    raw_time = payload.get("time")
    if raw_time is not None:
        try:
            ts = float(raw_time)
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        except Exception:
            pass

    for key in ("as_of", "datetime"):
        raw = payload.get(key)
        if raw is None:
            continue
        dt_val = raw if isinstance(raw, datetime) else parse_datetime(str(raw))
        if dt_val is None:
            continue
        if timezone.is_naive(dt_val):
            try:
                dt_val = timezone.make_aware(dt_val, timezone.utc)
            except Exception:
                dt_val = None
        if dt_val is not None:
            return dt_val

    return now


def _resolve_asset(data: dict) -> Asset | None:
    asset_id = data.get("asset_id")
    ticker = (data.get("ticker") or "").strip().upper()
    qs = Asset.objects.all()
    if asset_id:
        asset = qs.filter(pk=asset_id).first()
        if asset:
            return asset
    if ticker:
        return qs.filter(ticker=ticker).first()
    return None


def _latest_daily_price(asset: Asset) -> Tuple[float | None, object | None]:
    prices = get_daily_prices(asset, end_date=timezone.localdate())
    if not prices:
        return None, None
    last_row = prices[-1]
    return _safe_float(last_row.get("close")), last_row.get("date")


def _current_price(asset: Asset) -> Tuple[float | None, object | None, str | None]:
    quote = (
        QuoteLive.objects.filter(asset=asset)
        .order_by("-as_of", "-updated_at")
        .first()
    )

    price = None
    as_of = None
    source = None
    if quote:
        price = _safe_float(quote.price)
        if price is None:
            price = _safe_float(quote.last)
        as_of = quote.as_of or quote.updated_at
        source = getattr(quote, "source", None) or "mt5"
    if price is None:
        daily_price, daily_dt = _latest_daily_price(asset)
        return daily_price, daily_dt, "daily"
    return price, as_of, source or "mt5"


@require_http_methods(["GET"])
def stream_assets(request):
    auth_error = _ensure_api_auth(request)
    if auth_error:
        return auth_error

    qs = Asset.objects.filter(is_active=True)
    qs = qs.filter(use_mt5=True)

    tickers = [a.ticker.strip().upper() for a in qs if getattr(a, "ticker", "").strip()]
    unique_tickers = sorted(set(tickers))
    body = "\n".join(unique_tickers)
    response = HttpResponse(body, content_type="text/plain; charset=utf-8")
    response["X-Assets-Count"] = str(len(unique_tickers))
    return response


@csrf_exempt
@require_http_methods(["POST"])
def push_live_quote(request):
    """
    Endpoint MT5: grava cotacao ao vivo.
    Auth: header X-API-KEY.
    Payload aceito para tempo: time_msc (ms epoch), time (s epoch), as_of/datetime (ISO).
    Sempre atualiza QuoteLive; LiveTick so grava respeitando o throttle
    MT5_LIVETICK_MIN_INTERVAL_MS (default 250ms) via cache per ticker.
    """
    auth_error = _ensure_api_auth(request)
    if auth_error:
        return auth_error

    try:
        body = request.body.decode("utf-8") if request.body else "{}"
        data = json.loads(body or "{}")
    except json.JSONDecodeError:
        return _error_response("JSON invalido.", status=400, code="INVALID_JSON")

    asset = _resolve_asset(data)
    if not asset:
        return _error_response("Ativo nao encontrado.", status=404, code="ASSET_NOT_FOUND")

    bid = _safe_float(data.get("bid"))
    ask = _safe_float(data.get("ask"))
    last = _safe_float(data.get("last") or data.get("price"))

    price = last
    if price is None and bid is not None and ask is not None:
        price = (bid + ask) / 2.0
    if price is None:
        return _error_response("Preco invalido ou ausente.", status=400, code="INVALID_PRICE")

    as_of = _parse_tick_as_of(data)
    source = (data.get("source") or "mt5").lower()

    recorded_last = last if last is not None else price

    apply_live_quote(
        asset,
        bid=bid,
        ask=ask,
        last=last,
        price=price,
        as_of=as_of,
        source=source,
    )

    # Throttle de persistencia em LiveTick para nao inflar DB
    min_interval_ms = getattr(settings, "MT5_LIVETICK_MIN_INTERVAL_MS", 250) or 0
    cache_key = f"livetick:last:{asset.ticker}"
    saved_tick = True
    if min_interval_ms > 0:
        last_ts = cache.get(cache_key)
        now_ms = timezone.now().timestamp() * 1000
        if last_ts and (now_ms - last_ts) < float(min_interval_ms):
            saved_tick = False
        else:
            cache.set(cache_key, now_ms, timeout=10)

    if saved_tick:
        LiveTick.objects.create(
            ticker=asset.ticker,
            bid=bid,
            ask=ask,
            last=recorded_last,
            as_of=as_of,
            source=source,
        )

    return JsonResponse(
        {
            "ok": True,
            "asset_id": asset.id,
            "ticker": asset.ticker,
            "as_of": as_of.isoformat(),
            "received_at": timezone.now().isoformat(),
            "price": price,
            "source": source,
            "saved_tick": bool(saved_tick),
            "throttled": not bool(saved_tick),
        }
    )


@csrf_exempt
@require_http_methods(["POST"])
def push_daily_candle(request):
    """
    Endpoint presente apenas para compatibilidade. Dados recebidos nao sao persistidos.
    QuoteDaily agora e mantido exclusivamente via Yahoo (dias < hoje).
    """
    auth_error = _ensure_api_auth(request)
    if auth_error:
        return auth_error

    try:
        body = request.body.decode("utf-8") if request.body else "{}"
        data = json.loads(body or "{}")
    except json.JSONDecodeError:
        return _error_response("JSON invalido.", status=400, code="INVALID_JSON")

    asset = _resolve_asset(data)
    if not asset or not getattr(asset, "use_mt5", False):
        return _error_response("Ativo nao encontrado ou nao usa MT5.", status=404, code="ASSET_NOT_FOUND")

    ticker = asset.ticker.strip().upper()
    target_date = parse_date(str(data.get("date")))
    if target_date is None:
        return _error_response("Data invalida.", status=400, code="INVALID_DATE")
    if target_date >= timezone.localdate():
        return _error_response("Apenas candles D1 fechados (date < hoje).", status=400, code="INVALID_DATE")

    c = _safe_float(data.get("close") or data.get("last") or data.get("price"))
    if c is None:
        return _error_response("Preco de fechamento obrigatorio.", status=400, code="INVALID_PRICE")

    return _error_response(
        "Persistência de candles D1 via MT5 está desativada; use Yahoo para manter QuoteDaily.",
        status=400,
        code="MT5_DAILY_DISABLED"
    )


@require_http_methods(["GET"])
def get_signal(request):
    try:
        auth_error = _ensure_api_auth(request)
        if auth_error:
            return auth_error

        pair_id_raw = request.GET.get("pair_id")
        try:
            pair_id = int(pair_id_raw)
        except (TypeError, ValueError):
            return _error_response("pair_id obrigatorio.", status=400, code="MISSING_PAIR_ID")

        pair = (
            Pair.objects.select_related("left", "right")
            .filter(pk=pair_id)
            .first()
        )
        if not pair:
            return _error_response("Par nao encontrado.", status=404, code="PAIR_NOT_FOUND")

        window_raw = request.GET.get("window")
        half_life_max = _safe_float(request.GET.get("half_life_max"))
        if half_life_max is None:
            half_life_max = getattr(settings, "MT5_HALF_LIFE_MAX", DEFAULT_HALF_LIFE_MAX or 0) or 0

        try:
            window = int(window_raw) if window_raw else None
        except (TypeError, ValueError):
            window = None
        if not window or window <= 0:
            window = pair.chosen_window or pair.base_window or DEFAULT_BASE_WINDOW

        zscore_abs_min = _safe_float(request.GET.get("zscore_abs_min"))
        if zscore_abs_min is None:
            zscore_abs_min = getattr(settings, "MT5_ZSCORE_ABS_MIN", DEFAULT_ZSCORE_ABS_MIN)

        exit_band = _safe_float(request.GET.get("exit_band"))
        if exit_band is None:
            exit_band = 0.5

        capital = _safe_float(request.GET.get("capital"))
        if capital is None:
            capital = getattr(settings, "MT5_DEFAULT_CAPITAL", 50000.0)
        lot_size = _safe_float(request.GET.get("lot_size"))
        try:
            lot_size_int = int(lot_size) if lot_size else 100
        except (TypeError, ValueError):
            lot_size_int = 100

        left_price, left_as_of, left_source = _current_price(pair.left)
        right_price, right_as_of, right_source = _current_price(pair.right)

        if left_price is None or right_price is None:
            return JsonResponse(
                {
                    "ok": True,
                    "action": "HOLD",
                    "detail": "Faltando cotacao ao vivo ou diaria.",
                    "pair": {"id": pair.id, "left": pair.left.ticker, "right": pair.right.ticker},
                }
            )

        cache_key = f"mt5api:pair_metrics:{pair.id}:{window}"
        metrics = cache.get(cache_key)
        if metrics is None:
            metrics = compute_pair_window_metrics(pair=pair, window=window)
            cache.set(cache_key, metrics, timeout=10)
        metrics_payload = metrics if isinstance(metrics, dict) else {}
        metrics_payload.setdefault("window", window)
        zscore = metrics_payload.get("zscore")
        half_life = metrics_payload.get("half_life")

        as_of = left_as_of or right_as_of or timezone.now()
        if left_as_of and right_as_of:
            as_of = min(left_as_of, right_as_of)

        action = "HOLD"
        detail = None
        plan_payload = None
        qty_short = None
        qty_long = None
        px_short = None
        px_long = None

        if zscore is None:
            detail = "Z-score indisponivel."
        else:
            abs_z = abs(zscore)
            if half_life_max and half_life and half_life > half_life_max:
                action = "HOLD"
                detail = "Half-life acima do limite."
            elif abs_z >= zscore_abs_min:
                if zscore >= 0:
                    action = "ENTER_SHORT_SPREAD"
                    px_short = left_price
                    px_long = right_price
                    plan = calcular_proporcao_long_short(
                        preco_short=left_price,
                        preco_long=right_price,
                        limite_venda=capital,
                        lote=lot_size_int,
                        ticker_short=pair.left.ticker,
                        ticker_long=pair.right.ticker,
                        nome_short=getattr(pair.left, "name", None),
                        nome_long=getattr(pair.right, "name", None),
                        capital_informado=capital,
                    )
                else:
                    action = "ENTER_LONG_SPREAD"
                    px_short = right_price
                    px_long = left_price
                    plan = calcular_proporcao_long_short(
                        preco_short=right_price,
                        preco_long=left_price,
                        limite_venda=capital,
                        lote=lot_size_int,
                        ticker_short=pair.right.ticker,
                        ticker_long=pair.left.ticker,
                        nome_short=getattr(pair.right, "name", None),
                        nome_long=getattr(pair.left, "name", None),
                        capital_informado=capital,
                    )
                if plan is None:
                    action = "HOLD"
                    detail = "Capital insuficiente para montar lote."
                else:
                    plan_payload = plan.to_payload()
                    qty_short = plan.quantidade_vendida
                    qty_long = plan.quantidade_comprada
            elif abs_z <= exit_band:
                action = "EXIT"
            else:
                action = "HOLD"
                detail = "Dentro da banda neutra."

        if getattr(settings, "MT5_DEBUG_LOG", False):
            logger.debug(
                "get_signal pair=%s action=%s z=%.4f hl=%s left=%s right=%s",
                pair.id,
                action,
                zscore if zscore is not None else float("nan"),
                half_life,
                left_price,
                right_price,
            )

        response = {
            "ok": True,
            "pair": {"id": pair.id, "left": pair.left.ticker, "right": pair.right.ticker},
            "prices": {
                "left": left_price,
                "right": right_price,
                "left_source": left_source,
                "right_source": right_source,
            },
            "metrics": metrics_payload,
            "action": action,
            "detail": detail,
            "qty_short": qty_short,
            "qty_long": qty_long,
            "px_short": px_short,
            "px_long": px_long,
            "capital": capital,
            "window": window,
            "zscore_abs_min": zscore_abs_min,
            "exit_band": exit_band,
            "as_of": as_of.isoformat() if hasattr(as_of, "isoformat") else str(as_of),
        }
        if plan_payload:
            response["plan"] = plan_payload
        return JsonResponse(response)
    except Exception as exc:
        logger.exception("Erro inesperado em get_signal: %s", exc)
        return _error_response("Erro interno.", status=500, code="SERVER_ERROR")
