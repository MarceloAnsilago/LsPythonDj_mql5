from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Tuple

from django.conf import settings
from django.http import HttpResponse, JsonResponse
from django.utils import timezone
from django.utils.dateparse import parse_datetime, parse_date
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from acoes.models import Asset
from cotacoes.models import QuoteLive, QuoteDaily
from longshort.services.price_provider import get_daily_prices
from longshort.services.quotes import apply_live_quote
from .models import LiveTick
from longshort.services.metrics import compute_pair_window_metrics, calcular_proporcao_long_short
from pairs.constants import DEFAULT_BASE_WINDOW, DEFAULT_ZSCORE_ABS_MIN, DEFAULT_HALF_LIFE_MAX
from pairs.models import Pair


def _client_ip(request) -> str | None:
    forwarded = request.META.get("HTTP_X_FORWARDED_FOR")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.META.get("REMOTE_ADDR")


def _auth_error(message: str, status: int = 401) -> JsonResponse:
    return JsonResponse({"ok": False, "detail": message}, status=status)


def _ensure_api_auth(request):
    expected_key = getattr(settings, "MT5_API_KEY", None)
    provided_key = request.headers.get("X-API-KEY") or request.META.get("HTTP_X_API_KEY")
    if expected_key and expected_key.strip():
        if not provided_key or provided_key.strip() != expected_key.strip():
            return _auth_error("API key invalida.", status=401)
    allowed_ips = getattr(settings, "MT5_ALLOWED_IPS", []) or []
    if allowed_ips:
        ip = _client_ip(request)
        if ip and ip not in allowed_ips:
            return _auth_error("IP nao autorizado.", status=403)
    return None


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_as_of(raw) -> timezone.datetime:
    if raw is None:
        return timezone.now()
    if isinstance(raw, datetime):
        dt = raw
    else:
        dt = parse_datetime(str(raw))
    if dt is None:
        return timezone.now()
    if timezone.is_naive(dt):
        try:
            return timezone.make_aware(dt, timezone.utc)
        except Exception:
            return timezone.now()
    return dt


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
    quote = getattr(asset, "live_quote", None)
    if not quote:
        quote = QuoteLive.objects.filter(asset=asset).first()

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


@csrf_exempt
@require_http_methods(["GET"])
def stream_assets(request):
    auth_error = _ensure_api_auth(request)
    if auth_error:
        return auth_error

    qs = Asset.objects.filter(is_active=True)
    qs = qs.filter(use_mt5=True)

    tickers = [a.ticker.strip().upper() for a in qs if getattr(a, "ticker", "").strip()]
    body = "\n".join(sorted(set(tickers)))
    return HttpResponse(body, content_type="text/plain; charset=utf-8")


@csrf_exempt
@require_http_methods(["POST"])
def push_live_quote(request):
    auth_error = _ensure_api_auth(request)
    if auth_error:
        return auth_error

    try:
        body = request.body.decode("utf-8") if request.body else "{}"
        data = json.loads(body or "{}")
    except json.JSONDecodeError:
        return JsonResponse({"ok": False, "detail": "JSON invalido."}, status=400)

    asset = _resolve_asset(data)
    if not asset:
        return JsonResponse({"ok": False, "detail": "Ativo nao encontrado."}, status=404)

    bid = _safe_float(data.get("bid"))
    ask = _safe_float(data.get("ask"))
    last = _safe_float(data.get("last") or data.get("price"))

    price = last
    if price is None and bid is not None and ask is not None:
        price = (bid + ask) / 2.0
    if price is None:
        return JsonResponse({"ok": False, "detail": "Preco invalido ou ausente."}, status=400)

    as_of = _parse_as_of(data.get("as_of"))
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

    LiveTick.objects.create(
        ticker=asset.ticker,
        bid=bid,
        ask=ask,
        last=recorded_last,
    )

    return JsonResponse(
        {
            "ok": True,
            "asset_id": asset.id,
            "ticker": asset.ticker,
            "as_of": as_of.isoformat(),
            "received_at": timezone.now().isoformat(),
        }
    )


@csrf_exempt
@require_http_methods(["POST"])
def push_daily_candle(request):
    """
    Recebe OHLC D1 fechado do MT5 e grava em QuoteDaily.
    Rejeita datas de hoje ou futuras.
    """
    auth_error = _ensure_api_auth(request)
    if auth_error:
        return auth_error

    try:
        body = request.body.decode("utf-8") if request.body else "{}"
        data = json.loads(body or "{}")
    except json.JSONDecodeError:
        return JsonResponse({"ok": False, "detail": "JSON invalido."}, status=400)

    asset = _resolve_asset(data)
    if not asset or not getattr(asset, "use_mt5", False):
        return JsonResponse({"ok": False, "detail": "Ativo nao encontrado ou nao usa MT5."}, status=404)

    ticker = asset.ticker.strip().upper()
    target_date = parse_date(str(data.get("date")))
    if target_date is None:
        return JsonResponse({"ok": False, "detail": "Data invalida."}, status=400)
    if target_date >= timezone.localdate():
        return JsonResponse({"ok": False, "detail": "Apenas candles D1 fechados (date < hoje)."}, status=400)

    c = _safe_float(data.get("close") or data.get("last") or data.get("price"))
    if c is None:
        return JsonResponse({"ok": False, "detail": "Preco de fechamento obrigatorio."}, status=400)

    QuoteDaily.objects.update_or_create(
        asset=asset,
        date=target_date,
        defaults={"close": c, "is_provisional": False},
    )

    return JsonResponse(
        {
            "ok": True,
            "ticker": ticker,
            "date": target_date.isoformat(),
            "source": "mt5_push",
        }
    )


@csrf_exempt
@require_http_methods(["GET"])
def get_signal(request):
    auth_error = _ensure_api_auth(request)
    if auth_error:
        return auth_error

    pair_id_raw = request.GET.get("pair_id")
    try:
        pair_id = int(pair_id_raw)
    except (TypeError, ValueError):
        return JsonResponse({"ok": False, "detail": "pair_id obrigatorio."}, status=400)

    pair = (
        Pair.objects.select_related("left", "right")
        .filter(pk=pair_id)
        .first()
    )
    if not pair:
        return JsonResponse({"ok": False, "detail": "Par nao encontrado."}, status=404)

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

    metrics = compute_pair_window_metrics(pair=pair, window=window)
    zscore = metrics.get("zscore") if isinstance(metrics, dict) else None
    half_life = metrics.get("half_life") if isinstance(metrics, dict) else None

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

    response = {
        "ok": True,
        "pair": {"id": pair.id, "left": pair.left.ticker, "right": pair.right.ticker},
        "prices": {
            "left": left_price,
            "right": right_price,
            "left_source": left_source,
            "right_source": right_source,
        },
        "metrics": metrics if isinstance(metrics, dict) else {},
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
