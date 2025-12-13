from __future__ import annotations

from collections import defaultdict
from decimal import Decimal, InvalidOperation
from types import SimpleNamespace

from django.conf import settings
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.db.models import Prefetch, Q
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.template.loader import render_to_string
from django.urls import reverse
from django.utils import timezone
from django.utils.formats import number_format
from django.views.decorators.http import require_POST

from acoes.models import Asset
from cotacoes.models import QuoteLive
from longshort.services.price_provider import get_daily_prices
from longshort.services.metrics import (
    compute_pair_window_metrics,
    calcular_proporcao_long_short,
    get_zscore_series,
)
from longshort.services.quotes import fetch_latest_price, update_live_quotes
from pairs.constants import DEFAULT_BASE_WINDOW, DEFAULT_WINDOWS
from pairs.forms import UserMetricsConfigForm
from pairs.models import Pair, UserMetricsConfig
from operacoes.models import Operation, OperationMetricSnapshot


def _build_home_operations_payload(request):
    operations_cards: list[dict] = []

    def _fmt_money(value: Decimal | float | None) -> str:
        if value is None:
            return "--"
        try:
            return f"R$ {number_format(value, 2)}"
        except (TypeError, ValueError):
            return "--"

    def _fmt_int(value: int | Decimal | None) -> str:
        if value is None:
            return "--"
        try:
            return number_format(value, 0)
        except (TypeError, ValueError):
            return "--"

    def _fmt_metric(value: float | Decimal | None, digits: int = 2) -> str:
        if value is None:
            return "--"
        try:
            return f"{float(value):.{digits}f}"
        except (TypeError, ValueError):
            return "--"

    def _fmt_ratio(value: Decimal | float | None, digits: int = 2) -> str:
        if value is None:
            return "--"
        try:
            return f"{float(value):.{digits}f}"
        except (TypeError, ValueError):
            return "--"

    def _to_decimal(value: Decimal | float | int | None) -> Decimal:
        if value is None:
            return Decimal("0")
        if isinstance(value, Decimal):
            return value
        try:
            return Decimal(str(value))
        except (TypeError, ValueError, InvalidOperation):
            return Decimal("0")

    def _fmt_updated(dt_value):
        if not dt_value:
            return ""
        try:
            localized = timezone.localtime(dt_value)
        except Exception:
            localized = dt_value
        return localized.strftime("%d/%m %H:%M")

    def _format_days_open(dt_value):
        if not dt_value:
            return None
        try:
            opened = timezone.localtime(dt_value)
            now = timezone.localtime(timezone.now())
            delta = now - opened
            days = delta.days if delta.days >= 0 else 0
        except Exception:
            return None
        label = "dia" if days == 1 else "dias"
        return f"{days} {label}"

    live_price_cache: dict[str, tuple[Decimal | None, bool]] = {}
    manual_refresh_required = False
    use_mt5_live = getattr(settings, "USE_MT5_LIVE", True)

    def _normalize_ticker(value: str | None) -> str:
        return (value or "").strip().upper()

    def _try_fetch_live_price(ticker_norm: str) -> tuple[Decimal | None, bool]:
        if not ticker_norm:
            return None, False
        if not use_mt5_live:
            return None, False
        cached = live_price_cache.get(ticker_norm)
        if cached is not None:
            return cached
        price: Decimal | None = None
        error = False
        try:
            px = fetch_latest_price(ticker_norm)
            if px is not None:
                price = Decimal(str(px))
        except Exception:
            error = True
        live_price_cache[ticker_norm] = (price, error)
        return price, error

    def _refresh_live_price(
        asset: Asset | None, current_price: Decimal | None, current_updated, *, force: bool = False
    ):
        nonlocal manual_refresh_required
        if not asset:
            return current_price, current_updated
        if not getattr(asset, "use_mt5", False):
            return current_price, current_updated
        if not force and current_price is not None:
            return current_price, current_updated
        ticker_norm = _normalize_ticker(getattr(asset, "ticker", None))
        if not ticker_norm:
            return current_price, current_updated
        live_price, _ = _try_fetch_live_price(ticker_norm)
        if live_price is not None:
            current_price = live_price
            current_updated = timezone.now()
            try:
                QuoteLive.objects.update_or_create(asset=asset, defaults={"price": float(live_price)})
            except Exception:
                pass
        else:
            manual_refresh_required = True
        return current_price, current_updated

    money_quant = Decimal("0.01")

    operations_qs = (
        Operation.objects.select_related(
            "pair",
            "sell_asset__live_quote",
            "buy_asset__live_quote",
            "left_asset",
            "right_asset",
        )
        .prefetch_related(
            Prefetch(
                "metric_snapshots",
                queryset=OperationMetricSnapshot.objects.filter(snapshot_type=OperationMetricSnapshot.TYPE_OPEN)
                .order_by("-reference_date"),
                to_attr="entry_snapshots",
            ),
            Prefetch(
                "metric_snapshots",
                queryset=OperationMetricSnapshot.objects.filter(snapshot_type=OperationMetricSnapshot.TYPE_CURRENT)
                .order_by("-reference_date"),
                to_attr="current_snapshots",
            ),
        )
        .filter(user=request.user, status=Operation.STATUS_OPEN)
        .order_by("-opened_at")
    )

    for operation in operations_qs:
        entry_snapshot = (operation.entry_snapshots[0] if getattr(operation, "entry_snapshots", None) else None)
        entry_metrics_payload = {}
        entry_reference_label = None
        if entry_snapshot:
            entry_reference_label = entry_snapshot.reference_date.strftime("%d/%m/%Y") if entry_snapshot.reference_date else None
            if entry_snapshot.payload:
                entry_metrics_payload = entry_snapshot.payload
        elif isinstance(operation.pair_metrics, dict):
            entry_metrics_payload = operation.pair_metrics

        entry_zscore = entry_snapshot.zscore if entry_snapshot and entry_snapshot.zscore is not None else operation.entry_zscore

        pair_ref = operation.pair or SimpleNamespace(
            left=operation.left_asset,
            right=operation.right_asset,
        )

        current_snapshot = (
            operation.current_snapshots[0] if getattr(operation, "current_snapshots", None) else None
        )
        current_metrics_payload: dict[str, object] = {}
        current_zscore = None
        if current_snapshot and isinstance(current_snapshot.payload, dict):
            current_metrics_payload = current_snapshot.payload
            raw_z = current_metrics_payload.get("zscore")
            if raw_z is not None:
                try:
                    current_zscore = float(raw_z)
                except (TypeError, ValueError):
                    current_zscore = None
            # fallback para o campo salvo no snapshot caso o payload nao tenha o zscore
            if current_zscore is None and current_snapshot.zscore is not None:
                try:
                    current_zscore = float(current_snapshot.zscore)
                except (TypeError, ValueError):
                    current_zscore = None
        if current_zscore is None:
            try:
                metrics_now = compute_pair_window_metrics(pair=pair_ref, window=operation.window)
            except Exception:
                metrics_now = None
            if isinstance(metrics_now, dict):
                if not current_metrics_payload:
                    current_metrics_payload = metrics_now
                raw_z = metrics_now.get("zscore")
                if raw_z is not None:
                    try:
                        current_zscore = float(raw_z)
                    except (TypeError, ValueError):
                        current_zscore = None

        # Ultimo recurso: calcula zscore via serie se ainda estiver vazio
        if current_zscore is None:
            try:
                z_series = get_zscore_series(pair=pair_ref, window=operation.window)
            except Exception:
                z_series = []
            if z_series:
                last_point = z_series[-1]
                if last_point and len(last_point) >= 2:
                    try:
                        current_zscore = float(last_point[1])
                    except (TypeError, ValueError):
                        current_zscore = None

    def _build_live_price(asset):
        quote = getattr(asset, "live_quote", None) if asset else None
        price = None
        updated = None
        if quote:
            raw_price = quote.price if quote.price is not None else getattr(quote, "last", None)
            if raw_price is not None:
                try:
                    price = Decimal(str(raw_price))
                except (TypeError, ValueError):
                    price = None
            updated = getattr(quote, "as_of", None) or getattr(quote, "updated_at", None)
        return price, updated

    sell_live_price, sell_updated = _build_live_price(operation.sell_asset)
    sell_live_price, sell_updated = _refresh_live_price(
        operation.sell_asset, sell_live_price, sell_updated
    )
    buy_live_price, buy_updated = _build_live_price(operation.buy_asset)
    buy_live_price, buy_updated = _refresh_live_price(
        operation.buy_asset, buy_live_price, buy_updated
    )

    sell_qty_dec = Decimal(operation.sell_quantity)
    buy_qty_dec = Decimal(operation.buy_quantity)
    sell_pl = None
    if sell_live_price is not None:
        sell_pl = (operation.sell_price - sell_live_price) * sell_qty_dec
    buy_pl = None
    if buy_live_price is not None:
        buy_pl = (buy_live_price - operation.buy_price) * buy_qty_dec

    current_sell_total = None
    current_buy_total = None
    if sell_live_price is not None:
        current_sell_total = (sell_live_price * sell_qty_dec).quantize(money_quant)
    if buy_live_price is not None:
        current_buy_total = (buy_live_price * buy_qty_dec).quantize(money_quant)

    current_net_value = None
    if current_sell_total is not None and current_buy_total is not None:
        current_net_value = (current_sell_total - current_buy_total).quantize(money_quant)

    pl_total = None
    if sell_pl is not None and buy_pl is not None:
        pl_total = (sell_pl + buy_pl).quantize(money_quant)

    pnl_ready = sell_pl is not None and buy_pl is not None
    pnl_positive = pnl_ready and pl_total is not None and pl_total > 0
    pnl_negative = pnl_ready and pl_total is not None and pl_total < 0

    latest_update = max(dt for dt in (sell_updated, buy_updated) if dt is not None) if sell_updated or buy_updated else None

    z_delta_label = "--"
    is_delta_positive = False
    if entry_zscore is not None and current_zscore is not None:
        try:
            delta = float(current_zscore) - float(entry_zscore)
            z_delta_label = f"{delta:+.2f}"
            is_delta_positive = delta >= 0
        except (TypeError, ValueError):
            z_delta_label = "--"
            is_delta_positive = False

    entry_net_direction = ""
    if operation.net_value is not None:
        entry_net_direction = "recebe" if operation.net_value >= 0 else "paga"

    entry_prices = {
        "sell_qty_label": _fmt_int(operation.sell_quantity),
        "buy_qty_label": _fmt_int(operation.buy_quantity),
        "sell_price_label": _fmt_money(operation.sell_price),
        "buy_price_label": _fmt_money(operation.buy_price),
        "sell_total_label": _fmt_money(operation.sell_value),
        "buy_total_label": _fmt_money(operation.buy_value),
        "net_label": _fmt_money(operation.net_value),
        "net_direction_label": entry_net_direction,
    }

    def _asset_info_url(asset: Asset | None) -> str:
        # Placeholder neutro para futuras integrações (ex.: página interna).
        return "#"

    current_balance_value: Decimal | None = None
    if pl_total is not None:
        current_balance_value = pl_total
    elif current_net_value is not None and operation.net_value is not None:
        current_balance_value = (current_net_value - operation.net_value).quantize(money_quant)
    elif current_net_value is not None:
        current_balance_value = current_net_value
    final_direction_label = ""
    if current_balance_value is not None:
        final_direction_label = "recebe" if current_balance_value >= 0 else "paga"

    current_prices = {
        "updated_label": _fmt_updated(latest_update),
        "sell_price_label": _fmt_money(sell_live_price),
        "buy_price_label": _fmt_money(buy_live_price),
        "sell_total_label": _fmt_money(current_sell_total),
        "buy_total_label": _fmt_money(current_buy_total),
        "sell_pl_label": _fmt_money(sell_pl),
        "buy_pl_label": _fmt_money(buy_pl),
        "net_label": _fmt_money(current_net_value),
        "pl_total_label": _fmt_money(pl_total),
        "final_label": _fmt_money(current_balance_value),
        "final_direction_label": final_direction_label,
    }

    pnl_summary = None
    if sell_live_price is not None and buy_live_price is not None:
        capital_allocated_float = float(operation.capital_allocated or 0.0)
        total_trade_value = float(operation.sell_value + operation.buy_value)
        if total_trade_value > 0:
            short_share = float(operation.sell_value) / total_trade_value
        else:
            short_share = 0.5
        capital_short = capital_allocated_float * short_share
        capital_long = capital_allocated_float - capital_short
        pnl_stats = _long_short_pnl(
            entry_short=operation.sell_price,
            exit_short=sell_live_price,
            entry_long=operation.buy_price,
            exit_long=buy_live_price,
            capital_short=capital_short,
            capital_long=capital_long,
        )
        if pnl_stats:
            def _format_pct(value: float | None) -> str:
                if value is None:
                    return "--"
                try:
                    return f"{value:+.2f}%"
                except (TypeError, ValueError):
                    return "--"

            pnl_summary = {
                "capital_total_label": _fmt_money(pnl_stats.get("capital_total")),
                "lucro_short_label": _fmt_money(pnl_stats.get("lucro_short")),
                "lucro_long_label": _fmt_money(pnl_stats.get("lucro_long")),
                "lucro_total_label": _fmt_money(pnl_stats.get("lucro_total")),
                "retorno_short_label": _format_pct(pnl_stats.get("retorno_short_%")),
                "retorno_long_label": _format_pct(pnl_stats.get("retorno_long_%")),
                "retorno_total_label": _format_pct(pnl_stats.get("retorno_total_%")),
            }

    operations_cards.append(
        {
            "operation": operation,
            "url": reverse("core:operacao_encerrar", args=[operation.pk]),
            "operation_date_label": _fmt_updated(operation.opened_at),
            "operation_days_label": _format_days_open(operation.opened_at),
            "capital_label": _fmt_money(operation.capital_allocated),
            "entry": {
                "zscore_label": _fmt_metric(entry_zscore),
                "prices": entry_prices,
            },
            "entry_reference_label": entry_reference_label,
            "current": {
                "zscore_label": _fmt_metric(current_zscore),
                "prices": current_prices,
            },
            "is_delta_positive": is_delta_positive,
            "z_delta_label": z_delta_label,
            "pnl_ready": pnl_ready,
            "pnl_positive": pnl_positive,
            "pnl_negative": pnl_negative,
            "pnl_summary": pnl_summary,
            "current_zscore": current_zscore,
            "entry_zscore": entry_zscore,
            "asset_urls": {
                "sell": _asset_info_url(operation.sell_asset),
                "buy": _asset_info_url(operation.buy_asset),
            },
            "current_balance_value": current_balance_value,
            "entry_net_direction": entry_net_direction,
        }
    )
    def _card_zscore_sort_value(card):
        value = card.get("current_zscore")
        if value is None:
            return float("inf")
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("inf")

    operations_cards.sort(key=lambda card: (card.get("current_zscore") is None, abs(_card_zscore_sort_value(card))))

    total_operations = len(operations_cards)
    stats_capital = Decimal("0")
    stats_net_total = Decimal("0")
    stats_positive = 0
    stats_negative = 0
    stats_neutral = 0
    stats_pending = 0
    stats_gross_profit = Decimal("0")
    stats_gross_loss = Decimal("0")
    for card in operations_cards:
        stats_capital += _to_decimal(card["operation"].capital_allocated)
        pl_value = card.get("pl_total")
        if pl_value is None:
            stats_pending += 1
            continue
        pl_decimal = _to_decimal(pl_value)
        stats_net_total += pl_decimal
        if pl_decimal > 0:
            stats_positive += 1
            stats_gross_profit += pl_decimal
        elif pl_decimal < 0:
            stats_negative += 1
            stats_gross_loss += abs(pl_decimal)
        else:
            stats_neutral += 1

    max_capital = (
        max(
            (_to_decimal(card["operation"].capital_allocated) for card in operations_cards),
            default=Decimal("0"),
        )
        if operations_cards
        else Decimal("0")
    )

    resolved_operations = stats_positive + stats_negative + stats_neutral

    def _pct_label(value: int) -> str:
        if resolved_operations == 0:
            return "--"
        return f"{int((value / resolved_operations) * 100)}%"

    profit_factor_value = (
        stats_gross_profit / stats_gross_loss if stats_gross_loss > Decimal("0") else None
    )

    operations_summary = {
        "total_operations": total_operations,
        "positive_operations": stats_positive,
        "negative_operations": stats_negative,
        "neutral_operations": stats_neutral,
        "pending_operations": stats_pending,
        "net_label": _fmt_money(stats_net_total),
        "net_value": stats_net_total,
        "net_is_positive": stats_net_total >= 0,
        "profit_factor_label": _fmt_ratio(profit_factor_value),
        "profit_factor_value": profit_factor_value,
        "total_capital_label": _fmt_money(stats_capital),
        "total_operations_label": _fmt_int(total_operations),
        "positive_operations_label": _fmt_int(stats_positive),
        "negative_operations_label": _fmt_int(stats_negative),
        "neutral_operations_label": _fmt_int(stats_neutral),
        "positive_pct_label": _pct_label(stats_positive),
        "negative_pct_label": _pct_label(stats_negative),
        "neutral_pct_label": _pct_label(stats_neutral),
        "resolved_operations": resolved_operations,
    }

    ready_count = sum(1 for card in operations_cards if card.get("pnl_ready"))
    avg_ready_pl = stats_net_total / ready_count if ready_count else Decimal("0")
    operations_insight = {
        "ready_count": ready_count,
        "pending_count": stats_pending,
        "avg_ready_label": _fmt_money(avg_ready_pl) if ready_count else "--",
        "max_capital_label": _fmt_money(max_capital),
        "ready_pct_label": f"{int((ready_count / total_operations) * 100)}%" if total_operations else "--",
    }

    return {
        "operations_cards": operations_cards,
        "operations_summary": operations_summary,
        "operations_insight": operations_insight,
        "live_refresh_required": manual_refresh_required,
    }


def home(request):
    return render(
        request,
        "core/home.html",
        {
            "current": "home",
            "title": "Inicio - Operacoes em andamento",
            "operations_data_url": reverse("core:home_data"),
            "refresh_live_url": reverse("core:refresh_live_quotes"),
            "refresh_metrics_url": reverse("core:refresh_operation_metrics"),
        },
    )


@login_required
def home_data(request):
    payload = _build_home_operations_payload(request)
    html = render_to_string("core/_home_operations.html", payload, request=request)
    return JsonResponse({"ok": True, "html": html})


@login_required
@require_POST
def refresh_live_quotes(request):
    operations_qs = (
        Operation.objects.select_related("sell_asset", "buy_asset")
        .filter(user=request.user, status=Operation.STATUS_OPEN)
    )
    assets = {
        asset
        for operation in operations_qs
        for asset in (operation.sell_asset, operation.buy_asset)
        if asset
    }
    try:
        updated, total = update_live_quotes(list(assets))
    except Exception as exc:
        return JsonResponse({"ok": False, "detail": str(exc)}, status=500)
    return JsonResponse({"ok": True, "updated": updated, "total": total})


@login_required
@require_POST
def refresh_operation_metrics(request):
    operations_qs = (
        Operation.objects.select_related("pair", "left_asset", "right_asset")
        .filter(user=request.user, status=Operation.STATUS_OPEN)
        .order_by("-opened_at")
    )
    updated = 0
    errors: list[str] = []
    for operation in operations_qs:
        pair_ref = operation.pair or SimpleNamespace(
            left=operation.left_asset,
            right=operation.right_asset,
        )
        try:
            metrics = compute_pair_window_metrics(pair=pair_ref, window=operation.window)
        except Exception as exc:
            errors.append(f"{operation.pk}: {exc}")
            continue
        if not isinstance(metrics, dict):
            continue
        snapshot, _ = OperationMetricSnapshot.objects.update_or_create(
            operation=operation,
            snapshot_type=OperationMetricSnapshot.TYPE_CURRENT,
            defaults={"reference_date": timezone.localdate()},
        )
        snapshot.apply_payload(metrics)
        snapshot.save()
        updated += 1
    payload = {"ok": True, "updated": updated}
    if errors:
        payload["errors"] = errors
    return JsonResponse(payload)


def stub_page(request, page: str = "Pagina"):
    return render(
        request,
        "core/stub.html",
        {
            "current": page.lower(),
            "title": page,
        },
    )


def _decimal_from_value(value) -> Decimal | None:
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except (TypeError, ValueError, InvalidOperation):
        return None


@login_required
def encerradas(request):
    closed_qs = (
        Operation.objects.select_related("sell_asset", "buy_asset")
        .filter(user=request.user, status=Operation.STATUS_CLOSED)
        .order_by("-updated_at")
    )
    total_closed = closed_qs.count()
    asset_ids = set(
        a
        for pair in closed_qs.values_list("sell_asset_id", "buy_asset_id")
        for a in pair
        if a
    )
    asset_cache = {a.id: a for a in Asset.objects.filter(id__in=asset_ids)} if asset_ids else {}
    closing_price_cache: dict[tuple[int, object], Decimal | None] = {}

    def _fetch_price(asset_id: int | None, target_date):
        if not asset_id or not target_date:
            return None
        cache_key = (asset_id, target_date)
        if cache_key in closing_price_cache:
            return closing_price_cache[cache_key]
        asset_obj = asset_cache.get(asset_id)
        price = None
        if asset_obj:
            prices = get_daily_prices(asset_obj, end_date=target_date)
            if prices:
                last_row = prices[-1]
                price = _decimal_from_value(last_row.get("close"))
        closing_price_cache[cache_key] = price
        return price

    days_total = 0.0
    days_count = 0
    hit_count = 0
    pnl_valid = 0
    profit_sum = Decimal("0")
    loss_sum = Decimal("0")
    pnl_by_date: dict[object, Decimal] = defaultdict(lambda: Decimal("0"))
    days_by_date: dict[object, list[float]] = defaultdict(list)
    recent_operations: list[dict[str, object]] = []
    grid_operations: list[dict[str, object]] = []
    MAX_GRID = 12
    closing_records: list[tuple[object, Decimal | None]] = []
    MAX_RECENT = 6

    for operation in closed_qs:
        closing_dt = operation.updated_at or operation.opened_at
        if not closing_dt:
            continue
        try:
            closing_local = timezone.localtime(closing_dt)
        except Exception:
            closing_local = closing_dt
        try:
            opened_local = timezone.localtime(operation.opened_at) if operation.opened_at else closing_local
        except Exception:
            opened_local = operation.opened_at or closing_local
        days_open = 0.0
        if closing_local and opened_local:
            try:
                delta = closing_local - opened_local
                days_open = max(delta.total_seconds() / 86400, 0.0)
            except Exception:
                days_open = 0.0
        days_total += days_open
        days_count += 1
        closing_date = closing_local.date() if hasattr(closing_local, "date") else None
        if closing_date:
            days_by_date[closing_date].append(days_open)
        sell_close_price = _fetch_price(operation.sell_asset_id, closing_date)
        buy_close_price = _fetch_price(operation.buy_asset_id, closing_date)
        pl_value: Decimal | None = None
        if sell_close_price is not None and buy_close_price is not None:
            sell_qty = Decimal(operation.sell_quantity)
            buy_qty = Decimal(operation.buy_quantity)
            try:
                sell_pl = (operation.sell_price - sell_close_price) * sell_qty
                buy_pl = (buy_close_price - operation.buy_price) * buy_qty
                pl_value = sell_pl + buy_pl
            except (TypeError, ValueError, InvalidOperation):
                pl_value = None
        closing_records.append((closing_local, pl_value))
        if pl_value is not None:
            pnl_valid += 1
            if pl_value >= 0:
                hit_count += 1
                profit_sum += pl_value
            else:
                loss_sum += -pl_value
            if closing_date:
                pnl_by_date[closing_date] += pl_value
        if len(grid_operations) < MAX_GRID:
            grid_operations.append(
                {
                    "pair": operation.formatted_pair(),
                    "closed_on": closing_local.strftime("%d/%m/%Y") if closing_local else "--",
                    "status_label": "Acerto" if pl_value is not None and pl_value >= 0 else ("Erro" if pl_value is not None else "Sem dados"),
                    "status_type": "positive" if pl_value is not None and pl_value >= 0 else ("negative" if pl_value is not None else None),
                    "result_label": _format_money(pl_value) if pl_value is not None else "--",
                }
            )
        if len(recent_operations) < MAX_RECENT:
            recent_operations.append(
                {
                    "pair": operation.formatted_pair(),
                    "closed_on": closing_local.strftime("%d/%m/%Y") if closing_local else "--",
                    "days_label": f"{days_open:.1f} dias",
                    "result_label": _format_money(pl_value) if pl_value is not None else "Sem dados",
                    "result_direction": "recebe"
                    if pl_value is not None and pl_value >= 0
                    else ("paga" if pl_value is not None else ""),
                    "has_result": pl_value is not None,
                }
            )

    hit_rate_label = "--"
    hit_rate_detail = "Ainda não há fechamentos com resultado calculado."
    if pnl_valid:
        pct = round((hit_count / pnl_valid) * 100)
        hit_rate_label = f"{pct}%"
        hit_rate_detail = f"{hit_count} de {pnl_valid} fechamentos positivos"

    avg_days_label = "--"
    if days_count:
        avg_days_label = f"{(days_total / days_count):.1f} dias"

    ratio_label = "--"
    if loss_sum > Decimal("0"):
        try:
            ratio = float(profit_sum / loss_sum)
            ratio_label = f"{ratio:.2f}x"
        except (TypeError, ValueError, InvalidOperation):
            ratio_label = "--"
    elif profit_sum > Decimal("0"):
        ratio_label = "\u221e"

    net_total = profit_sum - loss_sum
    pnl_series = [
        {"label": closing_date.strftime("%d/%m"), "value": float(total)}
        for closing_date, total in sorted(pnl_by_date.items())
    ]
    days_series = []
    for closing_date in sorted(days_by_date):
        values = days_by_date[closing_date]
        if not values:
            continue
        avg = sum(values) / len(values)
        days_series.append({"label": closing_date.strftime("%d/%m"), "value": round(avg, 1)})
    profit_loss_series = [
        {"label": "Lucro", "value": float(profit_sum)},
        {"label": "Prejuízo", "value": float(loss_sum)},
    ]

    sorted_records = sorted(
        (rec for rec in closing_records if rec[0] is not None),
        key=lambda item: item[0],
    )
    positive_streak = 0
    negative_streak = 0
    current_type = None
    current_length = 0
    for _, pl_value in sorted_records:
        if pl_value is None:
            current_type = None
            current_length = 0
            continue
        run_type = "positive" if pl_value >= 0 else "negative"
        if run_type == current_type:
            current_length += 1
        else:
            current_type = run_type
            current_length = 1
        if run_type == "positive" and current_length > positive_streak:
            positive_streak = current_length
        if run_type == "negative" and current_length > negative_streak:
            negative_streak = current_length

    streak_series = [
        {"label": "Maior sequência positiva", "value": positive_streak},
        {"label": "Maior sequência negativa", "value": negative_streak},
    ]

    return render(
        request,
        "core/encerradas.html",
        {
            "current": "encerradas",
            "title": "Operações encerradas",
            "total_closed_label": number_format(total_closed, 0),
            "hit_rate_label": hit_rate_label,
            "hit_rate_detail": hit_rate_detail,
            "avg_days_label": avg_days_label,
            "profit_total_label": _format_money(profit_sum),
            "loss_total_label": _format_money(loss_sum),
            "net_total_label": _format_money(net_total),
            "profit_loss_ratio_label": ratio_label,
            "recent_operations": recent_operations,
        "chart_pnl_series": pnl_series,
        "chart_days_open_series": days_series,
        "chart_profit_loss": profit_loss_series,
        "chart_streak_series": streak_series,
        "operation_grid": grid_operations,
        "has_operations": total_closed > 0,
    },
)


def _format_detail_updated(dt_value) -> str:
    if not dt_value:
        return ""
    try:
        localized = timezone.localtime(dt_value)
    except Exception:
        localized = dt_value
    try:
        return localized.strftime("%d/%m %H:%M")
    except Exception:
        return ""


def _format_price(value) -> str:
    if value is None:
        return "--"
    try:
        return f"R$ {number_format(value, 2)}"
    except (TypeError, ValueError):
        return "--"


def _format_money(value) -> str:
    if value is None:
        return "--"
    return f"R$ {number_format(value, 2)}"


def _fmt_metric(value, digits: int = 2, fallback: str = "--") -> str:
    if value is None:
        return fallback
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return fallback


def _fmt_int(value: int | Decimal | None) -> str:
    if value is None:
        return "--"
    try:
        return number_format(value, 0)
    except (TypeError, ValueError):
        return "--"


def _fmt_samples(value) -> str:
    if value is None:
        return "--"
    try:
        return number_format(int(value), 0)
    except (TypeError, ValueError):
        return "--"


def _metrics_display(payload) -> list[dict[str, str]]:
    if not isinstance(payload, dict):
        return []
    return [
        {"label": "Z-score", "value": _fmt_metric(payload.get("zscore"), 2)},
        {"label": "Half-life", "value": _fmt_metric(payload.get("half_life"), 2)},
        {"label": "ADF", "value": _fmt_metric(payload.get("adf_pvalue"), 4)},
        {"label": "Beta", "value": _fmt_metric(payload.get("beta"), 4)},
        {"label": "Corr 30", "value": _fmt_metric(payload.get("corr30"), 3)},
        {"label": "Corr 60", "value": _fmt_metric(payload.get("corr60"), 3)},
        {"label": "Amostra", "value": _fmt_samples(payload.get("n_samples"))},
    ]


def _long_short_pnl(
    entry_short: Decimal | float | None,
    exit_short: Decimal | float | None,
    entry_long: Decimal | float | None,
    exit_long: Decimal | float | None,
    capital_short: Decimal | float | None,
    capital_long: Decimal | float | None,
) -> dict[str, float] | None:
    try:
        entry_short_f = float(entry_short)
        exit_short_f = float(exit_short)
        entry_long_f = float(entry_long)
        exit_long_f = float(exit_long)
    except (TypeError, ValueError):
        return None
    capital_short_f = float(capital_short or 0.0)
    capital_long_f = float(capital_long or 0.0)
    if entry_short_f == 0 or entry_long_f == 0:
        return None
    r_short = (entry_short_f - exit_short_f) / entry_short_f
    r_long = (exit_long_f - entry_long_f) / entry_long_f
    pnl_short = capital_short_f * r_short
    pnl_long = capital_long_f * r_long
    total_capital = capital_short_f + capital_long_f
    total_pnl = pnl_short + pnl_long
    total_return = total_pnl / total_capital if total_capital else 0.0
    return {
        "retorno_short_%": r_short * 100,
        "retorno_long_%": r_long * 100,
        "lucro_short": pnl_short,
        "lucro_long": pnl_long,
        "lucro_total": total_pnl,
        "retorno_total_%": total_return * 100,
        "capital_total": total_capital,
    }


def _format_pct(value: float | None) -> str:
    if value is None:
        return "--"
    try:
        return f"{value:+.2f}%"
    except (TypeError, ValueError):
        return "--"


def _build_pnl_summary(operation: Operation, sell_live_price, buy_live_price, sell_price, buy_price):
    if sell_live_price is None or buy_live_price is None:
        return None
    capital_allocated_float = float(operation.capital_allocated or 0.0)
    sell_value = float(getattr(operation, "sell_value", Decimal("0")) or Decimal("0"))
    buy_value = float(getattr(operation, "buy_value", Decimal("0")) or Decimal("0"))
    total_trade_value = sell_value + buy_value
    short_share = sell_value / total_trade_value if total_trade_value > 0 else 0.5
    capital_short = capital_allocated_float * short_share
    capital_long = capital_allocated_float - capital_short
    pnl_stats = _long_short_pnl(
        entry_short=sell_price,
        exit_short=sell_live_price,
        entry_long=buy_price,
        exit_long=buy_live_price,
        capital_short=capital_short,
        capital_long=capital_long,
    )
    if not pnl_stats:
        return None
    return {
        "capital_total_label": _format_money(pnl_stats.get("capital_total")),
        "lucro_short_label": _format_money(pnl_stats.get("lucro_short")),
        "lucro_long_label": _format_money(pnl_stats.get("lucro_long")),
        "lucro_total_label": _format_money(pnl_stats.get("lucro_total")),
        "retorno_short_label": _format_pct(pnl_stats.get("retorno_short_%")),
        "retorno_long_label": _format_pct(pnl_stats.get("retorno_long_%")),
        "retorno_total_label": _format_pct(pnl_stats.get("retorno_total_%")),
    }


def _source_label(src: str | None) -> str:
    if src == "mt5":
        return "MT5 (ao vivo)"
    if src == "cache":
        return "MT5 (cache)"
    if src == "daily":
        return "Historico (fechamento)"
    if src == "daily_mt5":
        return "DailyPrice (MT5)"
    return ""


def _build_current_asset_price(asset: Asset | None) -> tuple[Decimal | None, object | None, str | None]:
    if not asset:
        return None, None, None
    use_mt5_live = getattr(settings, "USE_MT5_LIVE", True)
    price: Decimal | None = None
    updated = None
    source = None
    live_quote = getattr(asset, "live_quote", None)
    if live_quote:
        raw_price = live_quote.price if live_quote.price is not None else getattr(live_quote, "last", None)
        if raw_price is not None:
            try:
                price = Decimal(str(raw_price))
            except (TypeError, ValueError, InvalidOperation):
                price = None
            else:
                source = getattr(live_quote, "source", None) or "cache"
        updated = getattr(live_quote, "as_of", None) or getattr(live_quote, "updated_at", None)
    if price is None and use_mt5_live and getattr(asset, "use_mt5", False):
        ticker = (getattr(asset, "ticker", "") or "").strip().upper()
        if ticker:
            live_price = None
            try:
                live_price = fetch_latest_price(ticker)
            except Exception:
                live_price = None
            if live_price is not None:
                try:
                    price = Decimal(str(live_price))
                except (TypeError, ValueError, InvalidOperation):
                    price = None
                else:
                    updated = timezone.now()
                    source = "mt5"
                    try:
                        QuoteLive.objects.update_or_create(
                            asset=asset,
                            defaults={"price": float(price), "source": "mt5"},
                        )
                    except Exception:
                        pass
    if price is None:
        daily_prices = get_daily_prices(asset, end_date=timezone.localdate())
        if daily_prices:
            last_row = daily_prices[-1]
            close_px = last_row.get("close")
            if close_px is not None:
                try:
                    price = Decimal(str(close_px))
                except (TypeError, ValueError, InvalidOperation):
                    price = None
                else:
                    updated = last_row.get("date")
                    source = "daily_mt5" if getattr(asset, "use_mt5", False) else "daily"
    return price, updated, source


@login_required
def operacoes(request):
    config_obj, _ = UserMetricsConfig.objects.get_or_create(
        user=request.user,
        defaults=UserMetricsConfig.default_kwargs(),
    )

    window_options = config_obj.windows_list() or list(DEFAULT_WINDOWS)
    default_window = config_obj.base_window or DEFAULT_BASE_WINDOW

    def _safe_window(value: str | None) -> int:
        if not value:
            return default_window
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return default_window
        return parsed if parsed > 0 else default_window

    def _normalize_ticker(value: str | None) -> str:
        return (value or "").strip().upper()

    def _parse_decimal(raw: str | None) -> Decimal:
        if raw is None:
            raise InvalidOperation
        text = str(raw).strip()
        if not text:
            raise InvalidOperation
        text = text.replace("R$", "").replace("r$", "").replace(" ", "")
        if "," in text and "." in text:
            text = text.replace(".", "")
            text = text.replace(",", ".")
        elif "," in text:
            text = text.replace(",", ".")
        return Decimal(text)

    def _format_money(value: Decimal | float | None) -> str | None:
        if value is None:
            return None
        return f"R$ {number_format(value, 2)}"

    def _format_decimal_input(value: Decimal) -> str:
        return format(value.quantize(Decimal("0.01")), "f")

    def _compose_asset_label(ticker: str | None, name: str | None) -> str:
        ticker_clean = (ticker or "").strip().upper()
        name_clean = (name or "").strip()
        if ticker_clean and name_clean:
            return f"{ticker_clean} ({name_clean})"
        return ticker_clean or name_clean or "--"

    def _get_asset(ticker: str) -> Asset | None:
        if not ticker:
            return None
        return Asset.objects.select_related("live_quote").filter(ticker=_normalize_ticker(ticker)).first()

    def _build_existing_operation_info(left_asset: Asset | None, right_asset: Asset | None, pair_obj: Pair | None):
        if not left_asset or not right_asset:
            return None
        filters = (
            Q(left_asset=left_asset, right_asset=right_asset)
            | Q(left_asset=right_asset, right_asset=left_asset)
        )
        if pair_obj:
            filters |= Q(pair=pair_obj)
        existing_op = (
            Operation.objects.select_related("sell_asset", "buy_asset")
            .filter(user=request.user, status=Operation.STATUS_OPEN)
            .filter(filters)
            .order_by("-opened_at")
            .first()
        )
        if not existing_op:
            return None
        opened_label = ""
        if existing_op.opened_at:
            try:
                opened_label = timezone.localtime(existing_op.opened_at).strftime("%d/%m %H:%M")
            except Exception:
                opened_label = ""
        is_inverted = (
            existing_op.left_asset == right_asset and existing_op.right_asset == left_asset
        )
        pair_label = existing_op.formatted_pair()
        descriptor = "par invertido" if is_inverted else "mesmo par selecionado"
        return {
            "message": f"Ja existe uma operacao em andamento para {pair_label} ({descriptor}). Deseja continuar?",
            "opened_label": opened_label,
            "url": reverse("core:operacao_encerrar", args=[existing_op.pk]),
            "pair_label": pair_label,
        }

    def _format_updated(dt_value):
        if not dt_value:
            return ""
        try:
            localized = timezone.localtime(dt_value)
        except Exception:
            localized = dt_value
        return localized.strftime("%d/%m %H:%M")

    def _build_trade_info(role: str, asset: Asset | None, ticker: str) -> dict[str, str | float | bool | None]:
        ticker_norm = _normalize_ticker(ticker)
        use_mt5_live = getattr(settings, "USE_MT5_LIVE", False)
        asset_obj = asset or _get_asset(ticker_norm)

        price = None
        updated_at = None
        price_source = None

        live_quote = getattr(asset_obj, "live_quote", None) if asset_obj else None
        if live_quote:
            raw_price = getattr(live_quote, "price", None)
            if raw_price is None:
                raw_price = getattr(live_quote, "last", None)
            price = raw_price
            updated_at = getattr(live_quote, "as_of", None) or getattr(live_quote, "updated_at", None)
            if price is not None:
                price_source = getattr(live_quote, "source", None) or "cache"

        live_px_error = False
        if ticker_norm and use_mt5_live and getattr(asset_obj, "use_mt5", False):
            try:
                live_px = fetch_latest_price(ticker_norm)
            except Exception:
                live_px = None
                live_px_error = True
            if live_px is not None:
                price = live_px
                updated_at = timezone.now()
                price_source = "mt5"
                if asset_obj:
                    QuoteLive.objects.update_or_create(asset=asset_obj, defaults={"price": live_px, "source": "mt5"})

        if price is None and asset_obj:
            daily_prices = get_daily_prices(asset_obj, end_date=timezone.localdate())
            if daily_prices:
                last_row = daily_prices[-1]
                close_px = last_row.get("close")
                if close_px is not None:
                    price = float(close_px)
                    updated_at = last_row.get("date")
                    price_source = "daily_mt5" if getattr(asset_obj, "use_mt5", False) else "daily"

        info = {
            "role": role,
            "label": "Venda" if role == "sell" else "Compra",
            "ticker": ticker_norm,
            "name": getattr(asset_obj, "name", ""),
            "price": price,
            "price_label": f"R$ {number_format(price, 2)}" if price is not None else None,
            "source": price_source,
            "source_label": _source_label(price_source),
            "updated_label": _format_updated(updated_at),
            "error_label": "",
            "fetched_now": price_source == "mt5",
        }

        if price is None and ticker_norm:
            if use_mt5_live:
                info["error_label"] = "Sem cotacao MT5 disponivel para este ticker." if not live_px_error else "Nao foi possivel contatar o provedor MT5 agora."

        return info

    def _handle_operation_post():
        posted = request.POST
        errors: list[str] = []

        def respond_with_errors():
            payload = {"ok": False, "errors": errors or ["Dados invalidos ou incompletos."]}
            return JsonResponse(payload, status=400)

        def parse_positive_decimal(field: str, label: str) -> Decimal | None:
            raw = posted.get(field)
            if not raw:
                errors.append(f"{label} nao informado.")
                return None
            try:
                value = _parse_decimal(raw)
            except InvalidOperation:
                errors.append(f"{label} invalido.")
                return None
            if value <= 0:
                errors.append(f"{label} precisa ser maior que zero.")
                return None
            return value

        def parse_positive_int(field: str, label: str) -> int | None:
            raw = posted.get(field)
            if not raw:
                errors.append(f"{label} nao informado.")
                return None
            try:
                value = int(str(raw).strip())
            except (TypeError, ValueError):
                errors.append(f"{label} invalido.")
                return None
            if value <= 0:
                errors.append(f"{label} precisa ser maior que zero.")
                return None
            return value

        left_ticker = _normalize_ticker(posted.get("left"))
        right_ticker = _normalize_ticker(posted.get("right"))
        sell_ticker = _normalize_ticker(posted.get("sell_ticker"))
        buy_ticker = _normalize_ticker(posted.get("buy_ticker"))

        if not left_ticker:
            errors.append("Ativo esquerdo nao informado.")
        if not right_ticker:
            errors.append("Ativo direito nao informado.")
        if not sell_ticker:
            errors.append("Ativo vendido nao informado.")
        if not buy_ticker:
            errors.append("Ativo comprado nao informado.")

        lot_size = 100
        lot_size_raw = posted.get("lot_size")
        if lot_size_raw:
            try:
                lot_size = int(str(lot_size_raw).strip())
                if lot_size <= 0:
                    errors.append("Tamanho do lote precisa ser maior que zero.")
                    lot_size = 100
            except (TypeError, ValueError):
                errors.append("Tamanho do lote invalido.")

        lot_multiplier = parse_positive_int("lot_multiplier", "Quantidade de lotes")
        sell_qty = parse_positive_int("sell_qty", "Quantidade vendida")
        buy_qty = parse_positive_int("buy_qty", "Quantidade comprada")
        capital = parse_positive_decimal("capital", "Capital alocado")
        sell_price = parse_positive_decimal("sell_price", "Preco de venda")
        buy_price = parse_positive_decimal("buy_price", "Preco de compra")

        if posted.get("window") is None:
            window_value = default_window
        else:
            window_value = _safe_window(posted.get("window"))
        source_value = (posted.get("source") or "").strip().lower()
        if source_value not in {"analysis", "manual"}:
            source_value = "analysis"
        is_real = bool(posted.get("is_real"))

        if errors:
            return respond_with_errors()

        if lot_multiplier is None:
            lot_multiplier = 1

        pair_obj: Pair | None = None
        pair_pk_raw = posted.get("pair_id")
        if pair_pk_raw:
            try:
                pair_pk = int(str(pair_pk_raw).strip())
            except (TypeError, ValueError):
                pair_pk = None
            else:
                pair_obj = (
                    Pair.objects.select_related("left", "right")
                    .filter(pk=pair_pk)
                    .first()
                )

        left_asset = pair_obj.left if pair_obj else None
        right_asset = pair_obj.right if pair_obj else None
        if not left_asset and left_ticker:
            left_asset = _get_asset(left_ticker)
        if not right_asset and right_ticker:
            right_asset = _get_asset(right_ticker)
        sell_asset = _get_asset(sell_ticker)
        buy_asset = _get_asset(buy_ticker)

        if not left_asset:
            errors.append(f"Ativo esquerdo {left_ticker or '?'} nao encontrado.")
        if not right_asset:
            errors.append(f"Ativo direito {right_ticker or '?'} nao encontrado.")
        if not sell_asset:
            errors.append(f"Ativo vendido {sell_ticker or '?'} nao encontrado.")
        if not buy_asset:
            errors.append(f"Ativo comprado {buy_ticker or '?'} nao encontrado.")

        if errors:
            return respond_with_errors()

        if not pair_obj and left_asset and right_asset:
            pair_obj = (
                Pair.objects.select_related("left", "right")
                .filter(left=left_asset, right=right_asset)
                .first()
            )

        pair_ref = pair_obj or (
            SimpleNamespace(left=left_asset, right=right_asset)
            if left_asset and right_asset
            else None
        )

        metrics_payload = None
        entry_zscore = None
        if pair_ref:
            try:
                metrics_payload = compute_pair_window_metrics(pair=pair_ref, window=window_value)
            except Exception:
                metrics_payload = None
            if isinstance(metrics_payload, dict):
                raw_zscore = metrics_payload.get("zscore")
                try:
                    entry_zscore = float(raw_zscore) if raw_zscore is not None else None
                except (TypeError, ValueError):
                    entry_zscore = None

        try:
            plan_result = calcular_proporcao_long_short(
                preco_short=float(sell_price),
                preco_long=float(buy_price),
                limite_venda=float(capital),
                lote=lot_size,
                ticker_short=sell_ticker,
                ticker_long=buy_ticker,
                capital_informado=float(capital),
            )
        except ValueError as exc:
            errors.append(str(exc))
            return respond_with_errors()

        if plan_result is None:
            errors.append("Nao foi possivel montar o plano com os dados informados.")
            return respond_with_errors()

        price_quant = Decimal("0.000001")
        money_quant = Decimal("0.01")
        sell_price = sell_price.quantize(price_quant)
        buy_price = buy_price.quantize(price_quant)
        sell_value = (sell_price * Decimal(sell_qty)).quantize(money_quant)
        buy_value = (buy_price * Decimal(buy_qty)).quantize(money_quant)
        net_value = (sell_value - buy_value).quantize(money_quant)
        capital_allocated = capital.quantize(money_quant)

        trade_plan_payload = plan_result.to_payload()

        orientation = "default"
        if left_ticker and right_ticker and sell_ticker and buy_ticker:
            if sell_ticker == right_ticker and buy_ticker == left_ticker:
                orientation = "inverted"

        operation = Operation.objects.create(
            user=request.user,
            pair=pair_obj,
            left_asset=left_asset,
            right_asset=right_asset,
            sell_asset=sell_asset,
            buy_asset=buy_asset,
            window=window_value,
            orientation=orientation,
            source=source_value,
            sell_quantity=sell_qty,
            buy_quantity=buy_qty,
            lot_size=lot_size,
            lot_multiplier=lot_multiplier,
            sell_price=sell_price,
            buy_price=buy_price,
            sell_value=sell_value,
            buy_value=buy_value,
            net_value=net_value,
            capital_allocated=capital_allocated,
            entry_zscore=entry_zscore,
            trade_plan=trade_plan_payload,
            pair_metrics=metrics_payload if isinstance(metrics_payload, dict) else None,
            is_real=is_real,
        )

        metrics_snapshot_payload = metrics_payload if isinstance(metrics_payload, dict) else None
        if metrics_snapshot_payload:
            snapshot = OperationMetricSnapshot(
                operation=operation,
                snapshot_type=OperationMetricSnapshot.TYPE_OPEN,
                reference_date=timezone.localdate(),
            )
            snapshot.apply_payload(metrics_snapshot_payload)
            snapshot.save()

        redirect_url = reverse("core:operacao_encerrar", args=[operation.pk])
        return JsonResponse({"ok": True, "redirect": redirect_url})

    if request.method == "POST":
        return _handle_operation_post()

    initial_window = _safe_window(request.GET.get("window"))
    source = (request.GET.get("source") or "").strip().lower()
    if source not in {"analysis", "manual"}:
        source = "analysis" if request.GET.get("pair") or request.GET.get("left") else "manual"

    initial_left = ""
    initial_right = ""
    pair_obj: Pair | None = None
    pair_param = request.GET.get("pair")

    if pair_param:
        try:
            pair_id = int(pair_param)
            pair_obj = Pair.objects.select_related("left", "right", "left__live_quote", "right__live_quote").get(pk=pair_id)
            initial_left = pair_obj.left.ticker
            initial_right = pair_obj.right.ticker
            window_guess = pair_obj.chosen_window or pair_obj.base_window
            if window_guess and initial_window == default_window:
                initial_window = int(window_guess)
            source = "analysis"
        except (Pair.DoesNotExist, ValueError):
            pair_obj = None
            messages.warning(request, "Par informado na analise nao foi encontrado.")

    if not pair_obj:
        left_param = (request.GET.get("left") or "").strip().upper()
        right_param = (request.GET.get("right") or "").strip().upper()
        if left_param or right_param:
            initial_left = left_param
            initial_right = right_param
            source = "analysis" if left_param and right_param else source

    initial_left = _normalize_ticker(initial_left)
    initial_right = _normalize_ticker(initial_right)

    if initial_window not in window_options:
        window_options = sorted(set(window_options + [initial_window]))

    left_asset = pair_obj.left if pair_obj else _get_asset(initial_left)
    right_asset = pair_obj.right if pair_obj else _get_asset(initial_right)

    metrics = None
    zscore_value: float | None = None
    if pair_obj and initial_window:
        try:
            metrics = compute_pair_window_metrics(pair=pair_obj, window=initial_window)
            raw_z = metrics.get("zscore") if isinstance(metrics, dict) else None
            if raw_z is not None:
                zscore_value = float(raw_z)
        except Exception:
            metrics = None
            messages.warning(request, "Nao foi possivel calcular o Z-score deste par no momento.")

    if zscore_value is not None and zscore_value < 0:
        sell_asset = right_asset
        sell_ticker = initial_right
        buy_asset = left_asset
        buy_ticker = initial_left
        orientation = "inverted"
    else:
        sell_asset = left_asset
        sell_ticker = initial_left
        buy_asset = right_asset
        buy_ticker = initial_right
        orientation = "default"

    sell_info = _build_trade_info("sell", sell_asset, sell_ticker)
    buy_info = _build_trade_info("buy", buy_asset, buy_ticker)

    summary_note = ""
    if zscore_value is not None and sell_info["ticker"] and buy_info["ticker"]:
        summary_note = (
            f"Z-score {zscore_value:.2f} sugere vender {sell_info['ticker']} e comprar {buy_info['ticker']}."
        )
    elif not sell_info["ticker"] and not buy_info["ticker"]:
        summary_note = "Informe os tickers para montar o plano da operacao."

    existing_operation_info = _build_existing_operation_info(left_asset, right_asset, pair_obj)

    summary = {
        "zscore": zscore_value,
        "zscore_label": f"{zscore_value:.2f}" if zscore_value is not None else None,
        "orientation": orientation,
        "note": summary_note,
        "sell": sell_info,
        "buy": buy_info,
        "from_analysis": source == "analysis",
    }

    def _fmt_metric(value: float | int | None, digits: int = 2, fallback: str = "--") -> str:
        if value is None:
            return fallback
        try:
            return f"{float(value):.{digits}f}"
        except (TypeError, ValueError):
            return fallback

    pair_metrics_display: list[dict[str, str]] = []
    if isinstance(metrics, dict) and metrics:
        pair_metrics_display = [
            {"label": "Z-score", "value": _fmt_metric(metrics.get("zscore"), 2)},
            {"label": "Half-life", "value": _fmt_metric(metrics.get("half_life"), 2)},
            {"label": "ADF p-valor", "value": _fmt_metric(metrics.get("adf_pvalue"), 4)},
            {"label": "Beta", "value": _fmt_metric(metrics.get("beta"), 4)},
            {"label": "Correlacao 30", "value": _fmt_metric(metrics.get("corr30"), 4)},
            {"label": "Correlacao 60", "value": _fmt_metric(metrics.get("corr60"), 4)},
            {
                "label": "Amostra",
                "value": number_format(metrics.get("n_samples"), 0) if metrics.get("n_samples") is not None else "--",
            },
        ]

    summary["pair_metrics_payload"] = metrics if isinstance(metrics, dict) else None
    summary["pair_metrics_display"] = pair_metrics_display

    lot_multiplier_param = request.GET.get("lotes") or request.GET.get("multiplicador")
    lot_multiplier = 1
    if lot_multiplier_param is not None:
        try:
            lot_multiplier = int(str(lot_multiplier_param).strip())
        except (TypeError, ValueError):
            lot_multiplier = 1
        else:
            if lot_multiplier < 1:
                lot_multiplier = 1
            elif lot_multiplier > 999:
                lot_multiplier = 999

    capital_param = request.GET.get("valor") or request.GET.get("capital")
    lot_size_base = 100
    valuation: dict[str, object] = {
        "input_raw": capital_param or "",
        "input_display": capital_param or "",
        "error": None,
        "has_result": False,
        "lot_size": lot_size_base,
        "lot_multiplier": lot_multiplier,
        "target_shares": lot_size_base * lot_multiplier,
        "suggested_value": None,
        "suggested_label": None,
        "input_adjusted": False,
        "input_adjusted_message": "",
    }

    sell_price = summary["sell"]["price"]
    buy_price = summary["buy"]["price"]

    lot_multiplier_dec = Decimal(lot_multiplier)
    lot_size_dec = Decimal(valuation["lot_size"])
    suggested_capital: Decimal | None = None
    sell_price_dec: Decimal | None = None
    buy_price_dec: Decimal | None = None
    if sell_price is not None and buy_price is not None:
        sell_price_dec = Decimal(str(sell_price))
        buy_price_dec = Decimal(str(buy_price))
        suggested_capital = max(sell_price_dec, buy_price_dec) * lot_size_dec * lot_multiplier_dec
        valuation["suggested_value"] = suggested_capital
        valuation["suggested_label"] = _format_money(suggested_capital)
        if not capital_param:
            valuation["input_display"] = _format_decimal_input(suggested_capital)
    elif not capital_param:
        valuation["input_display"] = ""

    capital_informado: Decimal | None = None
    capital_utilizado: Decimal | None = None

    if capital_param:
        try:
            capital_informado = _parse_decimal(capital_param)
        except InvalidOperation:
            valuation["error"] = "Informe um valor numerico valido."
        else:
            if capital_informado <= 0:
                valuation["error"] = "Informe um valor maior que zero."
            else:
                capital_utilizado = capital_informado
    elif suggested_capital is not None:
        capital_utilizado = suggested_capital

    if (
        valuation["error"] is None
        and capital_utilizado is not None
        and suggested_capital is not None
        and capital_utilizado < suggested_capital
    ):
        valuation["input_adjusted"] = True
        valuation["input_adjusted_message"] = (
            f"Valor informado ajustado para o minimo recomendado ({valuation['suggested_label']})."
        )
        capital_utilizado = suggested_capital

    if valuation["error"] is None and capital_utilizado is not None:
        valuation["input_display"] = _format_decimal_input(capital_utilizado)

    if (
        valuation["error"] is None
        and capital_utilizado is not None
        and (sell_price_dec is None or buy_price_dec is None)
    ):
        valuation["error"] = "Cotacoes indisponiveis para calcular os lotes."

    result = None
    if (
        valuation["error"] is None
        and capital_utilizado is not None
        and sell_price_dec is not None
        and buy_price_dec is not None
    ):
        result = calcular_proporcao_long_short(
            preco_short=float(sell_price_dec),
            preco_long=float(buy_price_dec),
            limite_venda=float(capital_utilizado),
            lote=int(valuation["lot_size"]),
            ticker_short=sell_info["ticker"],
            ticker_long=buy_info["ticker"],
            nome_short=sell_info["name"],
            nome_long=buy_info["name"],
            capital_informado=float(capital_informado) if capital_informado is not None else None,
        )
        if result is None:
            valuation["error"] = (
                f"Valor insuficiente para um lote de {valuation['lot_size']} acoes na ponta vendida."
            )
        else:
            payload = result.to_payload()
            minimum_total = result.valor_minimo_para_operar * lot_multiplier_dec
            target_shares_total = result.lote * lot_multiplier
            lots_result = result.lotes_vendidos
            description = result.resumo
            if lots_result > 1:
                description = (
                    f"{description} Plano calculado com {lots_result} lotes "
                    f"({result.quantidade_vendida} acoes) na ponta vendida."
                )
            sell_lot_notional = result.preco_short * Decimal(result.lote)
            buy_lot_notional = result.preco_long * Decimal(result.lote)
            valuation.update(
                {
                    "has_result": True,
                    "result": result,
                    "lots": result.lotes_vendidos,
                    "shares": result.quantidade_vendida,
                    "shares_buy": result.quantidade_comprada,
                    "capital_value": result.capital_utilizado,
                    "capital_label": _format_money(result.capital_utilizado),
                    "capital_informado_label": _format_money(result.capital_informado)
                    if result.capital_informado is not None
                    else None,
                    "lot_notional_label": _format_money(sell_lot_notional),
                    "sell_lot_notional_label": _format_money(sell_lot_notional),
                    "sell_amount": result.valor_vendido,
                    "sell_label": _format_money(result.valor_vendido),
                    "buy_amount": result.valor_comprado,
                    "buy_label": _format_money(result.valor_comprado),
                    "net_amount": result.saldo,
                    "net_label": _format_money(abs(result.saldo)),
                    "net_direction": "recebe" if result.saldo >= 0 else "paga",
                    "minimum_label": _format_money(minimum_total),
                    "proporcao_label": f"{result.proporcao:.4f}",
                    "description": description,
                    "result_payload": payload,
                    "target_shares": target_shares_total,
                    "sell_unit_label": _format_money(result.preco_short),
                    "sell_asset_label": _compose_asset_label(sell_info["ticker"], sell_info["name"]),
                    "buy_asset_label": _compose_asset_label(buy_info["ticker"], buy_info["name"]),
                    "buy_unit_label": _format_money(result.preco_long),
                    "buy_lot_notional_label": _format_money(buy_lot_notional),
                    "buy_lots": result.lotes_comprados,
                }
            )
            valuation["input_display"] = _format_decimal_input(result.capital_utilizado)
            if (
                result.capital_informado is not None
                and result.capital_informado != result.capital_utilizado
            ):
                valuation["input_adjusted"] = True
                valuation["input_adjusted_message"] = (
                    f"Valor informado { _format_money(result.capital_informado) } "
                    f"ajustado para { _format_money(result.capital_utilizado) }."
                )
            summary["trade_plan_description"] = description
            summary["trade_plan_metrics"] = payload
            summary["trade_plan"] = result

    context = {
        "current": "operacoes",
        "title": "Operacoes",
        "window_options": window_options,
        "initial": {
            "left": initial_left,
            "right": initial_right,
            "window": initial_window,
            "pair_id": pair_obj.pk if pair_obj else (pair_param or ""),
            "source": source,
        },
        "pair_obj": pair_obj,
        "prefilled": bool(initial_left and initial_right),
        "summary": summary,
        "existing_operation_info": existing_operation_info,
        "valuation": valuation,
    }
    return render(request, "core/operacoes.html", context)


def faltantes(request):
    return redirect("cotacoes:faltantes_home")


@login_required
def config(request):
    config_obj, _ = UserMetricsConfig.objects.get_or_create(
        user=request.user,
        defaults=UserMetricsConfig.default_kwargs(),
    )

    if request.method == "POST":
        form = UserMetricsConfigForm(request.POST, instance=config_obj)
        if form.is_valid():
            form.save()
            messages.success(request, "Configuracoes atualizadas com sucesso.")
            return redirect("core:config")
    else:
        form = UserMetricsConfigForm(instance=config_obj)

    sample_windows = ", ".join(str(w) for w in config_obj.windows_list())
    metrics_help = [
        {
            "title": "Janela base (Grid A)",
            "icon": "bi-bullseye",
            "description": (
                "Quantidade de pregoes usados no calculo inicial. "
                "Aprova ou reprova pares antes de aparecerem no scanner."
            ),
        },
        {
            "title": "Janelas do scanner",
            "icon": "bi-sliders",
            "description": (
                "Lista de janelas (dias) testadas no Grid B. "
                "Use diferentes horizontes para encontrar pares em ritmos variados."
            ),
        },
        {
            "title": "ADF minimo (%)",
            "icon": "bi-graph-up-arrow",
            "description": (
                "Filtro baseado no teste Augmented Dickey-Fuller. "
                "Representa (1 - p-valor). Valores maiores indicam serie mais estacionaria."
            ),
        },
        {
            "title": "Z-score minimo",
            "icon": "bi-activity",
            "description": (
                "Controla a magnitude minima do Z-score do spread padronizado. "
                "Evita oportunidades com desvio pequeno demais."
            ),
        },
        {
            "title": "Janela do beta movel",
            "icon": "bi-arrow-repeat",
            "description": (
                "Define quantos pregoes entram em cada bloco do grafico de beta movel. "
                "Janelas curtas deixam o beta mais sensivel."
            ),
        },
        {
            "title": "Half-life maximo",
            "icon": "bi-hourglass-split",
            "description": (
                "Limite superior para o tempo de meia vida do spread. "
                "Pares com half-life maior que o valor informado sao descartados."
            ),
        },
        {
            "title": "Beta",
            "icon": "bi-diagram-3",
            "description": (
                "Coeficiente da regressao log(left) x log(right). "
                "Mostra o quanto o ativo esquerdo deveria variar quando o direito move 1%."
            ),
        },
    ]

    return render(
        request,
        "core/config.html",
        {
            "form": form,
            "current": "config",
            "title": "Configuracoes de metricas",
            "sample_windows": sample_windows,
            "metrics_help": metrics_help,
        },
    )



@login_required
def operacao_encerrar(request, pk: int):
    operation = get_object_or_404(
        Operation.objects.select_related(
            "sell_asset",
            "buy_asset",
            "left_asset",
            "right_asset",
            "pair",
        ).prefetch_related(
            Prefetch(
                "metric_snapshots",
                queryset=OperationMetricSnapshot.objects.order_by("-reference_date"),
            )
        ),
        pk=pk,
        user=request.user,
        status=Operation.STATUS_OPEN,
    )

    pair_label = f"{operation.sell_asset.ticker} / {operation.buy_asset.ticker}"

    if request.method == "POST":
        action = request.POST.get("action")
        if action == "delete":
            operation.delete()
            messages.success(request, f"Operacao {pair_label} excluida com sucesso.")
            return redirect("core:home")
        if action == "close":
            operation.status = Operation.STATUS_CLOSED
            operation.save(update_fields=["status", "updated_at"])
            messages.success(request, f"Operacao {pair_label} encerrada com sucesso.")
            return redirect("core:home")

    entry_snapshot = None
    latest_snapshot = None
    entry_metrics_payload = {}
    snapshots = list(operation.metric_snapshots.all())
    if snapshots:
        latest_snapshot = snapshots[0]
        for snap in snapshots:
            if snap.snapshot_type == OperationMetricSnapshot.TYPE_OPEN and entry_snapshot is None:
                entry_snapshot = snap
    if entry_snapshot and entry_snapshot.payload:
        entry_metrics_payload = entry_snapshot.payload
    elif isinstance(operation.pair_metrics, dict):
        entry_metrics_payload = operation.pair_metrics

    entry_zscore = entry_snapshot.zscore if entry_snapshot and entry_snapshot.zscore is not None else operation.entry_zscore

    pair_ref = operation.pair if operation.pair else SimpleNamespace(
        left=operation.left_asset,
        right=operation.right_asset,
    )

    current_metrics_payload = {}
    current_zscore = None
    try:
        metrics_now = compute_pair_window_metrics(pair=pair_ref, window=operation.window)
    except Exception:
        metrics_now = None

    if isinstance(metrics_now, dict):
        current_metrics_payload = metrics_now
        current_zscore = metrics_now.get("zscore")

    try:
        raw_zscore_series = get_zscore_series(pair=pair_ref, window=operation.window)
    except Exception:
        raw_zscore_series = []

    zscore_series_points: list[dict[str, object]] = []
    for data_point in raw_zscore_series or []:
        if data_point is None or len(data_point) < 2:
            continue
        dt_value, z_value = data_point
        if z_value is None:
            continue
        if hasattr(dt_value, "strftime"):
            label = dt_value.strftime("%d/%m")
        else:
            label = str(dt_value)
        try:
            numeric = float(z_value)
        except (TypeError, ValueError):
            continue
        zscore_series_points.append({"label": label, "value": numeric})

    entry_display = _metrics_display(entry_metrics_payload)
    current_display = _metrics_display(current_metrics_payload)

    entry_z_label = _fmt_metric(entry_zscore)
    current_z_label = _fmt_metric(current_zscore)

    z_delta = None
    z_delta_label = "--"
    if current_zscore is not None and entry_zscore is not None:
        try:
            z_delta = float(current_zscore) - float(entry_zscore)
            z_delta_label = f"{z_delta:+.2f}"
        except (TypeError, ValueError):
            z_delta = None
            z_delta_label = "--"

    trade_info = operation.as_trade_dict()
    trade_summary = {
        "sell": {
            "ticker": operation.sell_asset.ticker,
            "quantity": operation.sell_quantity,
            "price_label": _format_price(trade_info.get("sell", {}).get("price")),
            "value_label": _format_money(trade_info.get("sell", {}).get("value")),
        },
        "buy": {
            "ticker": operation.buy_asset.ticker,
            "quantity": operation.buy_quantity,
            "price_label": _format_price(trade_info.get("buy", {}).get("price")),
            "value_label": _format_money(trade_info.get("buy", {}).get("value")),
        },
        "net_label": _format_money(trade_info.get("net")),
        "capital_label": _format_money(trade_info.get("capital_allocated")),
    }

    entry_net_direction = ""
    if operation.net_value is not None:
        entry_net_direction = "recebe" if operation.net_value >= 0 else "paga"

    entry_prices = {
        "sell_qty_label": _fmt_int(operation.sell_quantity),
        "buy_qty_label": _fmt_int(operation.buy_quantity),
        "sell_price_label": _format_price(operation.sell_price),
        "buy_price_label": _format_price(operation.buy_price),
        "sell_total_label": _format_money(operation.sell_value),
        "buy_total_label": _format_money(operation.buy_value),
        "net_label": _format_money(operation.net_value),
        "net_direction_label": entry_net_direction,
    }

    sell_current_price, sell_price_updated, sell_price_source = _build_current_asset_price(
        operation.sell_asset
    )
    buy_current_price, buy_price_updated, buy_price_source = _build_current_asset_price(
        operation.buy_asset
    )
    latest_price_update = max(
        (dt for dt in (sell_price_updated, buy_price_updated) if dt is not None),
        default=None,
    )

    money_quant = Decimal("0.01")
    sell_qty_dec = Decimal(operation.sell_quantity)
    buy_qty_dec = Decimal(operation.buy_quantity)
    current_sell_total = (
        (sell_current_price * sell_qty_dec).quantize(money_quant)
        if sell_current_price is not None
        else None
    )
    current_buy_total = (
        (buy_current_price * buy_qty_dec).quantize(money_quant)
        if buy_current_price is not None
        else None
    )

    current_net_value = None
    if current_sell_total is not None and current_buy_total is not None:
        current_net_value = (current_sell_total - current_buy_total).quantize(money_quant)

    sell_pl = None
    if sell_current_price is not None:
        sell_pl = ((operation.sell_price - sell_current_price) * sell_qty_dec).quantize(
            money_quant
        )

    buy_pl = None
    if buy_current_price is not None:
        buy_pl = ((buy_current_price - operation.buy_price) * buy_qty_dec).quantize(
            money_quant
        )

    pl_total = None
    if sell_pl is not None and buy_pl is not None:
        pl_total = (sell_pl + buy_pl).quantize(money_quant)

    current_balance_value: Decimal | None = None
    if pl_total is not None:
        current_balance_value = pl_total
    elif current_net_value is not None and operation.net_value is not None:
        current_balance_value = (current_net_value - operation.net_value).quantize(
            money_quant
        )
    elif current_net_value is not None:
        current_balance_value = current_net_value
    final_direction_label = ""
    if current_balance_value is not None:
        final_direction_label = "recebe" if current_balance_value >= 0 else "paga"

    current_prices = {
        "updated_label": _format_detail_updated(latest_price_update),
        "sell": {
            "price_label": _format_price(sell_current_price),
            "source_label": _source_label(sell_price_source),
        },
        "buy": {
            "price_label": _format_price(buy_current_price),
            "source_label": _source_label(buy_price_source),
        },
        "sell_total_label": _format_money(current_sell_total),
        "buy_total_label": _format_money(current_buy_total),
        "sell_pl_label": _format_money(sell_pl),
        "buy_pl_label": _format_money(buy_pl),
        "net_label": _format_money(current_net_value),
        "pl_total_label": _format_money(pl_total),
        "final_label": _format_money(current_balance_value),
        "final_direction_label": final_direction_label,
    }

    pnl_summary = _build_pnl_summary(
        operation=operation,
        sell_live_price=sell_current_price,
        buy_live_price=buy_current_price,
        sell_price=operation.sell_price,
        buy_price=operation.buy_price,
    )

    try:
        opened_local = timezone.localtime(operation.opened_at)
    except Exception:
        opened_local = operation.opened_at

    return render(
        request,
        "core/operacao_encerrar.html",
        {
            "current": "home",
            "title": f"Encerrar operacao {operation.sell_asset.ticker} x {operation.buy_asset.ticker}",
        "operation": operation,
        "entry_snapshot": entry_snapshot,
        "latest_snapshot": latest_snapshot,
        "entry_metrics": entry_display,
        "current_metrics": current_display,
        "entry_zscore_label": entry_z_label,
        "current_zscore_label": current_z_label,
        "z_delta_label": z_delta_label,
        "is_delta_positive": bool(z_delta is not None and z_delta >= 0),
        "capital_label": _format_money(operation.capital_allocated),
        "net_label": _format_money(operation.net_value),
        "net_direction": "recebe" if operation.net_value is not None and operation.net_value >= 0 else "paga",
        "trade_info": trade_info,
        "current_metrics_payload": current_metrics_payload,
        "trade_summary": trade_summary,
        "current_prices": current_prices,
        "pnl_summary": pnl_summary,
        "entry_prices": entry_prices,
        "opened_label": opened_local.strftime("%d/%m/%Y %H:%M") if opened_local else "",
        "operation_date_label": operation.operation_date.strftime("%d/%m/%Y") if operation.operation_date else "",
        "window": operation.window,
        "zscore_series_points": zscore_series_points,
        },
    )


@login_required
def operacao_refresh(request, pk: int):
    if request.method != "GET":
        return JsonResponse({"ok": False, "detail": "Metodo nao suportado."}, status=405)

    operation = get_object_or_404(
        Operation.objects.select_related(
            "sell_asset",
            "buy_asset",
            "left_asset",
            "right_asset",
            "pair",
        ).prefetch_related(
            Prefetch(
                "metric_snapshots",
                queryset=OperationMetricSnapshot.objects.order_by("-reference_date"),
            )
        ),
        pk=pk,
        user=request.user,
        status=Operation.STATUS_OPEN,
    )

    entry_snapshot = None
    snapshots = list(operation.metric_snapshots.all())
    for snap in snapshots:
        if snap.snapshot_type == OperationMetricSnapshot.TYPE_OPEN:
            entry_snapshot = snap
            break

    entry_zscore = entry_snapshot.zscore if entry_snapshot and entry_snapshot.zscore is not None else operation.entry_zscore

    pair_ref = operation.pair if operation.pair else SimpleNamespace(
        left=operation.left_asset,
        right=operation.right_asset,
    )

    current_metrics_payload = {}
    current_zscore = None
    try:
        metrics_now = compute_pair_window_metrics(pair=pair_ref, window=operation.window)
    except Exception:
        metrics_now = None

    if isinstance(metrics_now, dict):
        current_metrics_payload = metrics_now
        raw_z = metrics_now.get("zscore")
        if raw_z is not None:
            try:
                current_zscore = float(raw_z)
            except (TypeError, ValueError):
                current_zscore = None

    current_metrics = _metrics_display(current_metrics_payload)
    entry_z_label = _fmt_metric(entry_zscore)
    current_z_label = _fmt_metric(current_zscore)

    z_delta = None
    z_delta_label = "--"
    is_delta_positive = False
    if entry_zscore is not None and current_zscore is not None:
        try:
            z_delta = float(current_zscore) - float(entry_zscore)
            z_delta_label = f"{z_delta:+.2f}"
            is_delta_positive = z_delta >= 0
        except (TypeError, ValueError):
            z_delta = None
            z_delta_label = "--"
            is_delta_positive = False

    sell_price, sell_price_updated, sell_price_source = _build_current_asset_price(operation.sell_asset)
    buy_price, buy_price_updated, buy_price_source = _build_current_asset_price(operation.buy_asset)
    latest_price_update = max(
        (dt for dt in (sell_price_updated, buy_price_updated) if dt is not None),
        default=None,
    )
    money_quant = Decimal("0.01")
    sell_qty_dec = Decimal(operation.sell_quantity)
    buy_qty_dec = Decimal(operation.buy_quantity)
    current_sell_total = (
        (sell_price * sell_qty_dec).quantize(money_quant) if sell_price is not None else None
    )
    current_buy_total = (
        (buy_price * buy_qty_dec).quantize(money_quant) if buy_price is not None else None
    )

    current_net_value = None
    if current_sell_total is not None and current_buy_total is not None:
        current_net_value = (current_sell_total - current_buy_total).quantize(money_quant)

    sell_pl = None
    if sell_price is not None:
        sell_pl = ((operation.sell_price - sell_price) * sell_qty_dec).quantize(money_quant)

    buy_pl = None
    if buy_price is not None:
        buy_pl = ((buy_price - operation.buy_price) * buy_qty_dec).quantize(money_quant)

    pl_total = None
    if sell_pl is not None and buy_pl is not None:
        pl_total = (sell_pl + buy_pl).quantize(money_quant)

    current_balance_value: Decimal | None = None
    if pl_total is not None:
        current_balance_value = pl_total
    elif current_net_value is not None and operation.net_value is not None:
        current_balance_value = (current_net_value - operation.net_value).quantize(money_quant)
    elif current_net_value is not None:
        current_balance_value = current_net_value
    final_direction_label = ""
    if current_balance_value is not None:
        final_direction_label = "recebe" if current_balance_value >= 0 else "paga"

    current_prices = {
        "updated_label": _format_detail_updated(latest_price_update),
        "sell": {
            "price_label": _format_price(sell_price),
            "source_label": _source_label(sell_price_source),
        },
        "buy": {
            "price_label": _format_price(buy_price),
            "source_label": _source_label(buy_price_source),
        },
        "sell_total_label": _format_money(current_sell_total),
        "buy_total_label": _format_money(current_buy_total),
        "sell_pl_label": _format_money(sell_pl),
        "buy_pl_label": _format_money(buy_pl),
        "net_label": _format_money(current_net_value),
        "pl_total_label": _format_money(pl_total),
        "final_label": _format_money(current_balance_value),
        "final_direction_label": final_direction_label,
    }

    pnl_summary = _build_pnl_summary(
        operation=operation,
        sell_live_price=sell_price,
        buy_live_price=buy_price,
        sell_price=operation.sell_price,
        buy_price=operation.buy_price,
    )

    return JsonResponse(
        {
            "ok": True,
            "current_metrics": current_metrics,
            "current_zscore_label": current_z_label,
            "entry_zscore_label": entry_z_label,
            "z_delta_label": z_delta_label,
            "is_delta_positive": is_delta_positive,
            "current_prices": current_prices,
            "pnl_summary": pnl_summary,
        }
    )
