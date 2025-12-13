from __future__ import annotations

import time

from dataclasses import dataclass
from itertools import combinations
import logging
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, TYPE_CHECKING

from django.conf import settings
from django.db.models import Q
from django.utils import timezone

from acoes.models import Asset
from pairs.constants import (
    DEFAULT_ADF_MIN,
    DEFAULT_BASE_WINDOW,
    DEFAULT_WINDOWS,
    DEFAULT_ZSCORE_ABS_MIN,
    DEFAULT_HALF_LIFE_MAX,
)
from pairs.models import Pair

from longshort.services.metrics import CandleUniverse, compute_pair_window_metrics
from longshort.services.candles_loader import load_candles_for_universe

if TYPE_CHECKING:  # pragma: no cover - type hints only
    from pairs.models import UserMetricsConfig


BASE_WINDOW = DEFAULT_BASE_WINDOW

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Thresholds:
    adf_min: float = DEFAULT_ADF_MIN
    zscore_abs_min: float = DEFAULT_ZSCORE_ABS_MIN
    half_life_max: Optional[float] = (
        DEFAULT_HALF_LIFE_MAX if DEFAULT_HALF_LIFE_MAX and DEFAULT_HALF_LIFE_MAX > 0 else None
    )


ProgressEvent = Dict[str, Any]


@dataclass
class WindowRow:
    window: int
    adf_pct: Optional[float]
    adf_pvalue: Optional[float]
    beta: Optional[float]
    zscore: Optional[float]
    half_life: Optional[float]
    corr30: Optional[float]
    corr60: Optional[float]
    status: str
    message: str


def _resolve_windows(
    windows: Sequence[int] | None,
    config: "UserMetricsConfig" | None = None,
) -> List[int]:
    """
    Normalise an incoming windows list (or user config) into a unique positive list.
    Order is preserved and duplicates or invalid values are discarded.
    """
    if windows is not None:
        source: Iterable[int] = windows
    elif config is not None:
        source = config.windows_list()
    else:
        source = DEFAULT_WINDOWS

    resolved: List[int] = []
    for value in source:
        try:
            num = int(value)
        except (TypeError, ValueError):
            continue
        if num <= 0 or num in resolved:
            continue
        resolved.append(num)

    if not resolved:
        resolved = list(DEFAULT_WINDOWS)
    return resolved


def get_thresholds(
    overrides: Dict[str, Any] | None = None,
    *,
    config: "UserMetricsConfig" | None = None,
) -> Thresholds:
    """
    Build Thresholds honouring the default values, optional Django settings overrides,
    user specific configuration, and explicit overrides (in that precedence order).
    """
    data: Dict[str, Any] = {}
    cfg = getattr(settings, "PAIRS_THRESHOLDS", None)
    if isinstance(cfg, dict):
        data.update(cfg)

    if config is not None:
        if getattr(config, "adf_min", None) is not None:
            data["adf_min"] = config.adf_min
        if getattr(config, "zscore_abs_min", None) is not None:
            data["zscore_abs_min"] = config.zscore_abs_min
        if getattr(config, "half_life_max", None) is not None:
            data["half_life_max"] = config.half_life_max

    if overrides:
        data.update(overrides)

    cleaned: Dict[str, Any] = {}
    for key in ("adf_min", "zscore_abs_min", "half_life_max"):
        if key in data:
            try:
                cleaned[key] = float(data[key])
            except (TypeError, ValueError):
                pass

    if "half_life_max" in cleaned:
        if cleaned["half_life_max"] <= 0:
            cleaned["half_life_max"] = None

    try:
        return Thresholds(**cleaned)
    except TypeError:
        return Thresholds()


def _tie_break(best: WindowRow | None, candidate: WindowRow | None) -> WindowRow | None:
    if candidate is None:
        return best
    if best is None:
        return candidate
    # Higher ADF% -> higher |Z| -> lower |beta|
    cand_adf = candidate.adf_pct or -1.0
    best_adf = best.adf_pct or -1.0
    if cand_adf != best_adf:
        return candidate if cand_adf > best_adf else best
    cand_z = abs(candidate.zscore or 0.0)
    best_z = abs(best.zscore or 0.0)
    if cand_z != best_z:
        return candidate if cand_z > best_z else best
    cand_beta = abs(candidate.beta or 0.0)
    best_beta = abs(best.beta or 0.0)
    return candidate if cand_beta < best_beta else best


def scan_pair_windows(
    pair: Pair,
    windows: Sequence[int] | None = None,
    thresholds: Thresholds | None = None,
    *,
    metrics_config: "UserMetricsConfig" | None = None,
    max_age_seconds: int | None = None,
) -> Dict[str, Any]:
    """
    Evaluate the configured windows for a pair (Grid B) and persist the summary on the model.
    """
    resolved_windows = _resolve_windows(windows, metrics_config)
    thresholds = thresholds or get_thresholds(config=metrics_config)

    # Attempt short-cache reuse
    if (
        max_age_seconds
        and pair.scan_cache_json
        and pair.scan_cached_at
        and (timezone.now() - pair.scan_cached_at).total_seconds() <= max_age_seconds
    ):
        cached_scan = pair.scan_cache_json.get("scan") if isinstance(pair.scan_cache_json, dict) else None
        if cached_scan and cached_scan.get("windows") == resolved_windows:
            cached_rows = []
            for r in cached_scan.get("rows", []):
                cached_rows.append(
                    WindowRow(
                        window=r.get("window"),
                        adf_pct=r.get("adf_pct"),
                        adf_pvalue=r.get("adf_pvalue"),
                        beta=r.get("beta"),
                        zscore=r.get("zscore"),
                        half_life=r.get("half_life"),
                        corr30=r.get("corr30"),
                        corr60=r.get("corr60"),
                        status=r.get("status"),
                        message=r.get("message"),
                    )
                )
            best_window = cached_scan.get("best_window")
            best_row = next((row for row in cached_rows if row.window == best_window and row.status == "ok"), None)
            return {
                "rows": cached_rows,
                "best": best_row,
                "windows": resolved_windows,
                "thresholds": thresholds,
            }

    candles: CandleUniverse | None = None
    if resolved_windows:
        max_window = max(resolved_windows)
        try:
            candles = load_candles_for_universe(
                [pair.left_id, pair.right_id],
                lookback_windows=max_window,
            )
        except Exception:
            candles = None

    rows: List[WindowRow] = []
    best: Optional[WindowRow] = None

    for window_value in resolved_windows:
        status = "pendente"
        message = ""
        adf_pct = None
        adf_pvalue = None
        beta = None
        zscore = None
        half_life = None
        corr30 = None
        corr60 = None

        try:
            if compute_pair_window_metrics is None:
                raise RuntimeError("Funcao compute_pair_window_metrics nao encontrada.")

            metrics = compute_pair_window_metrics(pair=pair, window=window_value, candles=candles) or {}

            adf_pvalue_raw = metrics.get("adf_pvalue")
            if adf_pvalue_raw is not None:
                adf_pvalue = float(adf_pvalue_raw)
                adf_pct = (1.0 - adf_pvalue) * 100.0

            beta_raw = metrics.get("beta")
            beta = float(beta_raw) if beta_raw is not None else None
            zscore_raw = metrics.get("zscore")
            zscore = float(zscore_raw) if zscore_raw is not None else None
            half_life_raw = metrics.get("half_life")
            half_life = float(half_life_raw) if half_life_raw is not None else None
            corr30_raw = metrics.get("corr30")
            corr30 = float(corr30_raw) if corr30_raw is not None else None
            corr60_raw = metrics.get("corr60")
            corr60 = float(corr60_raw) if corr60_raw is not None else None

            samples = metrics.get("n_samples")
            samples_int = int(samples) if samples is not None else None

            if samples_int is not None and samples_int < 60:
                status, message = "reprovado", "Amostra insuficiente (N<60)"
            elif adf_pct is None:
                status, message = "reprovado", "ADF% indisponivel"
            elif adf_pct < thresholds.adf_min:
                status, message = "reprovado", f"ADF {adf_pct:.1f}% < minimo {thresholds.adf_min:.1f}%"
            elif zscore is None:
                status, message = "reprovado", "Z-score indisponivel"
            elif abs(zscore) < thresholds.zscore_abs_min:
                status, message = "reprovado", f"|Z| < {thresholds.zscore_abs_min:.1f}"
            elif thresholds.half_life_max is not None:
                if half_life is None:
                    status, message = "reprovado", "Half-life indisponivel"
                elif half_life > thresholds.half_life_max:
                    status, message = (
                        "reprovado",
                        f"Half-life {half_life:.1f}d > maximo {thresholds.half_life_max:.1f}d",
                    )
            else:
                status, message = "ok", "OK"
        except Exception as exc:  # pragma: no cover - defensive
            status, message = "erro", f"erro: {exc}"

        row = WindowRow(
            window=window_value,
            adf_pct=adf_pct,
            adf_pvalue=adf_pvalue,
            beta=beta,
            zscore=zscore,
            half_life=half_life,
            corr30=corr30,
            corr60=corr60,
            status=status,
            message=message,
        )
        rows.append(row)
        if status == "ok":
            best = _tie_break(best, row)

    cache_rows = [
        {
            "window": r.window,
            "adf_pct": r.adf_pct,
            "adf_pvalue": r.adf_pvalue,
            "beta": r.beta,
            "zscore": r.zscore,
            "half_life": r.half_life,
            "corr30": r.corr30,
            "corr60": r.corr60,
            "status": r.status,
            "message": r.message,
        }
        for r in rows
    ]

    scan_payload = {
        "rows": cache_rows,
        "best_window": best.window if best else None,
        "windows": resolved_windows,
        "thresholds": {
            "adf_min": thresholds.adf_min,
            "zscore_abs_min": thresholds.zscore_abs_min,
            "half_life_max": thresholds.half_life_max,
        },
    }

    sc = pair.scan_cache_json or {}
    sc["scan"] = scan_payload
    pair.scan_cache_json = sc
    pair.scan_cached_at = timezone.now()
    pair.save(update_fields=["scan_cache_json", "scan_cached_at"])

    return {"rows": rows, "best": best, "windows": resolved_windows, "thresholds": thresholds}


def _compute_base_for_pair(
    pair: Pair,
    window: int = DEFAULT_BASE_WINDOW,
    thresholds: Thresholds | None = None,
    *,
    metrics_config: "UserMetricsConfig" | None = None,
    candles: CandleUniverse | None = None,
    pair_label: str | None = None,
) -> tuple[bool, Dict[str, Any], str]:
    """
    Compute the Grid A metrics for a pair and decide approval based on thresholds.
    """
    if compute_pair_window_metrics is None:
        return False, {}, "Funcao compute_pair_window_metrics nao encontrada."

    thresholds = thresholds or get_thresholds(config=metrics_config)

    try:
        compute_start = time.perf_counter()
        metrics = compute_pair_window_metrics(
            pair=pair, window=window, candles=candles
        ) or {}
        compute_end = time.perf_counter()
        label = pair_label or f"{pair.left_id}-{pair.right_id}"
        ready = bool(metrics.get("ready_for_approval"))
        skip_reason = metrics.get("skip_reason")
        if getattr(settings, "PAIRS_DEBUG_LOG", False):
            logger.debug(
                "pair=%s compute_ms=%.1f ready=%s skip_reason=%s",
                label,
                (compute_end - compute_start) * 1000,
                ready,
                skip_reason,
            )

        if not ready:
            reason = metrics.get("skip_reason") or "Filtros iniciais nao atendidos"
            return False, {}, reason

        adf_pvalue_raw = metrics.get("adf_pvalue")
        adf_pvalue = float(adf_pvalue_raw) if adf_pvalue_raw is not None else None
        adf_pct = (1.0 - adf_pvalue) * 100.0 if adf_pvalue is not None else None
        beta_raw = metrics.get("beta")
        beta = float(beta_raw) if beta_raw is not None else None
        zscore_raw = metrics.get("zscore")
        zscore = float(zscore_raw) if zscore_raw is not None else None
        half_life_raw = metrics.get("half_life")
        half_life = float(half_life_raw) if half_life_raw is not None else None
        corr30_raw = metrics.get("corr30")
        corr30 = float(corr30_raw) if corr30_raw is not None else None
        corr60_raw = metrics.get("corr60")
        corr60 = float(corr60_raw) if corr60_raw is not None else None
        samples = metrics.get("n_samples")
        samples_int = int(samples) if samples is not None else None

        if samples_int is not None and samples_int < 60:
            return False, {}, "Amostra insuficiente (N<60)"
        if adf_pct is None:
            return False, {}, "ADF% indisponivel"
        if adf_pct < thresholds.adf_min:
            return False, {}, f"ADF {adf_pct:.1f}% < minimo {thresholds.adf_min:.1f}%"
        if zscore is None:
            return False, {}, "Z-score indisponivel"
        if abs(zscore) < thresholds.zscore_abs_min:
            return False, {}, f"|Z| < {thresholds.zscore_abs_min:.1f}"
        if thresholds.half_life_max is not None:
            if half_life is None:
                return False, {}, "Half-life indisponivel"
            if half_life > thresholds.half_life_max:
                return False, {}, f"Half-life {half_life:.1f}d > maximo {thresholds.half_life_max:.1f}d"

        payload = {
            "window": window,
            "adf_pvalue": adf_pvalue,
            "adf_pct": adf_pct,
            "beta": beta,
            "zscore": zscore,
            "half_life": half_life,
            "corr30": corr30,
            "corr60": corr60,
            "n_samples": samples_int,
            "status": "ok",
        }
        return True, payload, "OK"
    except Exception as exc:  # pragma: no cover - defensive
        return False, {}, f"erro: {exc}"


def build_pairs_base(
    window: int = DEFAULT_BASE_WINDOW,
    limit_assets: int | None = None,
    progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
    thresholds: Thresholds | None = None,
    *,
    metrics_config: "UserMetricsConfig" | None = None,
) -> Dict[str, Any]:
    """
    Scan Asset combinations (Grid A) and persist only the approved pairs.
    """
    try:
        qs = Asset.objects.filter(is_active=True)
        if not qs.exists():
            qs = Asset.objects.all()
    except Exception:  # pragma: no cover - defensive
        qs = Asset.objects.all()

    if limit_assets:
        qs = qs.order_by("id")[:limit_assets]

    assets = list(qs)
    n_assets = len(assets)
    total = (n_assets * (n_assets - 1)) // 2 if n_assets >= 2 else 0

    asset_ids = [asset.id for asset in assets]
    existing_pairs: dict[tuple[int, int], Pair] = {}
    if asset_ids:
        pairs_qs = Pair.objects.filter(
            Q(left_id__in=asset_ids, right_id__in=asset_ids)
        ).only("id", "left_id", "right_id", "scan_cache_json", "scan_cached_at")
        for existing in pairs_qs:
            key = tuple(sorted((existing.left_id, existing.right_id)))
            existing_pairs[key] = existing
    processed = 0

    created = 0
    updated = 0
    approved_ids: List[int] = []
    errors: List[str] = []

    t0 = time.perf_counter()
    thresholds = thresholds or get_thresholds(config=metrics_config)

    candles_for_assets: "CandleUniverse" | None = None
    if assets:
        asset_ids = [asset.id for asset in assets]
        if asset_ids:
            try:
                candles_for_assets = load_candles_for_universe(
                    asset_ids,
                    lookback_windows=window,
                )
            except Exception:
                candles_for_assets = None
    t1 = time.perf_counter()

    for left, right in combinations(assets, 2):
        processed += 1
        if progress_cb:
            progress_cb(
                {
                    "phase": "iter",
                    "i": processed,
                    "total": total,
                    "left": getattr(left, "ticker", str(left)),
                    "right": getattr(right, "ticker", str(right)),
                    "window": window,
                }
            )

        key = tuple(sorted((left.id, right.id)))
        existing_pair = existing_pairs.get(key)
        was_created = False
        temp_pair = existing_pair or Pair(left=left, right=right, base_window=window)

        try:
            pair_label = f"{getattr(left, 'ticker', left.id)}-{getattr(right, 'ticker', right.id)}"
            pair_start = time.perf_counter()
            approved, base_payload, message = _compute_base_for_pair(
                temp_pair,
                window=window,
                thresholds=thresholds,
                metrics_config=metrics_config,
                candles=candles_for_assets,
                pair_label=pair_label,
            )
            pair_end = time.perf_counter()
            event = {
                "phase": "pair",
                "pair_label": pair_label,
                "status": "approved" if approved else "reprovado",
                "message": "OK" if approved else message,
                "window": window,
                "approved": approved,
                "i": processed,
                "total": total,
                "compute_ms": (pair_end - pair_start) * 1000,
            }
            if progress_cb:
                progress_cb(event)

            if approved:
                if existing_pair is None:
                    temp_pair.save()
                    existing_pairs[key] = temp_pair
                    pair = temp_pair
                    was_created = True
                else:
                    pair = existing_pair

                sc = pair.scan_cache_json or {}
                sc["base"] = base_payload
                pair.scan_cache_json = sc
                pair.scan_cached_at = timezone.now()
                pair.save(update_fields=["scan_cache_json", "scan_cached_at"])

                approved_ids.append(pair.id)
                if was_created:
                    created += 1
                else:
                    updated += 1
            else:
                if existing_pair is not None:
                    sc = existing_pair.scan_cache_json or {}
                    sc["base"] = {
                        "window": window,
                        "status": "reprovado",
                        "message": message,
                    }
                    existing_pair.scan_cache_json = sc
                    existing_pair.scan_cached_at = timezone.now()
                    existing_pair.save(update_fields=["scan_cache_json", "scan_cached_at"])

        except Exception as exc:  # pragma: no cover - defensive
            errors.append(
                f"Pair {getattr(left, 'ticker', '?')}-{getattr(right, 'ticker', '?')}: {exc}"
            )

    if progress_cb:
        progress_cb({"phase": "done", "i": processed, "total": total, "window": window})
    t2 = time.perf_counter()
    if getattr(settings, "PAIRS_DEBUG_LOG", False):
        logger.debug("fetch_universe_ms=%.1f total_compute_ms=%.1f", (t1 - t0) * 1000, (t2 - t1) * 1000)

    return {
        "created": created,
        "updated": updated,
        "approved_ids": approved_ids,
        "errors": errors,
    }


def hunt_pairs_until_found(
    windows_desc: Sequence[int] | None = None,
    *,
    source: str = "assets",
    limit_assets: int | None = None,
    progress_cb: Optional[Callable[[ProgressEvent], None]] = None,
    thresholds: Thresholds | None = None,
    metrics_config: "UserMetricsConfig" | None = None,
    wait_for_next_window: Optional[Callable[[int, int, Sequence[int]], bool]] = None,
) -> Dict[str, Any]:
    """
    Iterate over a descending list of windows trying to approve pairs.
    Stops once at least one pair is approved.
    """
    if windows_desc is None:
        if metrics_config is not None:
            window_sequence = metrics_config.windows_descending()
        else:
            window_sequence = sorted(DEFAULT_WINDOWS, reverse=True)
    else:
        window_sequence = _resolve_windows(windows_desc, None)

    scanned: List[int] = []
    all_errors: List[str] = []
    thresholds = thresholds or get_thresholds(config=metrics_config)

    if source == "assets":
        if not window_sequence:
            if progress_cb:
                progress_cb({"phase": "done", "window": None, "approved": 0})
            return {
                "found": False,
                "window": None,
                "approved_ids": [],
                "errors": all_errors,
                "scanned_windows": scanned,
                "cancelled": False,
            }

        for idx, window_value in enumerate(window_sequence):
            scanned.append(window_value)
            if progress_cb:
                progress_cb({"phase": "window_start", "window": window_value})

            result = build_pairs_base(
                window=window_value,
                limit_assets=limit_assets,
                progress_cb=progress_cb,
                thresholds=thresholds,
                metrics_config=metrics_config,
            )

            if result.get("errors"):
                all_errors.extend(result["errors"])

            approved = result.get("approved_ids", [])
            if approved:
                if progress_cb:
                    progress_cb(
                        {
                            "phase": "done",
                            "window": window_value,
                            "approved": len(approved),
                        }
                    )
                return {
                    "found": True,
                    "window": window_value,
                    "approved_ids": approved,
                    "errors": all_errors,
                    "scanned_windows": scanned,
                    "cancelled": False,
                }
            elif idx < len(window_sequence) - 1:
                # Nenhum aprovado nesta janela; antes de seguir pede confirmacao do usuario
                next_window = window_sequence[idx + 1]
                if progress_cb:
                    progress_cb(
                        {
                            "phase": "window_next",
                            "window": window_value,
                            "next_window": next_window,
                            "approved": 0,
                            "state": "waiting" if wait_for_next_window else "running",
                        }
                    )
                if wait_for_next_window:
                    should_continue = wait_for_next_window(window_value, next_window, list(scanned))
                    if not should_continue:
                        if progress_cb:
                            progress_cb(
                                {
                                    "phase": "cancelled",
                                    "window": window_value,
                                    "approved": 0,
                                    "state": "done",
                                }
                            )
                        return {
                            "found": False,
                            "window": window_value,
                            "approved_ids": [],
                            "errors": all_errors,
                            "scanned_windows": scanned,
                            "cancelled": True,
                        }

        if progress_cb:
            progress_cb({"phase": "done", "window": None, "approved": 0})
        return {
            "found": False,
            "window": None,
            "approved_ids": [],
            "errors": all_errors,
            "scanned_windows": scanned,
            "cancelled": False,
        }

    if source == "existing_pairs":
        qs = Pair.objects.all().order_by("id")
        approved_ids: List[int] = []
        for idx, window_value in enumerate(window_sequence):
            scanned.append(window_value)
            approved_ids.clear()
            for pair in qs:
                try:
                    ok, base_payload, message = _compute_base_for_pair(
                        pair,
                        window=window_value,
                        thresholds=thresholds,
                        metrics_config=metrics_config,
                        pair_label=f"{pair.left_id}-{pair.right_id}",
                    )
                    sc = pair.scan_cache_json or {}
                    if ok:
                        sc["base"] = base_payload
                        pair.scan_cache_json = sc
                        pair.scan_cached_at = timezone.now()
                        pair.save(update_fields=["scan_cache_json", "scan_cached_at"])
                        approved_ids.append(pair.id)
                    else:
                        sc["base"] = {
                            "window": window_value,
                            "status": "reprovado",
                            "message": message,
                        }
                        pair.scan_cache_json = sc
                        pair.scan_cached_at = timezone.now()
                        pair.save(update_fields=["scan_cache_json", "scan_cached_at"])
                except Exception as exc:  # pragma: no cover - defensive
                    all_errors.append(f"Pair {pair.id}: {exc}")

            if approved_ids:
                if progress_cb:
                    progress_cb(
                        {
                            "phase": "done",
                            "window": window_value,
                            "approved": len(approved_ids),
                        }
                    )
                return {
                    "found": True,
                    "window": window_value,
                    "approved_ids": list(approved_ids),
                    "errors": all_errors,
                    "scanned_windows": scanned,
                    "cancelled": False,
                }

            if idx < len(window_sequence) - 1 and wait_for_next_window:
                next_window = window_sequence[idx + 1]
                if progress_cb:
                    progress_cb(
                        {
                            "phase": "window_next",
                            "window": window_value,
                            "next_window": next_window,
                            "approved": 0,
                            "state": "waiting",
                        }
                    )
                should_continue = wait_for_next_window(window_value, next_window, list(scanned))
                if not should_continue:
                    if progress_cb:
                        progress_cb(
                            {
                                "phase": "cancelled",
                                "window": window_value,
                                "approved": 0,
                                "state": "done",
                            }
                        )
                    return {
                        "found": False,
                        "window": window_value,
                        "approved_ids": [],
                        "errors": all_errors,
                        "scanned_windows": scanned,
                        "cancelled": True,
                    }

        if progress_cb:
            progress_cb({"phase": "done", "window": None, "approved": 0})
        return {
            "found": False,
            "window": None,
            "approved_ids": [],
            "errors": all_errors,
            "scanned_windows": scanned,
            "cancelled": False,
        }

    return {
        "found": False,
        "window": None,
        "approved_ids": [],
        "errors": [f"source invalido: {source}"],
        "scanned_windows": [],
        "cancelled": False,
    }
