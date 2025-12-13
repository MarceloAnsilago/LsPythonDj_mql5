from __future__ import annotations

from dataclasses import asdict
from datetime import date
from decimal import Decimal
from typing import Iterable, List
import logging

from django.db import transaction

from acoes.models import Asset
from longshort.services.quotes import bulk_update_quotes
from longshort.services.price_provider import get_daily_prices
from pairs.models import Pair
from pairs.services.scan import WindowRow, scan_pair_windows

from .models import PairScanResult, PriceHistory

LOGGER = logging.getLogger(__name__)


def atualizar_cotacoes(
    dia: date,
    *,
    assets: Iterable[Asset] | None = None,
    period: str = "2y",
    interval: str = "1d",
    use_stooq: bool = False,
) -> dict[str, int]:
    """
    Baixa as cotacoes pendentes e retorna estatisticas basicas.
    """
    todos_os_ativos = list(assets or Asset.objects.filter(is_active=True))
    inserted_assets, rows_inserted = bulk_update_quotes(
        todos_os_ativos,
        period=period,
        interval=interval,
        use_stooq=use_stooq,
    )
    LOGGER.info(
        "Cotacoes %s: ativos processeados=%d ativos atualizados=%d linhas inseridas=%d",
        dia,
        len(todos_os_ativos),
        inserted_assets,
        rows_inserted,
    )
    return {
        "date": dia.isoformat(),
        "assets": len(todos_os_ativos),
        "updated_assets": inserted_assets,
        "rows_inserted": rows_inserted,
        "use_stooq": use_stooq,
    }


def _sync_price_history(dia: date) -> dict[str, int]:
    """
    Mantem a tabela PriceHistory alinhada com a fonte diaria (QuoteDaily ou DailyPrice).
    """
    assets = list(Asset.objects.filter(is_active=True))
    created = 0
    updated = 0
    records = 0

    for asset in assets:
        prices = get_daily_prices(asset, start_date=dia, end_date=dia)
        if not prices:
            continue
        row = prices[-1]
        close_px = row.get("close")
        if close_px is None:
            continue

        defaults = {
            "close": Decimal(str(close_px)),
            "source": "QuoteDaily",
        }
        _, was_created = PriceHistory.objects.update_or_create(
            asset=asset,
            date=dia,
            defaults=defaults,
        )
        records += 1
        if was_created:
            created += 1
        else:
            updated += 1
    LOGGER.info(
        "PriceHistory %s: registros=%d criados=%d atualizados=%d",
        dia,
        records,
        created,
        updated,
    )
    return {
        "date": dia.isoformat(),
        "records": records,
        "created": created,
        "updated": updated,
    }


def _row_to_dict(row: WindowRow | None) -> dict | None:
    if row is None:
        return None
    return {
        "window": row.window,
        "adf_pct": row.adf_pct,
        "adf_pvalue": row.adf_pvalue,
        "beta": row.beta,
        "zscore": row.zscore,
        "half_life": row.half_life,
        "corr30": row.corr30,
        "corr60": row.corr60,
        "status": row.status,
        "message": row.message,
    }


def rodar_scan_pares(dia: date, *, pairs: Iterable[Pair] | None = None) -> dict:
    """
    Executa o scanner de janelas e persiste os resultados por par e data.
    """
    candidatos = list(pairs or Pair.objects.order_by("id"))
    created = 0
    updated = 0
    errors: List[str] = []

    for pair in candidatos:
        try:
            scan_payload = scan_pair_windows(pair)
            rows = [_row_to_dict(row) for row in scan_payload.get("rows", [])]
            best_row = _row_to_dict(scan_payload.get("best"))
            thresholds = scan_payload.get("thresholds")
            with transaction.atomic():
                obj, was_created = PairScanResult.objects.update_or_create(
                    pair=pair,
                    run_date=dia,
                    defaults={
                        "best_window": best_row["window"] if best_row else None,
                        "status": best_row["status"] if best_row else "reprovado",
                        "message": (best_row["message"] if best_row else "sem resultado")[:200],
                        "rows": rows,
                        "best_row": best_row,
                        "thresholds": asdict(thresholds) if thresholds else {},
                    },
                )
            if was_created:
                created += 1
            else:
                updated += 1
        except Exception as exc:  # pragma: no cover - defensive
            msg = f"Pair {pair.id}: {exc}"
            LOGGER.exception("Erro escaneando par %s: %s", pair, exc)
            errors.append(msg)

    return {
        "date": dia.isoformat(),
        "pairs_processed": len(candidatos),
        "created": created,
        "updated": updated,
        "errors": errors,
    }


def rodar_scan_diario(dia: date, *, use_stooq: bool = False, logger: logging.Logger | None = None) -> dict:
    log = logger or LOGGER
    log.info("Iniciando scan diario para %s", dia)
    quote_summary = atualizar_cotacoes(dia, use_stooq=use_stooq)
    history_summary = _sync_price_history(dia)
    scan_summary = rodar_scan_pares(dia)
    log.info("Scan diario %s concluido (pares=%s)", dia, scan_summary["pairs_processed"])
    return {
        "date": dia.isoformat(),
        "quotes": quote_summary,
        "history": history_summary,
        "pairs": scan_summary,
    }
