from __future__ import annotations

from datetime import datetime, time, date

from django.core.management import call_command
from django.core.management.base import BaseCommand, CommandError
from django.db import models
from django.db.models.functions import Coalesce
from django.utils import timezone

from acoes.models import Asset
from cotacoes.models import QuoteDaily
from mt5api.models import LiveTick


class Command(BaseCommand):
    help = "Consolida ticks MT5 em OHLC e valida se todos os ativos use_mt5 receberam dados."

    def add_arguments(self, parser) -> None:
        parser.add_argument(
            "--date",
            dest="date",
            help="Data alvo no formato YYYY-MM-DD. Padrao: hoje (fuso local).",
        )
        parser.add_argument(
            "--min-ticks",
            type=int,
            default=2,
            help="Quantidade minima de ticks para considerar um ativo valido (default: 2).",
        )

    def handle(self, *args, **options) -> None:
        raw_date = options.get("date")
        target_date: date
        if raw_date:
            try:
                target_date = datetime.strptime(raw_date, "%Y-%m-%d").date()
            except ValueError as exc:
                raise CommandError(f"Formato invalido de data: {exc}") from exc
        else:
            target_date = timezone.localdate()

        if target_date >= timezone.localdate():
            raise CommandError("Somente candles D1 fechados (date < hoje) podem ser consolidados.")

        self.stdout.write(f"[mt5sync_daily] Consolidando ticks para {target_date}...")
        call_command("agg_daily_prices", date=target_date.isoformat())

        tz = timezone.get_default_timezone()
        start_dt = timezone.make_aware(datetime.combine(target_date, time.min), tz)
        end_dt = timezone.make_aware(datetime.combine(target_date, time.max), tz)

        assets = list(Asset.objects.filter(use_mt5=True, is_active=True))
        if not assets:
            self.stdout.write("Nenhum ativo marcado com use_mt5=True.")
            return

        min_ticks = max(1, int(options.get("min_ticks") or 2))

        ticks_qs = (
            LiveTick.objects.annotate(effective_dt=Coalesce("as_of", "timestamp"))
            .filter(
                models.Q(as_of__isnull=False, as_of__gte=start_dt, as_of__lte=end_dt)
                | models.Q(as_of__isnull=True, timestamp__gte=start_dt, timestamp__lte=end_dt),
            )
            .values("ticker")
            .annotate(count=models.Count("id"))
        )
        tick_counts = {row["ticker"]: row["count"] for row in ticks_qs}

        daily_map: dict[int, QuoteDaily] = {}
        for qd in QuoteDaily.objects.select_related("asset").filter(date=target_date):
            # QuoteDaily tem unique_together (asset, date); manter ultimo caso haja lixo.
            daily_map[qd.asset_id] = qd

        warnings: list[str] = []
        ok_count = 0

        for asset in assets:
            tick_count = tick_counts.get(asset.ticker, 0)
            daily = daily_map.get(asset.id)
            if daily is None:
                warnings.append(
                    f"{asset.ticker}: sem QuoteDaily para {target_date} (ticks={tick_count}, min_ticks={min_ticks}, source=daily)"
                )
                continue
            if tick_count < min_ticks:
                warnings.append(
                    f"{asset.ticker}: poucos ticks ({tick_count}<min_ticks={min_ticks}) para {target_date} (source=ticks)"
                )
            else:
                ok_count += 1

        if warnings:
            self.stdout.write(self.style.WARNING("Avisos:"))
            for msg in warnings:
                self.stdout.write(f"- {msg}")
        self.stdout.write(
            self.style.SUCCESS(
                f"mt5sync_daily concluido para {target_date}: ok={ok_count} avisos={len(warnings)} ativos_mt5={len(assets)}"
            )
        )
