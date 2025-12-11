from __future__ import annotations

from datetime import datetime, time, date

from django.core.management import call_command
from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from acoes.models import Asset
from mt5api.models import DailyPrice, LiveTick


class Command(BaseCommand):
    help = "Consolida ticks MT5 em OHLC e valida se todos os ativos use_mt5 receberam dados."

    def add_arguments(self, parser) -> None:
        parser.add_argument(
            "--date",
            dest="date",
            help="Data alvo no formato YYYY-MM-DD. Padrao: hoje (fuso local).",
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

        self.stdout.write(f"[mt5sync_daily] Consolidando ticks para {target_date}...")
        call_command("agg_daily_prices", date=target_date.isoformat())

        tz = timezone.get_default_timezone()
        start_dt = timezone.make_aware(datetime.combine(target_date, time.min), tz)
        end_dt = timezone.make_aware(datetime.combine(target_date, time.max), tz)

        assets = list(Asset.objects.filter(use_mt5=True, is_active=True))
        if not assets:
            self.stdout.write("Nenhum ativo marcado com use_mt5=True.")
            return

        warnings: list[str] = []
        ok_count = 0

        for asset in assets:
            tick_count = LiveTick.objects.filter(
                ticker=asset.ticker,
                timestamp__gte=start_dt,
                timestamp__lte=end_dt,
            ).count()
            daily = DailyPrice.objects.filter(ticker=asset.ticker, date=target_date).first()
            if daily is None:
                warnings.append(f"{asset.ticker}: sem DailyPrice para {target_date} (ticks={tick_count})")
                continue
            if tick_count < 2:
                warnings.append(f"{asset.ticker}: poucos ticks ({tick_count}) para {target_date}")
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
