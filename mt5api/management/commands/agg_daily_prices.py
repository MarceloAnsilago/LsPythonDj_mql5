from __future__ import annotations

from datetime import datetime, time, date

from django.core.management.base import BaseCommand, CommandError
from django.db.models import Max, Min
from django.utils import timezone

from mt5api.models import DailyPrice, LiveTick


class Command(BaseCommand):
    help = "Consolida ticks intradiarios em OHLC diario."

    def add_arguments(self, parser) -> None:
        parser.add_argument(
            "--date",
            dest="date",
            help="Data alvo no formato YYYY-MM-DD. Padrao: hoje no fuso configurado.",
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

        tz = timezone.get_default_timezone()
        start_dt = timezone.make_aware(datetime.combine(target_date, time.min), tz)
        end_dt = timezone.make_aware(datetime.combine(target_date, time.max), tz)

        ticks_qs = LiveTick.objects.filter(
            timestamp__gte=start_dt,
            timestamp__lte=end_dt,
            last__isnull=False,
        )

        if not ticks_qs.exists():
            self.stdout.write(f"Nenhum tick encontrado para {target_date}.")
            return

        tickers = list(ticks_qs.values_list("ticker", flat=True).distinct())
        created = 0
        updated = 0

        for ticker in tickers:
            ticker_qs = ticks_qs.filter(ticker=ticker).order_by("timestamp")
            first_tick = ticker_qs.first()
            last_tick = ticker_qs.last()
            if not first_tick or not last_tick:
                continue

            agg = ticker_qs.aggregate(
                high=Max("last"),
                low=Min("last"),
            )

            _, was_created = DailyPrice.objects.update_or_create(
                ticker=ticker,
                date=target_date,
                defaults={
                    "open": first_tick.last,
                    "close": last_tick.last,
                    "high": agg["high"],
                    "low": agg["low"],
                },
            )
            if was_created:
                created += 1
            else:
                updated += 1

            self.stdout.write(
                f"{ticker} {target_date}: open={first_tick.last} high={agg['high']} low={agg['low']} close={last_tick.last}"
            )

        self.stdout.write(
            self.style.SUCCESS(
                f"Consolidacao finalizada para {target_date}. tickers={len(tickers)} criados={created} atualizados={updated}"
            )
        )
