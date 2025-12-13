from __future__ import annotations

from datetime import datetime, time, date

from django.core.management.base import BaseCommand, CommandError
from django.db import models
from django.db.models.functions import Coalesce
from django.utils import timezone

from acoes.models import Asset
from cotacoes.models import QuoteDaily
from mt5api.models import LiveTick


class Command(BaseCommand):
    help = "Consolida ticks intradiarios em OHLC diario."

    def add_arguments(self, parser) -> None:
        parser.add_argument(
            "--date",
            dest="date",
            help="Data alvo no formato YYYY-MM-DD. Padrao: hoje no fuso configurado.",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Nao grava no banco; apenas mostra o que seria criado/atualizado.",
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

        tz = timezone.get_default_timezone()
        start_dt = timezone.make_aware(datetime.combine(target_date, time.min), tz)
        end_dt = timezone.make_aware(datetime.combine(target_date, time.max), tz)

        ticks_qs = (
            LiveTick.objects.annotate(effective_dt=Coalesce("as_of", "timestamp"))
            .filter(
                models.Q(as_of__isnull=False, as_of__gte=start_dt, as_of__lte=end_dt)
                | models.Q(as_of__isnull=True, timestamp__gte=start_dt, timestamp__lte=end_dt),
                last__isnull=False,
            )
        )

        if not ticks_qs.exists():
            self.stdout.write(f"Nenhum tick encontrado para {target_date}.")
            return

        tickers = list(ticks_qs.values_list("ticker", flat=True).distinct())
        created = 0
        updated = 0
        dry_run = bool(options.get("dry_run"))

        for ticker in tickers:
            asset = Asset.objects.filter(ticker=ticker).first()
            if asset is None:
                continue
            ticker_qs = ticks_qs.filter(ticker=ticker).order_by("effective_dt", "timestamp")
            first_tick = ticker_qs.first()
            last_tick = ticker_qs.last()
            if not first_tick or not last_tick:
                continue

            would_create = not QuoteDaily.objects.filter(asset=asset, date=target_date).exists()
            action = "created" if would_create else "updated"

            if dry_run:
                msg = f"[DRY-RUN] {ticker} {target_date}: open={first_tick.last} close={last_tick.last} -> {action}"
                self.stdout.write(msg)
                if would_create:
                    created += 1
                else:
                    updated += 1
                continue

            # close-only consolidation into QuoteDaily (OHLC completo nao e gravado aqui)
            _, was_created = QuoteDaily.objects.update_or_create(
                asset=asset,
                date=target_date,
                defaults={
                    "close": last_tick.last,
                    "is_provisional": False,
                },
            )
            action = "created" if was_created else "updated"
            if was_created:
                created += 1
            else:
                updated += 1

            self.stdout.write(
                f"{ticker} {target_date}: open={first_tick.last} close={last_tick.last} ({action})"
            )

        self.stdout.write(
            self.style.SUCCESS(
                f"Consolidacao finalizada para {target_date}. tickers={len(tickers)} criados={created} atualizados={updated}"
            )
        )
