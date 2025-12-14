from __future__ import annotations

from datetime import datetime, date

from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone


class Command(BaseCommand):
    help = "Informativo: a consolidacao MT5 em QuoteDaily esta desativada."

    def add_arguments(self, parser) -> None:
        parser.add_argument(
            "--date",
            dest="date",
            help="Data alvo no formato YYYY-MM-DD (sem uso na logica atual).",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Sem efeito; a consolidacao MT5 esta desativada.",
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
            raise CommandError("Somente candles D1 fechados (date < hoje) podiam ser consolidados.")

        self.stdout.write(
            self.style.WARNING(
                "A consolidacao MT5 para QuoteDaily foi desativada; o historico D1 agora vem via Yahoo."
            )
        )
