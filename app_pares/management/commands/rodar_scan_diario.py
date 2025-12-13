from __future__ import annotations

from datetime import datetime, date

from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from app_pares.services import rodar_scan_diario


class Command(BaseCommand):
    help = "Atualiza cotações e roda o scan diario de pares."

    def add_arguments(self, parser) -> None:
        parser.add_argument(
            "--data",
            dest="data",
            help="Data no formato YYYY-MM-DD. Padrão: dia atual.",
        )
        parser.add_argument(
            "--skip-quotes",
            action="store_true",
            help="Nao atualiza cotacoes antes do scan.",
        )
        parser.add_argument(
            "--skip-scan",
            action="store_true",
            help="Nao roda o scan de pares (apenas cotacoes).",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Nao grava PairScanResult; apenas executa calculos.",
        )
        parser.add_argument(
            "--enable-history",
            action="store_true",
            help="(Deprecated) Tenta sincronizar PriceHistory. Desativado por padrao.",
        )

    def handle(self, *args, **options) -> None:
        raw_date = options.get("data")
        target_date: date
        if raw_date:
            try:
                target_date = datetime.strptime(raw_date, "%Y-%m-%d").date()
            except ValueError as exc:
                raise CommandError(f"Formato inválido de data: {exc}") from exc
        else:
            target_date = timezone.localdate()

        self.stdout.write(f"Rodando scan diario para {target_date}")
        summary = rodar_scan_diario(
            target_date,
            skip_quotes=options.get("skip_quotes", False),
            skip_scan=options.get("skip_scan", False),
            dry_run=options.get("dry_run", False),
            enable_history=options.get("enable_history", False),
        )

        quotes = summary.get("quotes") or {}
        pairs = summary.get("pairs") or {"pairs_processed": 0, "created": 0, "updated": 0, "errors": []}

        if quotes:
            self.stdout.write(f"cotações: ativos={quotes.get('assets')} atualizados={quotes.get('updated_assets')}")
        else:
            self.stdout.write("cotações: skipped")

        history = summary.get("history") or {}
        if history:
            self.stdout.write(
                f"price history: registros={history.get('records')} criados={history.get('created')} atualizados={history.get('updated')}"
            )
        else:
            self.stdout.write("price history: skipped")

        self.stdout.write(
            f"pares: processados={pairs['pairs_processed']} criados={pairs['created']} atualizados={pairs['updated']} erros={len(pairs['errors'])}"
        )
        if pairs["errors"]:
            for error in pairs["errors"]:
                self.stdout.write(f"- {error}")
