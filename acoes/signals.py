from django.db.models.signals import post_save
from django.dispatch import receiver

from acoes.models import Asset
from longshort.services.quotes import ensure_min_history_for_asset


@receiver(post_save, sender=Asset)
def seed_history_for_new_asset(sender, instance: Asset, created: bool, **kwargs):
    """
    Quando um novo ativo é criado, garante pelo menos 200 barras diárias via MT5.
    """
    if not created:
        return

    if hasattr(instance, "use_mt5") and not getattr(instance, "use_mt5", False):
        return

    try:
        inserted = ensure_min_history_for_asset(instance)
        if inserted > 0:
            print(f"[seed_history_for_new_asset] {instance.ticker}: {inserted} linhas inseridas.")
    except Exception as exc:
        print(f"[seed_history_for_new_asset] erro ao semear historico para {getattr(instance, 'ticker', '')}: {exc}")
