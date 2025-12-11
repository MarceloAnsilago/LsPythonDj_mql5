from django.apps import AppConfig

class AcoesConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'acoes'
    verbose_name = 'Ações'

    def ready(self):
        import acoes.signals  # noqa
