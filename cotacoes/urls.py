# cotacoes/urls.py
from django.urls import path
from . import views
from .views import (
    QuotesHomeView, update_quotes, quotes_pivot,
    clear_logs, quotes_progress, update_quotes_ajax,
    update_live_quotes_view, save_today_quotes_view, download_history_view,
    faltantes_home, faltantes_scan,
    faltantes_detail, faltantes_fetch_one, faltantes_insert_one,
)

app_name = "cotacoes"

urlpatterns = [
    path("", QuotesHomeView.as_view(), name="home"),
    path("atualizar/", update_quotes, name="update"),
    path("atualizar-ao-vivo/", update_live_quotes_view, name="update_live"),
    path("atualizar-hoje/", save_today_quotes_view, name="save_today"),
    path("baixar-historico/", download_history_view, name="download_history"),
    path("ajax/atualizar/", update_quotes_ajax, name="update_ajax"),
    path("progresso/", quotes_progress, name="progress"),
    path("pivot/", quotes_pivot, name="pivot"),
    path("logs/limpar/", clear_logs, name="logs_clear"),

    # Faltantes - lista e scanner
    path("faltantes/", faltantes_home, name="faltantes_home"),
    path("faltantes/scan/", faltantes_scan, name="faltantes_scan"),

    # Faltantes - detalhe por ticker
    path("faltantes/<str:ticker>/", faltantes_detail, name="faltantes_detail"),
    path("faltantes/<str:ticker>/fetch/<slug:dt>/", faltantes_fetch_one, name="faltantes_fetch_one"),
    path("faltantes/<str:ticker>/insert/", faltantes_insert_one, name="faltantes_insert_one"),
]
