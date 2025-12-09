from django.contrib import admin
from django.urls import path, include
from django.http import JsonResponse


def healthcheck(request):
    return JsonResponse({"status": "ok"})


urlpatterns = [
    path("admin/", admin.site.urls),

    path("acoes/", include("acoes.urls")),
    path("cotacoes/", include("cotacoes.urls")),
    path("api/mt5/", include(("mt5api.urls", "mt5api"), namespace="mt5api")),

    path("pares/", include(("pairs.urls", "pairs"), namespace="pairs")),

    path("accounts/", include("accounts.urls")),

    path("health/", healthcheck, name="healthcheck"),

    path("", include("core.urls")),
]
