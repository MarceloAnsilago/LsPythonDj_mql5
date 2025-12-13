"""
Configurações base e utilitários compartilhados.
Responsável apenas por helpers, BASE_DIR e listas básicas.
"""

from __future__ import annotations

import os
from pathlib import Path

# Caminho base do projeto
BASE_DIR = Path(__file__).resolve().parent.parent


# Helpers de ambiente
def _env_bool(key: str, default: bool = False) -> bool:
    value = os.environ.get(key)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_hosts(key: str, default: str) -> list[str]:
    raw = os.environ.get(key, default)
    return [host.strip() for host in raw.split(",") if host.strip()]


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, default))
    except (TypeError, ValueError):
        return float(default)


# Listas base de apps (com comentários para organização)
DJANGO_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
]

PROJECT_APPS = [
    "core",
    "acoes",
    "cotacoes",
    "operacoes",
    "app_pares",
    "pairs",
    "accounts",
    "mt5api",
]

THIRD_PARTY_APPS = [
    "django_htmx",
    "widget_tweaks",
]
