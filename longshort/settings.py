"""
Configurações principais do projeto longshort.
- Carrega .env apenas em DEBUG ou se DJANGO_LOAD_DOTENV=1.
- Monta blocos (base, DB, MT5) a partir de módulos auxiliares.
- Mantém compatibilidade com Fly.io e Django 5.2.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from longshort.settings_base import (
    BASE_DIR,
    _env_bool,
    _env_hosts,
    _env_float,
    DJANGO_APPS,
    PROJECT_APPS,
    THIRD_PARTY_APPS,
)
from longshort.settings_db import build_databases
from longshort.settings_mt5 import load_mt5_settings


# ------------------------------------------------------------------------------
# Dotenv (condicional)
# ------------------------------------------------------------------------------
def _should_load_dotenv() -> bool:
    """Carrega .env se DEBUG apontar para True ou DJANGO_LOAD_DOTENV=1."""
    if os.environ.get("DJANGO_LOAD_DOTENV") == "1":
        return True
    raw_debug = os.environ.get("DJANGO_DEBUG")
    if raw_debug is None:
        # default de DEBUG é True quando não setado
        return True
    return raw_debug.strip().lower() in {"1", "true", "yes", "on"}


if _should_load_dotenv():
    load_dotenv()


# ------------------------------------------------------------------------------
# Configurações básicas
# ------------------------------------------------------------------------------
SECRET_KEY = os.environ.get(
    "DJANGO_SECRET_KEY",
    "django-insecure-gntby)lk+iv$k9jk@re@1dqsn4gm898auvmc7x6zqvt&*!_29$",
)

# DEBUG usa helper após possível dotenv
DEBUG = _env_bool("DJANGO_DEBUG", True)

if not DEBUG and not os.environ.get("DJANGO_SECRET_KEY"):
    raise RuntimeError("DJANGO_SECRET_KEY deve estar definido em produção (DEBUG=False).")

ALLOWED_HOSTS = _env_hosts(
    "DJANGO_ALLOWED_HOSTS",
    # Default abre localhost, loopback e bind 0.0.0.0 para dev/LAN (quando DEBUG=True)
    "localhost,127.0.0.1,0.0.0.0,.fly.dev",
)

CSRF_TRUSTED_ORIGINS = _env_hosts(
    "DJANGO_CSRF_TRUSTED_ORIGINS",
    # Inclui http(s) para uso local/LAN; adicione http://<ip_da_maquina> via env para MT5
    "http://localhost,http://127.0.0.1,http://0.0.0.0,https://lspythondj.fly.dev,https://*.fly.dev",
)


# ------------------------------------------------------------------------------
# Apps
# ------------------------------------------------------------------------------
INSTALLED_APPS = [
    # Django
    *DJANGO_APPS,
    # Terceiros
    *THIRD_PARTY_APPS,
    # Projeto
    *PROJECT_APPS,
]


# ------------------------------------------------------------------------------
# Middleware
# ------------------------------------------------------------------------------
MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
    "django_htmx.middleware.HtmxMiddleware",
]


# ------------------------------------------------------------------------------
# URLs / Templates / WSGI
# ------------------------------------------------------------------------------
ROOT_URLCONF = "longshort.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [
            BASE_DIR / "longshort" / "templates",
            BASE_DIR / "templates",
        ],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "longshort.wsgi.application"


# ------------------------------------------------------------------------------
# Database
# ------------------------------------------------------------------------------
DATABASES = build_databases(BASE_DIR)


# ------------------------------------------------------------------------------
# Senhas / Auth
# ------------------------------------------------------------------------------
AUTH_PASSWORD_VALIDATORS = [
    {
        "NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.MinimumLengthValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.CommonPasswordValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.NumericPasswordValidator",
    },
]


# ------------------------------------------------------------------------------
# Internacionalização
# ------------------------------------------------------------------------------
LANGUAGE_CODE = "pt-br"
TIME_ZONE = os.environ.get("DJANGO_TIME_ZONE", "America/Sao_Paulo")
USE_I18N = True
USE_TZ = True


# ------------------------------------------------------------------------------
# Static files
# ------------------------------------------------------------------------------
STATIC_URL = "static/"
STATICFILES_DIRS = [BASE_DIR / "static"]
STATIC_ROOT = BASE_DIR / "staticfiles"
STATICFILES_STORAGE = "whitenoise.storage.CompressedManifestStaticFilesStorage"


# ------------------------------------------------------------------------------
# Defaults / Redirecionamentos
# ------------------------------------------------------------------------------
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
LOGIN_REDIRECT_URL = "acoes:lista"
LOGOUT_REDIRECT_URL = "core:home"


# ------------------------------------------------------------------------------
# MT5 (via módulo dedicado)
# ------------------------------------------------------------------------------
_mt5 = load_mt5_settings()
USE_MT5_LIVE = _mt5["USE_MT5_LIVE"]
MT5_API_KEY = _mt5["MT5_API_KEY"]
MT5_ALLOWED_IPS = _mt5["MT5_ALLOWED_IPS"]
MT5_DEFAULT_CAPITAL = _mt5["MT5_DEFAULT_CAPITAL"]
MT5_HTTP_BASE = _mt5["MT5_HTTP_BASE"]
