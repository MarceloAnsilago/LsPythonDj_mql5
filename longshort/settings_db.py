"""
Configuração de banco de dados isolada.
Mantém compatibilidade com SQLite e Postgres (Fly.io).
"""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse


def _sqlite_path(base_dir: Path, url: str | None, default_name: Path) -> Path:
    if not url or not url.startswith("sqlite"):
        return default_name

    parsed_sqlite = urlparse(url)
    raw_path = parsed_sqlite.path or ""
    if parsed_sqlite.netloc:
        raw_path = f"{parsed_sqlite.netloc}{raw_path}"
    candidate = Path(raw_path.lstrip("/")) if raw_path else default_name
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    return candidate


def build_databases(base_dir: Path) -> dict:
    database_url = os.environ.get("DATABASE_URL")
    default_sqlite_name = base_dir / "db.sqlite3"

    if database_url and database_url.startswith("postgres"):
        parsed = urlparse(database_url)
        return {
            "default": {
                "ENGINE": "django.db.backends.postgresql",
                "NAME": parsed.path.lstrip("/") or "postgres",
                "USER": parsed.username,
                "PASSWORD": parsed.password,
                "HOST": parsed.hostname,
                "PORT": parsed.port or "5432",
            }
        }

    return {
        "default": {
            "ENGINE": "django.db.backends.sqlite3",
            "NAME": _sqlite_path(base_dir, database_url, default_sqlite_name),
            "OPTIONS": {
                "timeout": 30,  # aumenta timeout para reduzir "database is locked"
            },
        }
    }
