"""
Configurações MT5 isoladas.
"""

from __future__ import annotations

import os

from longshort.settings_base import _env_bool, _env_hosts, _env_float


def load_mt5_settings() -> dict:
    return {
        "USE_MT5_LIVE": _env_bool("USE_MT5_LIVE", True),
        "MT5_API_KEY": os.environ.get("MT5_API_KEY"),
        # Permite csv no env, ex.: MT5_ALLOWED_IPS="127.0.0.1,192.168.0.50"
        "MT5_ALLOWED_IPS": _env_hosts("MT5_ALLOWED_IPS", "127.0.0.1,::1"),
        "MT5_DEFAULT_CAPITAL": _env_float("MT5_DEFAULT_CAPITAL", 50000.0),
        "MT5_HTTP_BASE": os.environ.get("MT5_HTTP_BASE", ""),
        # Intervalo minimo entre registros de LiveTick por ticker (ms) para evitar inflar DB
        "MT5_LIVETICK_MIN_INTERVAL_MS": int(os.environ.get("MT5_LIVETICK_MIN_INTERVAL_MS", "250")),
    }
