"""Descubrimiento automatico al arrancar, con cache.

Enumerar tarda unos segundos: `Get-StartApps` lanza PowerShell y los .lnk del
menu Inicio se recorren en disco. Pagar eso en cada arranque no tiene sentido
porque las aplicaciones instaladas no cambian de un dia para otro, asi que el
resultado se cachea y solo se vuelve a enumerar cuando caduca.

PRECEDENCIA: lo declarado a mano en `apps:` SIEMPRE gana sobre lo descubierto.
Si has puesto la ruta exacta de Spotify porque la deteccion fallaba, el
descubrimiento no te la pisa.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

from asistente.config import Config
from asistente.discovery import DiscoveredApp
from asistente.router.text import normalize

log = logging.getLogger(__name__)


def _cache_path() -> Path:
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~/.cache")
    directory = Path(base) / "asistente-local"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / "apps-cache.json"


def _load_cache(max_age_hours: float) -> list[DiscoveredApp] | None:
    path = _cache_path()
    if not path.is_file() or max_age_hours <= 0:
        return None

    age_hours = (time.time() - path.stat().st_mtime) / 3600
    if age_hours > max_age_hours:
        log.debug("cache de aplicaciones caducado (%.1f h)", age_hours)
        return None

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        apps = [DiscoveredApp(**entry) for entry in raw]
    except Exception:
        log.debug("cache de aplicaciones ilegible; se vuelve a enumerar", exc_info=True)
        return None

    log.debug("cache de aplicaciones: %d entradas (%.1f h)", len(apps), age_hours)
    return apps


def _save_cache(apps: list[DiscoveredApp]) -> None:
    try:
        _cache_path().write_text(
            json.dumps(
                [
                    {
                        "name": a.name,
                        "command": a.command,
                        "source": a.source,
                        "process": a.process,
                        "aliases": list(a.aliases),
                        "slug": a.slug,
                    }
                    for a in apps
                ],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    except OSError:
        log.debug("no se pudo escribir el cache de aplicaciones", exc_info=True)


def augment_config(config: Config) -> Config:
    """Devuelve una copia de la config con las apps descubiertas anadidas.

    No modifica la original. Las declaradas a mano tienen prioridad, tanto por
    clave como por alias: si ya reconoces "spotify" a mano, lo descubierto con
    ese mismo nombre se descarta en vez de competir en el matching difuso.
    """
    if not config.discovery.enabled:
        return config

    from asistente.config import AppSpec

    apps = _load_cache(config.discovery.cache_hours)
    if apps is None:
        started = time.perf_counter()
        from asistente.discovery import discover_all

        apps = discover_all(include_steam=config.discovery.steam)
        log.info(
            "descubiertas %d aplicaciones en %.1f s (se cachea %.0f h)",
            len(apps),
            time.perf_counter() - started,
            config.discovery.cache_hours,
        )
        _save_cache(apps)

    if not apps:
        return config

    # Todo lo que ya reconoce la config a mano: claves y alias, normalizados.
    reservados: set[str] = set()
    for key, spec in config.apps.items():
        reservados.add(normalize(key))
        reservados.update(normalize(alias) for alias in spec.aliases)

    nuevas: dict[str, AppSpec] = {}
    for app in apps:
        if normalize(app.name) in reservados or app.slug in config.apps:
            continue
        alias = tuple(dict.fromkeys([app.name, *app.aliases]))
        nuevas[app.slug] = AppSpec(
            command=app.command,
            process=app.process,
            aliases=alias,
        )

    if not nuevas:
        return config

    log.info(
        "allowlist: %d apps configuradas + %d descubiertas",
        len(config.apps),
        len(nuevas),
    )
    # Las manuales al final para que ganen ante cualquier colision de clave.
    return config.model_copy(update={"apps": {**nuevas, **config.apps}})
