"""Autodescubrimiento de aplicaciones instaladas.

Mantener la allowlist a mano no escala: cada app hay que buscarla, averiguar su
ruta y escribirla. Aqui se enumeran solas.

CUATRO FUENTES, PORQUE NINGUNA LAS VE TODAS
-------------------------------------------
    Get-StartApps    apps de la Store Y de escritorio que salen en Inicio.
                     Es la fuente mas completa, pero necesita PowerShell.
    Menu Inicio      los .lnk en disco. Respaldo si PowerShell falla, y coge
                     cosas que Get-StartApps a veces no lista.
    App Paths        el registro. Poco a poco, pero da rutas de .exe exactas.
    Steam            los juegos NO aparecen en ninguna de las anteriores como
                     algo lanzable: hay que leer los manifiestos de Steam.

El resultado se deduplica por nombre normalizado y se ordena de forma que gane
la fuente que da un lanzamiento mas fiable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from asistente.router.text import normalize

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DiscoveredApp:
    """Una aplicacion encontrada en el sistema."""

    #: Nombre tal como lo muestra Windows ("Spotify", "Counter-Strike 2").
    name: str
    #: Destino para `launcher.launch()`: ruta, .lnk, shell:AppsFolder o URI.
    command: str
    #: De donde salio, para poder priorizar y para explicarlo en el informe.
    source: str
    #: Proceso a matar al cerrarla. None si no se sabe.
    process: str | None = None
    #: Formas alternativas de nombrarla al hablar.
    aliases: tuple[str, ...] = ()
    #: Clave sugerida para config.yaml (kebab/snake sin acentos).
    slug: str = field(default="")

    def with_slug(self, slug: str) -> DiscoveredApp:
        return DiscoveredApp(
            name=self.name,
            command=self.command,
            source=self.source,
            process=self.process,
            aliases=self.aliases,
            slug=slug,
        )


#: Que fuente gana cuando dos describen la misma app. Un .exe resuelto se lanza
#: mas rapido y permite cerrar la app; un .lnk arrastra el directorio de trabajo
#: correcto; el AppsFolder es lo unico que funciona para las de la Store.
_SOURCE_PRIORITY = {
    "steam": 0,
    "app_paths": 1,
    "start_menu": 2,
    "start_apps": 3,
}


def discover_all(include_steam: bool = True) -> list[DiscoveredApp]:
    """Enumera todo lo instalado, deduplicado y listo para config.local.yaml."""
    from asistente.discovery.steam import discover_steam_games
    from asistente.discovery.windows import (
        discover_app_paths,
        discover_start_apps,
        discover_start_menu,
    )

    found: list[DiscoveredApp] = []
    for label, fn in (
        ("Steam", discover_steam_games) if include_steam else ("", None),  # type: ignore[misc]
        ("App Paths", discover_app_paths),
        ("menu Inicio", discover_start_menu),
        ("Get-StartApps", discover_start_apps),
    ):
        if fn is None:
            continue
        try:
            apps = fn()
        except Exception:
            log.warning("fallo el descubrimiento por %s", label, exc_info=True)
            continue
        log.debug("%s: %d aplicaciones", label, len(apps))
        found.extend(apps)

    return _dedupe(found)


def _dedupe(apps: list[DiscoveredApp]) -> list[DiscoveredApp]:
    """Una entrada por nombre normalizado, quedandose con la mejor fuente."""
    best: dict[str, DiscoveredApp] = {}
    for app in apps:
        key = normalize(app.name)
        if not key:
            continue
        current = best.get(key)
        if current is None or _SOURCE_PRIORITY.get(app.source, 9) < _SOURCE_PRIORITY.get(
            current.source, 9
        ):
            best[key] = app

    # Asignar slugs unicos al final, cuando ya se sabe quien sobrevive.
    result: list[DiscoveredApp] = []
    used: set[str] = set()
    for app in sorted(best.values(), key=lambda a: a.name.lower()):
        slug = make_slug(app.name, used)
        used.add(slug)
        result.append(app.with_slug(slug))
    return result


def make_slug(name: str, used: set[str]) -> str:
    """Clave corta y estable para config.yaml, sin colisiones."""
    base = normalize(name).replace(" ", "_")[:32].strip("_") or "app"
    if base not in used:
        return base
    for n in range(2, 100):
        candidate = f"{base}_{n}"
        if candidate not in used:
            return candidate
    return base
