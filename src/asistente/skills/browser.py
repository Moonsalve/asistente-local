"""Apertura de URLs en un navegador concreto.

`webbrowser.open()` usa el navegador por defecto de Windows, que no siempre es
el que uno quiere. Aqui se abre en el navegador configurado (Brave por defecto)
y solo se recurre al del sistema si ese no esta instalado.

Se resuelve una sola vez al arrancar y se recuerda: buscar en el registro en
cada busqueda seria trabajo repetido para un dato que no cambia.
"""

from __future__ import annotations

import logging
import subprocess
import webbrowser
from functools import lru_cache
from pathlib import Path

from asistente.skills.launcher import resolve_executable

log = logging.getLogger(__name__)

#: Nombres de ejecutable por navegador. Se prueban en orden hasta que uno
#: resuelva: Brave se instala con nombres distintos segun la version.
_BROWSER_EXECUTABLES: dict[str, tuple[str, ...]] = {
    "brave": ("brave", "brave-browser", "BraveBrowser"),
    "chrome": ("chrome", "googlechrome"),
    "firefox": ("firefox",),
    "edge": ("msedge",),
    "opera": ("opera",),
    "vivaldi": ("vivaldi",),
}


@lru_cache(maxsize=8)
def find_browser(name: str) -> Path | None:
    """Localiza el ejecutable del navegador, o None si no esta instalado."""
    candidates = _BROWSER_EXECUTABLES.get(name.lower(), (name,))
    for candidate in candidates:
        if (found := resolve_executable(candidate)) is not None:
            return found
    return None


class Browser:
    """Abre URLs en el navegador elegido."""

    def __init__(self, preferred: str = "brave") -> None:
        self._preferred = preferred
        self._executable = find_browser(preferred) if preferred else None

        if preferred and self._executable is None:
            log.warning(
                "no se encontro el navegador '%s'; se usara el del sistema. "
                "Si lo tienes instalado, pon su ruta en config.local.yaml: "
                "web: {browser_path: 'C:\\\\ruta\\\\a\\\\brave.exe'}",
                preferred,
            )
        elif self._executable is not None:
            log.info("navegador: %s (%s)", preferred, self._executable)

    @classmethod
    def from_path(cls, preferred: str, explicit_path: Path | None) -> Browser:
        """Permite fijar la ruta a mano cuando la deteccion automatica falla."""
        browser = cls.__new__(cls)
        browser._preferred = preferred
        if explicit_path is not None and explicit_path.is_file():
            browser._executable = explicit_path
            log.info("navegador: %s (ruta explicita)", explicit_path)
        else:
            if explicit_path is not None:
                log.warning("la ruta del navegador no existe: %s", explicit_path)
            browser._executable = find_browser(preferred) if preferred else None
            if browser._executable is not None:
                log.info("navegador: %s (%s)", preferred, browser._executable)
        return browser

    def open(self, url: str) -> bool:
        if self._executable is not None:
            try:
                subprocess.Popen(  # noqa: S603
                    [str(self._executable), url],
                    start_new_session=True,
                )
            except OSError as exc:
                log.warning("fallo al abrir en %s: %s", self._executable.name, exc)
            else:
                return True

        # Sin el navegador preferido, mejor abrir en el del sistema que no
        # hacer nada: el usuario pidio ver algo.
        return webbrowser.open(url)
