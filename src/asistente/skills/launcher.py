"""Resolucion y lanzamiento de aplicaciones en Windows.

EL PROBLEMA
-----------
`subprocess.Popen("spotify", shell=True)` falla con

    'spotify' is not recognized as an internal or external command

porque `cmd.exe` solo busca en el PATH, y casi ningun instalador de Windows
mete su ejecutable ahi. Lo que si hacen es registrarse en

    HKEY_*\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\App Paths\\<nombre>.exe

Esa clave del registro es la razon de que Win+R -> "spotify" funcione. Aqui se
consulta explicitamente, que ademas permite decir con precision si una app esta
instalada o no en vez de fallar al ejecutarla.

ORDEN DE RESOLUCION
-------------------
    1. Ruta absoluta o relativa que exista.
    2. PATH (`shutil.which`), para lo que si esta ahi: code, python...
    3. App Paths del registro, primero HKCU (instalaciones por usuario, que es
       como se instala Spotify) y luego HKLM.
    4. Ultimo recurso: `start ""`, que resuelve los alias de las apps de la
       Microsoft Store.

Devolver la ruta resuelta en vez de tirar de shell tiene otra ventaja: se lanza
con una lista de argumentos y sin shell, con lo que desaparece cualquier
posibilidad de inyeccion aunque el comando viniera de un sitio menos fiable que
la allowlist.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from pathlib import Path

log = logging.getLogger(__name__)

_APP_PATHS_KEY = r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths"


def resolve_executable(command: str) -> Path | None:
    """Localiza el ejecutable, o None si no se encuentra por ninguna via."""
    candidate = Path(command)
    if candidate.suffix and candidate.exists():
        return candidate

    if (found := shutil.which(command)) is not None:
        return Path(found)

    if sys.platform == "win32":
        return _lookup_app_paths(command)
    return None


def _lookup_app_paths(command: str) -> Path | None:
    """Consulta App Paths del registro de Windows."""
    import winreg

    names = (command,) if command.lower().endswith(".exe") else (f"{command}.exe", command)
    # HKCU primero: Spotify, Discord y compania se instalan por usuario, y si
    # existieran ambas entradas la del usuario es la que Windows prioriza.
    for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
        for name in names:
            try:
                with winreg.OpenKey(hive, rf"{_APP_PATHS_KEY}\{name}") as key:
                    value, _ = winreg.QueryValueEx(key, "")
            except OSError:
                continue
            path = Path(str(value).strip('"'))
            if path.exists():
                return path
    return None


def launch(command: str) -> bool:
    """Lanza la aplicacion. False si no se pudo.

    `command` viene SIEMPRE de la allowlist de `config.yaml`, nunca del texto
    transcrito ni del LLM.
    """
    if (executable := resolve_executable(command)) is not None:
        try:
            subprocess.Popen(  # noqa: S603
                [str(executable)],
                cwd=str(executable.parent),
                start_new_session=True,
            )
        except OSError as exc:
            log.warning("no se pudo lanzar %s: %s", executable, exc)
            return False
        return True

    if sys.platform != "win32":
        log.warning("no se encontro el ejecutable %r", command)
        return False

    # Ultimo recurso: `start` resuelve alias de apps de la Store que no tienen
    # ejecutable localizable. El comando sale de la allowlist, no de la voz.
    log.debug("%r no se resolvio; probando con `start`", command)
    try:
        result = subprocess.run(  # noqa: S602
            f'start "" "{command}"',
            shell=True,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        log.warning("fallo `start %s`: %s", command, exc)
        return False

    if result.returncode != 0:
        log.warning(
            "no se encontro %r. Comprueba la ruta en config.yaml; "
            "diagnostico: python scripts/diagnose_apps.py",
            command,
        )
        return False
    return True
