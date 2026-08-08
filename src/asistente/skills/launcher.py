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


#: Prefijo con el que Windows identifica las apps de la Microsoft Store.
APPS_FOLDER = "shell:AppsFolder\\"


def launch(command: str) -> bool:
    """Lanza la aplicacion. False si no se pudo.

    `command` viene SIEMPRE de la allowlist de `config.yaml`, nunca del texto
    transcrito ni del LLM. Admite cuatro tipos de destino, porque en Windows no
    todo se abre igual:

        C:\\...\\app.exe              ejecutable normal
        C:\\...\\acceso directo.lnk   atajo del menu inicio
        shell:AppsFolder\\Pkg!App     app de la Microsoft Store
        steam://rungameid/440        juego de Steam (o cualquier URI)
    """
    if command.startswith(APPS_FOLDER):
        return _launch_store_app(command)
    if "://" in command:
        return _launch_uri(command)
    if command.lower().endswith(".lnk"):
        return _launch_shortcut(command)

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


def _launch_store_app(command: str) -> bool:
    """Apps de la Microsoft Store.

    No tienen un .exe que se pueda ejecutar: viven en un contenedor y solo el
    shell sabe arrancarlas. `explorer.exe shell:AppsFolder\\<id>` es la via
    soportada.
    """
    if sys.platform != "win32":
        log.warning("las apps de la Store solo se pueden abrir en Windows")
        return False
    try:
        subprocess.Popen(["explorer.exe", command])  # noqa: S603, S607
    except OSError as exc:
        log.warning("no se pudo abrir la app de la Store %r: %s", command, exc)
        return False
    return True


def _launch_uri(uri: str) -> bool:
    """URIs registradas: `steam://rungameid/440`, `spotify:track:...`, https://...

    Es asi como se lanza un juego de Steam: no se ejecuta su .exe directamente
    porque entonces el cliente no lo registra como sesion de juego (ni cuenta
    horas, ni funciona el overlay, ni los DRM que dependen de el).
    """
    if sys.platform == "win32":
        try:
            import os

            os.startfile(uri)  # noqa: S606
        except OSError as exc:
            log.warning("no se pudo abrir %r: %s", uri, exc)
            return False
        return True

    opener = "open" if sys.platform == "darwin" else "xdg-open"
    try:
        subprocess.Popen([opener, uri])  # noqa: S603
    except OSError as exc:
        log.warning("no se pudo abrir %r: %s", uri, exc)
        return False
    return True


def _launch_shortcut(path: str) -> bool:
    """Accesos directos del menu inicio.

    Se lanza el .lnk tal cual en vez de resolver a que apunta: el atajo lleva
    ademas el directorio de trabajo y los argumentos que la app necesita, y
    algunas (los lanzadores de juegos, sobre todo) no arrancan bien sin ellos.
    """
    target = Path(path)
    if not target.is_file():
        log.warning("el acceso directo no existe: %s", path)
        return False
    if sys.platform != "win32":
        log.warning("los accesos directos .lnk solo funcionan en Windows")
        return False
    try:
        import os

        os.startfile(str(target))  # noqa: S606
    except OSError as exc:
        log.warning("no se pudo abrir %s: %s", target, exc)
        return False
    return True
