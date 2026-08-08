"""Descubrimiento de aplicaciones de Windows: Store y escritorio.

Tres fuentes complementarias. Ninguna las ve todas, de ahi que se usen las tres.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

from asistente.discovery import DiscoveredApp

log = logging.getLogger(__name__)

_APP_PATHS_KEY = r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths"

#: Entradas del menu Inicio que no son aplicaciones que uno quiera abrir por voz.
_SKIP_KEYWORDS = (
    "uninstall", "desinstalar", "readme", "leame", "help", "ayuda",
    "documentation", "manual", "license", "licencia", "website", "sitio web",
    "release notes", "changelog", "报告", "repair", "reparar",
)


def _is_useful(name: str) -> bool:
    lowered = name.lower()
    return not any(keyword in lowered for keyword in _SKIP_KEYWORDS)


def discover_start_apps() -> list[DiscoveredApp]:
    """`Get-StartApps`: la fuente mas completa.

    Devuelve todo lo que sale en el menu Inicio, tanto apps de la Microsoft
    Store como de escritorio, con el AppID que necesita `shell:AppsFolder`.

    Se distingue una de otra por la forma del AppID: las de la Store lo tienen
    como `PackageFamilyName!AppId` (con '!'), y las de escritorio traen la ruta
    de su acceso directo.
    """
    if sys.platform != "win32":
        return []

    try:
        result = subprocess.run(  # noqa: S603
            [
                "powershell", "-NoProfile", "-NonInteractive", "-Command",
                "Get-StartApps | ConvertTo-Json -Compress",
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        log.debug("Get-StartApps no disponible: %s", exc)
        return []

    if result.returncode != 0 or not result.stdout.strip():
        log.debug("Get-StartApps no devolvio nada (rc=%s)", result.returncode)
        return []

    try:
        entries = json.loads(result.stdout)
    except json.JSONDecodeError:
        log.debug("Get-StartApps devolvio un JSON invalido")
        return []

    if isinstance(entries, dict):  # una sola app: PowerShell no la envuelve
        entries = [entries]

    apps: list[DiscoveredApp] = []
    for entry in entries:
        name = str(entry.get("Name", "")).strip()
        app_id = str(entry.get("AppID", "")).strip()
        if not name or not app_id or not _is_useful(name):
            continue
        apps.append(
            DiscoveredApp(
                name=name,
                command=f"shell:AppsFolder\\{app_id}",
                source="start_apps",
                # Las de la Store corren dentro de un contenedor y no hay un
                # nombre de proceso fiable que matar.
                process=None if "!" in app_id else _process_from_appid(app_id),
            )
        )
    return apps


def _process_from_appid(app_id: str) -> str | None:
    """Para apps de escritorio, el AppID suele ser la ruta del .lnk o del .exe."""
    stem = Path(app_id).stem
    return stem or None


def discover_start_menu() -> list[DiscoveredApp]:
    """Los .lnk del menu Inicio, de usuario y de todos los usuarios.

    Respaldo por si PowerShell esta restringido, y ademas recoge accesos
    directos que Get-StartApps a veces no lista. Se guarda la ruta del .lnk en
    vez de resolver a que apunta: el atajo lleva tambien el directorio de
    trabajo y los argumentos, que algunos lanzadores necesitan.
    """
    if sys.platform != "win32":
        return []

    roots = [
        Path(os.path.expandvars(r"%APPDATA%\Microsoft\Windows\Start Menu\Programs")),
        Path(os.path.expandvars(r"%ProgramData%\Microsoft\Windows\Start Menu\Programs")),
    ]

    apps: list[DiscoveredApp] = []
    for root in roots:
        if not root.is_dir():
            continue
        try:
            shortcuts = list(root.rglob("*.lnk"))
        except OSError:
            continue
        for shortcut in shortcuts:
            name = shortcut.stem.strip()
            if not name or not _is_useful(name):
                continue
            apps.append(
                DiscoveredApp(
                    name=name,
                    command=str(shortcut),
                    source="start_menu",
                    process=name.replace(" ", ""),
                )
            )
    return apps


def discover_app_paths() -> list[DiscoveredApp]:
    """App Paths del registro: pocas entradas, pero con rutas de .exe exactas.

    Es la mejor fuente cuando aparece, porque da un ejecutable que se puede
    lanzar directamente y un nombre de proceso fiable para poder cerrarlo.
    """
    if sys.platform != "win32":
        return []

    import winreg

    apps: list[DiscoveredApp] = []
    for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
        try:
            root = winreg.OpenKey(hive, _APP_PATHS_KEY)
        except OSError:
            continue

        with root:
            for index in range(winreg.QueryInfoKey(root)[0]):
                try:
                    entry = winreg.EnumKey(root, index)
                    with winreg.OpenKey(root, entry) as key:
                        value, _ = winreg.QueryValueEx(key, "")
                except OSError:
                    continue

                executable = Path(str(value).strip('"'))
                if not executable.is_file():
                    continue

                name = executable.stem
                if not _is_useful(name):
                    continue
                apps.append(
                    DiscoveredApp(
                        name=name,
                        command=str(executable),
                        source="app_paths",
                        process=executable.stem,
                    )
                )
    return apps
