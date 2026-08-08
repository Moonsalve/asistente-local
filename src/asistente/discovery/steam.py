"""Descubrimiento de juegos de Steam.

Los juegos no aparecen en App Paths ni en Get-StartApps como algo lanzable, asi
que hay que leer los datos del propio Steam. Son dos ficheros en su formato VDF:

    steamapps/libraryfolders.vdf   donde estan las bibliotecas (varios discos)
    steamapps/appmanifest_<id>.acf uno por juego INSTALADO, con su id y nombre

Se lee el .acf y no la biblioteca de la cuenta a proposito: la cuenta lista
tambien lo que tienes comprado pero no instalado, y ofrecer abrir algo que no
esta en el disco solo produce fallos.

POR QUE steam://rungameid Y NO EL .exe DEL JUEGO
------------------------------------------------
Ejecutar el .exe directamente se salta al cliente: no cuenta horas, no funciona
el overlay, no se sincronizan las partidas guardadas en la nube, y los juegos
con DRM de Steam directamente se niegan a arrancar. La URI es la forma correcta
y ademas arranca Steam si no estaba abierto.

El parseo del VDF se hace con expresiones regulares en vez de con un parser
completo: solo hacen falta dos campos por fichero, y meter una dependencia para
esto no compensa.
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

from asistente.discovery import DiscoveredApp

log = logging.getLogger(__name__)

_PATH_RE = re.compile(r'"path"\s+"([^"]+)"', re.IGNORECASE)
_APPID_RE = re.compile(r'"appid"\s+"(\d+)"', re.IGNORECASE)
_NAME_RE = re.compile(r'"name"\s+"([^"]+)"', re.IGNORECASE)

#: Entradas del runtime de Steam que aparecen como si fueran juegos.
_NOT_GAMES = frozenset({
    "steamworks common redistributables",
    "steam linux runtime",
    "proton experimental",
})


def steam_root() -> Path | None:
    """Localiza la instalacion de Steam."""
    if sys.platform == "win32":
        import winreg

        for hive, key in (
            (winreg.HKEY_CURRENT_USER, r"Software\Valve\Steam"),
            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Valve\Steam"),
        ):
            try:
                with winreg.OpenKey(hive, key) as handle:
                    value, _ = winreg.QueryValueEx(
                        handle, "SteamPath" if hive == winreg.HKEY_CURRENT_USER else "InstallPath"
                    )
            except OSError:
                continue
            path = Path(str(value))
            if path.is_dir():
                return path

    for candidate in (
        Path.home() / ".steam/steam",
        Path.home() / "Library/Application Support/Steam",
        Path(r"C:\Program Files (x86)\Steam"),
    ):
        if candidate.is_dir():
            return candidate
    return None


def library_folders(root: Path) -> list[Path]:
    """Todas las bibliotecas, que pueden estar repartidas por varios discos."""
    libraries = [root / "steamapps"]

    vdf = root / "steamapps" / "libraryfolders.vdf"
    if vdf.is_file():
        try:
            content = vdf.read_text(encoding="utf-8", errors="replace")
        except OSError:
            content = ""
        for raw in _PATH_RE.findall(content):
            # El VDF escapa las barras invertidas.
            steamapps = Path(raw.replace("\\\\", "\\")) / "steamapps"
            if steamapps.is_dir() and steamapps not in libraries:
                libraries.append(steamapps)

    return [lib for lib in libraries if lib.is_dir()]


def discover_steam_games() -> list[DiscoveredApp]:
    """Juegos INSTALADOS, listos para lanzar por URI."""
    root = steam_root()
    if root is None:
        log.debug("Steam no esta instalado")
        return []

    games: list[DiscoveredApp] = []
    for library in library_folders(root):
        for manifest in sorted(library.glob("appmanifest_*.acf")):
            try:
                content = manifest.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue

            appid_match = _APPID_RE.search(content)
            name_match = _NAME_RE.search(content)
            if appid_match is None or name_match is None:
                continue

            name = name_match.group(1).strip()
            if name.lower() in _NOT_GAMES:
                continue

            games.append(
                DiscoveredApp(
                    name=name,
                    command=f"steam://rungameid/{appid_match.group(1)}",
                    source="steam",
                    aliases=_game_aliases(name),
                )
            )

    log.debug("Steam: %d juegos instalados", len(games))
    return games


def _game_aliases(name: str) -> tuple[str, ...]:
    """Formas cortas con las que se suele nombrar un juego al hablar.

    Los titulos de juegos son largos ("Counter-Strike 2", "Grand Theft Auto V")
    y nadie los dice enteros. Se generan las siglas y una version sin subtitulo
    para que el matching difuso tenga a que agarrarse.
    """
    aliases: set[str] = set()

    # Sin lo que va tras ':' o '-': "Half-Life 2: Episode One" -> "Half-Life 2"
    for separator in (":", " - "):
        if separator in name:
            aliases.add(name.split(separator)[0].strip())

    # Siglas si son tres palabras o mas: "Counter Strike Global Offensive" -> "csgo"
    words = [w for w in re.split(r"[\s\-:]+", name) if w]
    if len(words) >= 3:
        initials = "".join(w[0] for w in words if w[0].isalpha()).lower()
        if 2 <= len(initials) <= 6:
            aliases.add(initials)

    aliases.discard(name)
    return tuple(sorted(aliases))
