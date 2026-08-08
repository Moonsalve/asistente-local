"""Tests del autodescubrimiento.

Las fuentes concretas dependen de Windows y no se pueden probar aqui, pero la
logica que las combina si: deduplicacion, prioridad de fuentes, slugs unicos y
el parseo de los manifiestos de Steam (que son ficheros de texto).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from asistente.discovery import DiscoveredApp, _dedupe, make_slug
from asistente.discovery.steam import _game_aliases, discover_steam_games, library_folders


def _app(name: str, source: str, command: str = "x") -> DiscoveredApp:
    return DiscoveredApp(name=name, command=command, source=source)


def test_dedupe_keeps_the_most_reliable_source() -> None:
    """Un .exe resuelto se lanza mejor que un atajo del menu Inicio."""
    apps = _dedupe([
        _app("Spotify", "start_apps", "shell:AppsFolder\\X!App"),
        _app("Spotify", "app_paths", r"C:\Spotify.exe"),
        _app("Spotify", "start_menu", r"C:\Spotify.lnk"),
    ])
    assert len(apps) == 1
    assert apps[0].source == "app_paths"


def test_dedupe_ignores_case_and_accents() -> None:
    apps = _dedupe([_app("Café", "start_menu"), _app("CAFE", "start_apps")])
    assert len(apps) == 1


def test_steam_wins_over_everything() -> None:
    """Un juego hay que lanzarlo por URI: su .exe se salta al cliente."""
    apps = _dedupe([
        _app("Counter-Strike 2", "start_menu", r"C:\cs2.lnk"),
        _app("Counter-Strike 2", "steam", "steam://rungameid/730"),
    ])
    assert apps[0].command == "steam://rungameid/730"


def test_slugs_are_unique() -> None:
    apps = _dedupe([_app("Mi App", "steam"), _app("mi-app", "start_menu")])
    slugs = [a.slug for a in apps]
    assert len(slugs) == len(set(slugs))


def test_make_slug_avoids_collisions() -> None:
    assert make_slug("Spotify", set()) == "spotify"
    assert make_slug("Spotify", {"spotify"}) == "spotify_2"


@pytest.mark.parametrize(
    ("title", "expected_alias"),
    [
        ("Half-Life 2: Episode One", "Half-Life 2"),
        ("Counter Strike Global Offensive", "csgo"),
        ("Portal", None),  # una sola palabra: no hay alias que inventar
    ],
)
def test_game_aliases(title: str, expected_alias: str | None) -> None:
    aliases = _game_aliases(title)
    if expected_alias is None:
        assert aliases == ()
    else:
        assert expected_alias in aliases


def test_steam_manifest_parsing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Se leen los manifiestos en disco (juegos INSTALADOS), no la biblioteca
    de la cuenta: ofrecer abrir algo no instalado solo produce fallos."""
    steamapps = tmp_path / "steamapps"
    steamapps.mkdir()
    (steamapps / "appmanifest_730.acf").write_text(
        '"AppState"\n{\n\t"appid"\t\t"730"\n\t"name"\t\t"Counter-Strike 2"\n}\n'
    )
    (steamapps / "appmanifest_228980.acf").write_text(
        '"AppState"\n{\n\t"appid"\t\t"228980"\n'
        '\t"name"\t\t"Steamworks Common Redistributables"\n}\n'
    )

    monkeypatch.setattr("asistente.discovery.steam.steam_root", lambda: tmp_path)
    games = discover_steam_games()

    nombres = {g.name for g in games}
    assert "Counter-Strike 2" in nombres
    assert "Steamworks Common Redistributables" not in nombres, "el runtime no es un juego"
    assert games[0].command == "steam://rungameid/730"


def test_library_folders_reads_extra_disks(tmp_path: Path) -> None:
    """Steam reparte los juegos por varios discos; hay que mirarlos todos."""
    (tmp_path / "steamapps").mkdir()
    otro = tmp_path / "D_Games"
    (otro / "steamapps").mkdir(parents=True)
    (tmp_path / "steamapps" / "libraryfolders.vdf").write_text(
        '"libraryfolders"\n{\n\t"0"\n\t{\n\t\t"path"\t\t"%s"\n\t}\n}\n'
        % str(otro).replace("\\", "\\\\")
    )
    assert (otro / "steamapps") in library_folders(tmp_path)
