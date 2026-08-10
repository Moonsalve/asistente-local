"""Tests del autodescubrimiento.

Las fuentes concretas dependen de Windows y no se pueden probar aqui, pero la
logica que las combina si: deduplicacion, prioridad de fuentes, slugs unicos y
el parseo de los manifiestos de Steam (que son ficheros de texto).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from asistente.config import AppSpec
from asistente.discovery import DiscoveredApp, _dedupe, make_slug
from asistente.discovery.auto import repair_manual_entries
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


# --------------------------------------------------------- reparar la config
#
# La precedencia "lo manual gana" es correcta salvo en un caso: cuando lo manual
# NO FUNCIONA. Es lo que pasaba con `spotify: {command: spotify}`, que depende de
# una clave del registro que no siempre existe, mientras el descubrimiento tenia
# el AppID bueno y lo tiraba por ser "menos prioritario".

_NUNCA = lambda command: False  # noqa: E731 - nada se puede lanzar
_SIEMPRE = lambda command: True  # noqa: E731 - todo se puede lanzar


def test_a_broken_manual_command_is_replaced() -> None:
    manual = {"spotify": AppSpec(command="spotify", process="Spotify", aliases=("spoti",))}
    descubierto = [DiscoveredApp(name="Spotify", command="shell:AppsFolder\\X!App",
                                 source="start_apps", slug="spotify")]

    arregladas = repair_manual_entries(manual, descubierto, puede_lanzar=_NUNCA)
    assert arregladas["spotify"].command == "shell:AppsFolder\\X!App"


def test_a_working_manual_command_is_left_alone() -> None:
    """Si escribiste la ruta exacta porque la deteccion fallaba, nadie te la toca."""
    manual = {"spotify": AppSpec(command=r"C:\mi\ruta\Spotify.exe")}
    descubierto = [DiscoveredApp(name="Spotify", command="otra-cosa", source="start_apps",
                                 slug="spotify")]
    assert repair_manual_entries(manual, descubierto, puede_lanzar=_SIEMPRE) == {}


def test_repairing_keeps_your_aliases_and_your_process() -> None:
    """Solo el comando esta roto. Tus alias y tu nombre de proceso son mejores
    que los deducidos: el descubrimiento no sabe que a Chrome le llamas "el
    navegador" ni deduce proceso alguno de una app de la Store."""
    manual = {"chrome": AppSpec(command="chrome", process="chrome",
                                aliases=("el navegador", "navegador"))}
    descubierto = [DiscoveredApp(name="Google Chrome", command=r"C:\chrome.exe",
                                 source="app_paths", process="GoogleChrome", slug="google_chrome",
                                 aliases=())]

    arregladas = repair_manual_entries(manual, descubierto, puede_lanzar=_NUNCA)
    assert arregladas == {}, "no coincide ni la clave ni ningun alias: no hay a que agarrarse"


def test_an_alias_is_enough_to_find_the_replacement() -> None:
    manual = {"vscode": AppSpec(command="code", aliases=("visual studio code",))}
    descubierto = [DiscoveredApp(name="Visual Studio Code", command=r"C:\Code.exe",
                                 source="app_paths", process="Code", slug="visual_studio_code")]

    arregladas = repair_manual_entries(manual, descubierto, puede_lanzar=_NUNCA)
    assert arregladas["vscode"].command == r"C:\Code.exe"
    assert arregladas["vscode"].process == "Code", "no habia proceso a mano; se toma el descubierto"


def test_nothing_to_replace_with_leaves_the_entry_as_is() -> None:
    """Sin candidata, mejor la entrada rota que una inventada: el mensaje de
    "no pude abrir X" es diagnosticable, abrir otra cosa no."""
    manual = {"spotify": AppSpec(command="spotify")}
    assert repair_manual_entries(manual, [], puede_lanzar=_NUNCA) == {}


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
