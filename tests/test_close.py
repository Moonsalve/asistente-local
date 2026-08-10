"""Tests del cierre de aplicaciones.

`taskkill` es de Windows y no se puede ejecutar aqui, pero lo que fallaba no era
taskkill: era DECIDIR QUE MATAR. Esa parte es logica pura y es la que se prueba.
"""

from __future__ import annotations

import pytest

from asistente.config import AppSpec, Config
from asistente.skills.apps import CloseAppSkill, CloseArgs, candidate_images, pick_running
from asistente.skills.winproc import normalize_image

VIVOS = ("System Idle Process", "explorer.exe", "Spotify.exe", "Code.exe", "chrome.exe")


def _config(**apps: AppSpec) -> Config:
    return Config(apps=dict(apps))


def test_a_declared_process_wins_over_everything() -> None:
    """Lo escrito a mano es lo unico que puso una persona sabiendo la respuesta."""
    spec = AppSpec(command="code", process="Code", aliases=("editor",))
    assert candidate_images("vscode", spec)[0] == "Code"


def test_the_command_gives_a_name_when_the_config_does_not() -> None:
    spec = AppSpec(command=r"C:\Program Files\Brave\brave.exe")
    assert "brave" in [c.lower() for c in candidate_images("navegador", spec)]


@pytest.mark.parametrize("command", ["shell:AppsFolder\\Pkg_8we!Spotify", "steam://rungameid/730"])
def test_non_executable_commands_contribute_no_name(command: str) -> None:
    """El `stem` de una URI es basura: de `steam://rungameid/730` saldria
    "rungameid", y matar un proceso llamado asi no le pasa a nadie... hasta que
    coincide."""
    candidatos = [normalize_image(c) for c in candidate_images("juego", AppSpec(command=command))]
    assert "rungameid" not in candidatos
    assert "pkg_8we!spotify" not in candidatos


def test_store_apps_are_closable_through_their_name() -> None:
    """EL CASO QUE ESTABA ROTO. Una app de la Store llega del descubrimiento con
    `process=None` y la skill se rendia antes de mirar. Pero corre como un .exe
    normal, y su nombre lo dice la propia entrada."""
    spec = AppSpec(command="shell:AppsFolder\\SpotifyAB.SpotifyMusic_zpd!Spotify",
                   aliases=("Spotify",))
    assert pick_running(candidate_images("spotify", spec), VIVOS) == "Spotify.exe"


def test_the_running_name_is_returned_with_its_real_case() -> None:
    spec = AppSpec(command="chrome", process="chrome")
    assert pick_running(candidate_images("chrome", spec), VIVOS) == "chrome.exe"


def test_a_wrong_guess_does_not_kill_a_lookalike() -> None:
    """El descubrimiento adivina el proceso pegando el titulo
    ("Visual Studio Code" -> "VisualStudioCode"), que no existe. Lo que NO puede
    pasar es que la busqueda se relaje hasta matar cualquier cosa parecida: se
    prefiere no cerrar nada."""
    spec = AppSpec(command=r"C:\...\Visual Studio Code.lnk", process="VisualStudioCode")
    assert pick_running(candidate_images("visual studio code", spec), ("Discord.exe",)) is None


def test_nothing_running_is_reported_as_not_open(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("asistente.skills.apps.can_list_processes", lambda: True)
    monkeypatch.setattr("asistente.skills.apps.running_images", lambda: VIVOS)
    skill = CloseAppSkill(_config(discord=AppSpec(command="discord", process="Discord")))

    result = skill.execute(CloseArgs(app="discord"))
    assert not result.ok
    assert "no está abierto" in (result.speech or "")


def test_an_app_outside_the_allowlist_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """La frontera de seguridad. Sin entrada en la allowlist no hay nada que
    matar, por muy claro que sea lo que se dijo."""
    monkeypatch.setattr("asistente.skills.apps.can_list_processes", lambda: True)
    monkeypatch.setattr("asistente.skills.apps.running_images", lambda: VIVOS)
    skill = CloseAppSkill(_config(chrome=AppSpec(command="chrome", process="chrome")))

    assert not skill.execute(CloseArgs(app="antivirus")).ok


def test_the_kill_targets_what_is_running(monkeypatch: pytest.MonkeyPatch) -> None:
    matados: list[str] = []
    monkeypatch.setattr("asistente.skills.apps.can_list_processes", lambda: True)
    monkeypatch.setattr("asistente.skills.apps.running_images", lambda: VIVOS)
    monkeypatch.setattr(
        "asistente.skills.apps.kill_image",
        lambda image: (matados.append(image), (True, ""))[1],
    )
    skill = CloseAppSkill(_config(spotify=AppSpec(command="spotify", process="Spotify")))

    assert skill.execute(CloseArgs(app="spotify")).ok
    assert matados == ["Spotify.exe"]


def test_when_the_process_list_is_unavailable_it_still_tries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`tasklist` puede estar restringido. Degradar al comportamiento anterior
    -matar lo que dice la config- es mejor que negarse a cerrar nada."""
    matados: list[str] = []
    monkeypatch.setattr("asistente.skills.apps.can_list_processes", lambda: True)
    monkeypatch.setattr("asistente.skills.apps.running_images", tuple)
    monkeypatch.setattr(
        "asistente.skills.apps.kill_image",
        lambda image: (matados.append(image), (True, ""))[1],
    )
    skill = CloseAppSkill(_config(spotify=AppSpec(command="spotify", process="Spotify")))

    assert skill.execute(CloseArgs(app="spotify")).ok
    assert matados == ["Spotify"]


def test_a_failed_kill_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    """Antes se decia "no estaba abierto" pasara lo que pasara, que es mentira
    cuando el proceso esta y el sistema deniega el permiso."""
    monkeypatch.setattr("asistente.skills.apps.can_list_processes", lambda: True)
    monkeypatch.setattr("asistente.skills.apps.running_images", lambda: VIVOS)
    monkeypatch.setattr("asistente.skills.apps.kill_image",
                        lambda image: (False, "Acceso denegado"))
    skill = CloseAppSkill(_config(spotify=AppSpec(command="spotify", process="Spotify")))

    result = skill.execute(CloseArgs(app="spotify"))
    assert not result.ok
    assert "no está abierto" not in (result.speech or "")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("Spotify.exe", "spotify"), ("  Code.EXE ", "code"), ("chrome", "chrome")],
)
def test_image_names_compare_without_case_or_extension(raw: str, expected: str) -> None:
    assert normalize_image(raw) == expected
