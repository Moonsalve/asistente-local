"""Tests del control de volumen: resolucion de destino y cascada de fallbacks.

La parte que de verdad importa aqui no es que suba el volumen -eso solo se puede
comprobar en Windows- sino el ORDEN en que se intentan los mecanismos. Cada
destino tiene dos vias y elegir mal la primera es la diferencia entre un comando
instantaneo y uno que hace un viaje de red para nada, o que no hace nada.
"""

from __future__ import annotations

from typing import Any

import pytest

from asistente.skills import volume as volume_module
from asistente.skills.base import SkillResult
from asistente.skills.volume import (
    MuteSkill,
    VolumeController,
    VolumeSetArgs,
    VolumeSetSkill,
    VolumeStepArgs,
    VolumeStepSkill,
    VolumeTarget,
    resolve_target,
)

PROCESOS = ("Spotify.exe",)


# ------------------------------------------------- formas de la API de pycaw


class _EndpointModerno:
    """pycaw >= 20251023: `GetSpeakers()` devuelve un `AudioDevice` que ya trae
    la interfaz hecha en `EndpointVolume`."""

    def __init__(self) -> None:
        self.EndpointVolume = "interfaz-lista"


class _EndpointAntiguo:
    """pycaw <= 20240210: devolvia el `IMMDevice` crudo, que hay que activar."""

    def __init__(self) -> None:
        self.activado = False

    def Activate(self, iid: Any, clsctx: Any, params: Any) -> str:  # noqa: N802
        self.activado = True
        return "interfaz-cruda"


class _IfaceFalsa:
    _iid_ = "{fake}"


def test_endpoint_acepta_la_forma_nueva_de_pycaw() -> None:
    """La que rompio el volumen del PC: pycaw 20251023 devuelve un envoltorio
    sin `Activate`, y el codigo lo llamaba igualmente."""
    from asistente.skills.winaudio import endpoint_from_speakers

    resultado = endpoint_from_speakers(
        _EndpointModerno(), _IfaceFalsa, None, lambda x, t: x, lambda t: t
    )
    assert resultado == "interfaz-lista"


def test_endpoint_sigue_aceptando_la_forma_antigua() -> None:
    """El pyproject pide `pycaw>=20240210`, asi que la rama vieja sigue viva."""
    from asistente.skills.winaudio import endpoint_from_speakers

    speakers = _EndpointAntiguo()
    resultado = endpoint_from_speakers(
        speakers, _IfaceFalsa, None, lambda x, t: x, lambda t: t
    )
    assert speakers.activado is True
    assert resultado == "interfaz-cruda"


class FakeSpotify:
    """Cliente de Spotify de mentira. `volume` es None cuando no hay dispositivo
    activo, que es el caso real que obliga a degradar."""

    def __init__(self, volume: int | None = 50, writable: bool = True) -> None:
        self.volume = volume
        self.writable = writable
        self.writes: list[int] = []

    def get_volume_percent(self) -> int | None:
        return self.volume

    def set_volume_percent(self, level: int) -> bool:
        if not self.writable:
            return False
        self.writes.append(level)
        self.volume = level
        return True


class FakeWinAudio:
    """Sustituto de `winaudio` con estado en memoria.

    `app_percent` devuelve None cuando Spotify no tiene sesion en el mezclador
    (cerrado, o abierto pero sin haber sonado nunca).
    """

    def __init__(self, master: int | None = 40, app: int | None = 80) -> None:
        self.master = master
        self.app = app
        self.master_mute = False
        self.app_mute: bool | None = False

    def master_percent(self) -> int | None:
        return self.master

    def set_master_percent(self, level: int) -> bool:
        if self.master is None:
            return False
        self.master = max(0, min(100, level))
        return True

    def master_muted(self) -> bool | None:
        return None if self.master is None else self.master_mute

    def set_master_muted(self, muted: bool) -> bool:
        if self.master is None:
            return False
        self.master_mute = muted
        return True

    def app_percent(self, names: Any) -> int | None:
        return self.app

    def set_app_percent(self, names: Any, level: int) -> bool:
        if self.app is None:
            return False
        self.app = max(0, min(100, level))
        return True

    def app_muted(self, names: Any) -> bool | None:
        return self.app_mute

    def set_app_muted(self, names: Any, muted: bool) -> bool:
        if self.app is None:
            return False
        self.app_mute = muted
        return True


@pytest.fixture
def win(monkeypatch: pytest.MonkeyPatch) -> FakeWinAudio:
    fake = FakeWinAudio()
    monkeypatch.setattr(volume_module, "winaudio", fake)
    return fake


# --------------------------------------------------------------- destino


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("spotify", VolumeTarget.SPOTIFY),
        ("la musica", VolumeTarget.SPOTIFY),
        ("la cancion", VolumeTarget.SPOTIFY),
        ("el reproductor", VolumeTarget.SPOTIFY),
        # Whisper transcribe asi la marca mas a menudo de lo que parece.
        ("espotifai", VolumeTarget.SPOTIFY),
        # Sin destino, o con uno que no significa nada, manda el PC.
        (None, VolumeTarget.SYSTEM),
        ("", VolumeTarget.SYSTEM),
        ("el pc", VolumeTarget.SYSTEM),
        ("windows", VolumeTarget.SYSTEM),
    ],
)
def test_resolve_target(raw: str | None, expected: VolumeTarget) -> None:
    assert resolve_target(raw) is expected


def test_target_defaults_to_system_when_absent() -> None:
    """El campo tiene default para que el catalogo no tenga que declararlo en
    `fixed_args` de los cuatro intents."""
    assert VolumeStepArgs(delta=10).resolved_target is VolumeTarget.SYSTEM


# --------------------------------------------------------------- sistema


def test_system_step_uses_the_audio_api(win: FakeWinAudio) -> None:
    controller = VolumeController(FakeSpotify(), PROCESOS)
    assert controller.step(VolumeTarget.SYSTEM, 10) is True
    assert win.master == 50


def test_system_step_clamps_at_the_top(win: FakeWinAudio) -> None:
    """Windows recorta solo, pero si el recorte se hiciera en el asistente con
    un signo mal puesto el volumen saltaria al 0% en vez de quedarse al 100%."""
    win.master = 95
    controller = VolumeController(FakeSpotify(), PROCESOS)
    controller.step(VolumeTarget.SYSTEM, 10)
    assert win.master == 100


def test_system_step_falls_back_to_media_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sin API de audio (pycaw ausente o COM roto) quedan las teclas. Suben de
    2% en 2%, asi que un delta de 10 son cinco pulsaciones."""
    monkeypatch.setattr(volume_module, "winaudio", FakeWinAudio(master=None))
    pulsaciones: list[Any] = []
    monkeypatch.setattr(volume_module, "press", lambda key: pulsaciones.append(key) or True)

    controller = VolumeController(FakeSpotify(), PROCESOS)
    assert controller.step(VolumeTarget.SYSTEM, 10) is True
    assert len(pulsaciones) == 5


def test_system_mute_toggles(win: FakeWinAudio) -> None:
    controller = VolumeController(FakeSpotify(), PROCESOS)
    controller.toggle_mute(VolumeTarget.SYSTEM)
    assert win.master_mute is True
    controller.toggle_mute(VolumeTarget.SYSTEM)
    assert win.master_mute is False


# --------------------------------------------------------------- spotify


def test_spotify_prefers_the_windows_mixer(win: FakeWinAudio) -> None:
    """El mezclador va primero: es local, instantaneo y funciona sin dispositivo
    de Connect. La Web API solo entra cuando aqui no hay nada."""
    spotify = FakeSpotify()
    controller = VolumeController(spotify, PROCESOS)

    assert controller.step(VolumeTarget.SPOTIFY, 10) is True
    assert win.app == 90
    assert spotify.writes == []


def test_spotify_falls_back_to_the_web_api(win: FakeWinAudio) -> None:
    """Sin sesion local -musica sonando en el movil o en un altavoz- la Web API
    es el unico mando que existe."""
    win.app = None
    spotify = FakeSpotify(volume=50)
    controller = VolumeController(spotify, PROCESOS)

    assert controller.step(VolumeTarget.SPOTIFY, 10) is True
    assert spotify.writes == [60]


def test_spotify_fails_cleanly_when_nothing_is_playing(win: FakeWinAudio) -> None:
    """Ni sesion local ni dispositivo activo: se devuelve False para que la
    skill lo diga en voz alta en vez de fingir que funciono."""
    win.app = None
    controller = VolumeController(FakeSpotify(volume=None), PROCESOS)
    assert controller.step(VolumeTarget.SPOTIFY, 10) is False


def test_spotify_without_client_still_uses_the_mixer(win: FakeWinAudio) -> None:
    """Spotify sin autorizar (sin client_id) no impide bajarle el volumen: el
    mezclador de Windows no necesita credenciales."""
    controller = VolumeController(None, PROCESOS)
    assert controller.step(VolumeTarget.SPOTIFY, -20) is True
    assert win.app == 60


def test_spotify_mute_via_web_api_sets_volume_to_zero(win: FakeWinAudio) -> None:
    """La Web API no tiene silencio; lo mas parecido es el 0%."""
    win.app = None
    win.app_mute = None
    spotify = FakeSpotify()
    controller = VolumeController(spotify, PROCESOS)

    assert controller.toggle_mute(VolumeTarget.SPOTIFY) is True
    assert spotify.writes == [0]


# ----------------------------------------------------------------- skills


def test_step_skill_says_what_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Los mensajes de error se separan a proposito: 'no encontré Spotify' es
    accionable, 'no pude cambiar el volumen' no dice donde mirar."""
    monkeypatch.setattr(volume_module, "winaudio", FakeWinAudio(master=None, app=None))
    monkeypatch.setattr(volume_module, "press", lambda key: False)
    controller = VolumeController(FakeSpotify(volume=None), PROCESOS)

    pc = VolumeStepSkill(controller).execute(VolumeStepArgs(delta=10))
    musica = VolumeStepSkill(controller).execute(VolumeStepArgs(delta=10, target="spotify"))

    assert pc == SkillResult.failed("No pude cambiar el volumen.")
    assert musica == SkillResult.failed("No encontré Spotify sonando.")


def test_set_skill_accepts_numbers_in_words(win: FakeWinAudio) -> None:
    """Whisper devuelve "cuarenta" tanto como "40"; la skill coacciona ambos."""
    skill = VolumeSetSkill(VolumeController(FakeSpotify(), PROCESOS))

    assert skill.execute(VolumeSetArgs(level="cuarenta")).ok is True
    assert win.master == 40
    assert skill.execute(VolumeSetArgs(level="75")).ok is True
    assert win.master == 75


def test_set_skill_routes_to_spotify_with_target(win: FakeWinAudio) -> None:
    skill = VolumeSetSkill(VolumeController(FakeSpotify(), PROCESOS))
    assert skill.execute(VolumeSetArgs(level="treinta", target="la musica")).ok is True
    assert win.app == 30
    assert win.master == 40  # el del PC no se toca


def test_set_skill_rejects_unparseable_level(win: FakeWinAudio) -> None:
    """'pon el volumen a la mitad' llega hasta aqui; mejor decirlo que inventar
    un numero."""
    skill = VolumeSetSkill(VolumeController(FakeSpotify(), PROCESOS))
    result = skill.execute(VolumeSetArgs(level="la mitad"))
    assert result.ok is False
    assert win.master == 40


def test_mute_skill_targets_only_spotify(win: FakeWinAudio) -> None:
    """Silenciar Spotify no debe silenciar el PC entero: es justo la diferencia
    que motiva todo este modulo."""
    MuteSkill(VolumeController(FakeSpotify(), PROCESOS)).execute(
        volume_module.VolumeMuteArgs(target="spotify")
    )
    assert win.app_mute is True
    assert win.master_mute is False
