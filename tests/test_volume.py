"""Tests del control de volumen: resolucion de destino y eleccion de mecanismo.

Lo que de verdad se comprueba aqui no es que suba el volumen -eso solo se puede
ver en Windows- sino A QUE MANDO va cada comando. El sistema y Spotify usan APIs
distintas que no se hablan entre si, y equivocarse de mando produce el peor tipo
de fallo: el que parece funcionar porque algo se mueve, pero no lo que pedias.
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
        #: Cuantas veces se pidio el dispositivo por separado. Debe quedarse en
        #: 0: el id viene de la misma lectura que el volumen.
        self.devices_consultados = 0

    def volume_state(self) -> tuple[str, int] | None:
        """None cuando no hay dispositivo activo: Spotify cerrado o parado."""
        return None if self.volume is None else ("device-1", self.volume)

    def get_volume_percent(self) -> int | None:
        return self.volume

    def set_volume_percent(self, level: int, device_id: str | None = None) -> bool:
        if device_id is None:
            self.devices_consultados += 1
        if not self.writable:
            return False
        self.writes.append(level)
        self.volume = level
        return True


class FakeWinAudio:
    """Sustituto de `winaudio` con estado en memoria.

    `master=None` simula que la API de audio no esta disponible, que es cuando
    el volumen del PC cae a las teclas multimedia.
    """

    def __init__(self, master: int | None = 40) -> None:
        self.master = master
        self.master_mute = False

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
    controller = VolumeController(FakeSpotify())
    assert controller.step(VolumeTarget.SYSTEM, 10).ok is True
    assert win.master == 50


def test_system_step_clamps_at_the_top(win: FakeWinAudio) -> None:
    """Windows recorta solo, pero si el recorte se hiciera en el asistente con
    un signo mal puesto el volumen saltaria al 0% en vez de quedarse al 100%."""
    win.master = 95
    controller = VolumeController(FakeSpotify())
    controller.step(VolumeTarget.SYSTEM, 10)
    assert win.master == 100


def test_system_step_falls_back_to_media_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sin API de audio (pycaw ausente o COM roto) quedan las teclas. Suben de
    2% en 2%, asi que un delta de 10 son cinco pulsaciones."""
    monkeypatch.setattr(volume_module, "winaudio", FakeWinAudio(master=None))
    pulsaciones: list[Any] = []
    monkeypatch.setattr(volume_module, "press", lambda key: pulsaciones.append(key) or True)

    controller = VolumeController(FakeSpotify())
    assert controller.step(VolumeTarget.SYSTEM, 10).ok is True
    assert len(pulsaciones) == 5


def test_system_mute_toggles(win: FakeWinAudio) -> None:
    controller = VolumeController(FakeSpotify())
    controller.toggle_mute(VolumeTarget.SYSTEM)
    assert win.master_mute is True
    controller.toggle_mute(VolumeTarget.SYSTEM)
    assert win.master_mute is False


# --------------------------------------------------------------- spotify
#
# SIEMPRE la Web API, NUNCA el mezclador de Windows. Son mandos distintos: el de
# Spotify se ve en la barra de la app y se sincroniza con el movil; el del
# mezclador solo afecta a lo que sale por los altavoces de este PC. Bajar el que
# no es produce el fallo mas confuso posible, porque algo baja de volumen.


def test_spotify_always_goes_through_the_app(win: FakeWinAudio) -> None:
    spotify = FakeSpotify(volume=80)
    controller = VolumeController(spotify)

    outcome = controller.step(VolumeTarget.SPOTIFY, 10)

    assert outcome.ok is True
    assert outcome.via == "spotify-api"
    assert spotify.writes == [90]
    assert win.master == 40  # el del PC intacto


def test_spotify_step_clamps_without_a_wasted_call(win: FakeWinAudio) -> None:
    """Ya al 100%: se envia 100, no 110. `at_limit` deja que la skill lo diga."""
    spotify = FakeSpotify(volume=100)
    outcome = VolumeController(spotify).step(VolumeTarget.SPOTIFY, 10)

    assert spotify.writes == [100]
    assert outcome.at_limit is True


def test_spotify_fails_cleanly_when_nothing_is_playing(win: FakeWinAudio) -> None:
    """Sin dispositivo activo no hay alternativa: NO se degrada al mezclador,
    porque seria cambiar un volumen distinto del pedido y sin avisar."""
    outcome = VolumeController(FakeSpotify(volume=None)).step(VolumeTarget.SPOTIFY, 10)
    assert outcome.ok is False


def test_spotify_without_client_fails_instead_of_touching_the_mixer(
    win: FakeWinAudio,
) -> None:
    """Spotify sin autorizar: el comando falla y lo dice. Antes esto bajaba el
    volumen del mezclador, que es exactamente lo que no se quiere."""
    outcome = VolumeController(None).step(VolumeTarget.SPOTIFY, -20)
    assert outcome.ok is False


def test_spotify_reuses_the_device_from_the_state_read(win: FakeWinAudio) -> None:
    """El id del dispositivo sale de la misma lectura que el volumen. Pedirlo
    aparte serian dos viajes de red por comando, y la Web API esta en la ruta
    critica desde que es la unica via."""
    spotify = FakeSpotify(volume=50)
    VolumeController(spotify).step(VolumeTarget.SPOTIFY, 10)
    assert spotify.devices_consultados == 0


def test_spotify_mute_remembers_where_it_came_from(win: FakeWinAudio) -> None:
    """La Web API no tiene silencio: silenciar es poner 0. Sin recordar el valor
    previo, quitar el silencio seria imposible."""
    spotify = FakeSpotify(volume=70)
    controller = VolumeController(spotify)

    controller.toggle_mute(VolumeTarget.SPOTIFY)
    assert spotify.writes == [0]

    controller.toggle_mute(VolumeTarget.SPOTIFY)
    assert spotify.writes == [0, 70]


def test_spotify_unmute_without_memory_lands_somewhere_audible(
    win: FakeWinAudio,
) -> None:
    """Spotify ya estaba en 0 al arrancar el asistente: no hay valor previo que
    restaurar, y dejarlo en 0 haria que el comando pareciera roto."""
    spotify = FakeSpotify(volume=0)
    VolumeController(spotify).toggle_mute(VolumeTarget.SPOTIFY)
    assert spotify.writes == [50]


# ----------------------------------------------------------------- skills


def test_step_skill_says_what_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Los mensajes de error se separan a proposito: 'no encontré Spotify' es
    accionable, 'no pude cambiar el volumen' no dice donde mirar."""
    monkeypatch.setattr(volume_module, "winaudio", FakeWinAudio(master=None))
    monkeypatch.setattr(volume_module, "press", lambda key: False)
    controller = VolumeController(FakeSpotify(volume=None))

    pc = VolumeStepSkill(controller).execute(VolumeStepArgs(delta=10))
    musica = VolumeStepSkill(controller).execute(VolumeStepArgs(delta=10, target="spotify"))

    assert pc == SkillResult.failed("No pude cambiar el volumen.")
    assert musica == SkillResult.failed("No encontré Spotify sonando.")


def test_step_skill_prefers_an_explicit_level_over_the_step(win: FakeWinAudio) -> None:
    """"Sube el volumen al 50" gana `volume.up` en el router, no `volume.set`:
    el verbo pesa mas que el numero. La skill hace que esa duda no importe."""
    skill = VolumeStepSkill(VolumeController(FakeSpotify()))

    assert skill.execute(VolumeStepArgs(delta=10, level="50")).ok is True
    assert win.master == 50  # fijado, no 40+10
    assert skill.execute(VolumeStepArgs(delta=10, level="cincuenta y cinco")).ok is True
    assert win.master == 55


def test_step_skill_falls_back_to_the_step_when_the_level_is_nonsense(
    win: FakeWinAudio,
) -> None:
    """"Sube el volumen a tope" extrae algo que no es un numero. Mejor subir
    un paso que quedarse quieto: la intencion de subir estaba clara."""
    skill = VolumeStepSkill(VolumeController(FakeSpotify()))
    assert skill.execute(VolumeStepArgs(delta=10, level="tope")).ok is True
    assert win.master == 50  # 40 + 10, el paso normal


def test_step_skill_says_when_it_is_already_at_the_limit(win: FakeWinAudio) -> None:
    """Spotify al 100% y "súbele" no hace nada. El silencio ahi es
    indistinguible de que no te hubiera entendido."""
    skill = VolumeStepSkill(VolumeController(FakeSpotify(volume=100)))

    result = skill.execute(VolumeStepArgs(delta=10, target="spotify"))
    assert result.ok is True
    assert result.speech == "Spotify ya está al máximo."


def test_step_reports_the_mechanism_it_used(
    win: FakeWinAudio, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`via` existe para poder depurar "el volumen no funciona" leyendo el log
    en vez de probando los mecanismos a ciegas."""
    controller = VolumeController(FakeSpotify())
    assert controller.step(VolumeTarget.SYSTEM, 10).via == "api-maestra"
    assert controller.step(VolumeTarget.SPOTIFY, -10).via == "spotify-api"

    monkeypatch.setattr(volume_module, "winaudio", FakeWinAudio(master=None))
    monkeypatch.setattr(volume_module, "press", lambda key: True)
    assert controller.step(VolumeTarget.SYSTEM, 10).via.startswith("teclas")


def test_set_skill_accepts_numbers_in_words(win: FakeWinAudio) -> None:
    """Whisper devuelve "cuarenta" tanto como "40"; la skill coacciona ambos."""
    skill = VolumeSetSkill(VolumeController(FakeSpotify()))

    assert skill.execute(VolumeSetArgs(level="cuarenta")).ok is True
    assert win.master == 40
    assert skill.execute(VolumeSetArgs(level="75")).ok is True
    assert win.master == 75


def test_set_skill_routes_to_spotify_with_target(win: FakeWinAudio) -> None:
    spotify = FakeSpotify(volume=80)
    skill = VolumeSetSkill(VolumeController(spotify))

    assert skill.execute(VolumeSetArgs(level="treinta", target="la musica")).ok is True
    assert spotify.writes == [30]
    assert win.master == 40  # el del PC no se toca


def test_set_skill_rejects_unparseable_level(win: FakeWinAudio) -> None:
    """'pon el volumen a la mitad' llega hasta aqui; mejor decirlo que inventar
    un numero."""
    skill = VolumeSetSkill(VolumeController(FakeSpotify()))
    result = skill.execute(VolumeSetArgs(level="la mitad"))
    assert result.ok is False
    assert win.master == 40


def test_mute_skill_targets_only_spotify(win: FakeWinAudio) -> None:
    """Silenciar Spotify no debe silenciar el PC entero: es justo la diferencia
    que motiva todo este modulo."""
    spotify = FakeSpotify(volume=70)
    MuteSkill(VolumeController(spotify)).execute(volume_module.VolumeMuteArgs(target="spotify"))

    assert spotify.writes == [0]
    assert win.master_mute is False
