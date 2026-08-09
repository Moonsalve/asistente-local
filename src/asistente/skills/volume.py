"""Skills de volumen: del PC y de Spotify por separado.

POR QUE UN SLOT `target` Y NO INTENTS SEPARADOS
-----------------------------------------------
"Sube el volumen" y "sube el volumen de Spotify" son la misma intencion con un
destino distinto. Partirlas en dos intents las haria indistinguibles: comparten
casi todas las palabras y el coseno centrado no las separa de forma fiable —el
mismo problema medido con `app.open` y `web.open`, que acabaron fusionadas en
`open.target`.

Asi que hay UN intent por direccion (subir, bajar, fijar, silenciar) y el
destino viaja como slot. Como los `slot_patterns` se aplican *despues* de que el
intent ya esta decidido, el regex del destino no desambigua nada: solo lee. Si
no aparece ningun destino en la frase, manda `fixed_args` y el destino es el
sistema, que es lo que uno espera al decir "sube el volumen" a secas.

COMO SE RESUELVE CADA DESTINO
-----------------------------
- **Sistema**: `IAudioEndpointVolume` (porcentaje exacto). Si falla, teclas
  multimedia, que solo se mueven a saltos de 2%.
- **Spotify**: primero la sesion del mezclador de Windows (instantanea, no
  necesita autenticacion, funciona aunque no haya dispositivo de Connect); si
  Spotify no tiene audio abierto en este PC, la Web API, que es lo unico que
  sirve cuando la musica esta sonando en el movil o en un altavoz.
"""

from __future__ import annotations

import logging
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from asistente.numbers import parse_spanish_number
from asistente.skills import winaudio
from asistente.skills.base import Skill, SkillResult
from asistente.skills.spotify import SpotifyClient
from asistente.skills.winkeys import VirtualKey, press

log = logging.getLogger(__name__)

#: Cada pulsacion de VOLUME_UP/DOWN mueve el volumen de Windows un 2%.
_KEY_STEP_PERCENT = 2


class VolumeTarget(StrEnum):
    SYSTEM = "system"
    SPOTIFY = "spotify"


#: Palabras que, dichas dentro de un comando de volumen, significan "la musica"
#: y no "el PC". Van sin acentos porque el slot llega ya normalizado.
#: "spotifai"/"espotifai" estan porque Whisper transcribe asi la marca a menudo.
_SPOTIFY_HINTS = (
    "spotify",
    "spotifai",
    "espotifai",
    "musica",
    "cancion",
    "tema",
    "reproductor",
)


def resolve_target(raw: str | None) -> VolumeTarget:
    """Destino a partir del slot crudo. Todo lo que no suene a musica es el PC.

    El sesgo es deliberado: equivocarse hacia el sistema deja el comando a medias
    de forma obvia y corregible, mientras que equivocarse hacia Spotify cuando
    Spotify no esta sonando deja el comando sin efecto ninguno.
    """
    if not raw:
        return VolumeTarget.SYSTEM
    lowered = raw.strip().lower()
    if any(hint in lowered for hint in _SPOTIFY_HINTS):
        return VolumeTarget.SPOTIFY
    return VolumeTarget.SYSTEM


class VolumeController:
    """Un unico sitio donde vive la logica de 'a que mando le hablo'.

    Se inyecta en las tres skills para que no repitan la cascada de fallbacks y
    para poder testear la resolucion sin Windows.
    """

    def __init__(self, spotify: SpotifyClient | None, process_names: tuple[str, ...]) -> None:
        self._spotify = spotify
        self._process_names = process_names

    # ------------------------------------------------------------- sistema
    def _system_set(self, level: int) -> bool:
        return winaudio.set_master_percent(level)

    def _system_step(self, delta: int) -> bool:
        current = winaudio.master_percent()
        if current is not None and winaudio.set_master_percent(current + delta):
            return True
        # Sin API de audio: aproximar con pulsaciones. Se redondea al alza para
        # que "sube el volumen" siempre haga algo perceptible.
        key = VirtualKey.VOLUME_UP if delta > 0 else VirtualKey.VOLUME_DOWN
        presses = max(1, round(abs(delta) / _KEY_STEP_PERCENT))
        return all(press(key) for _ in range(presses))

    def _system_toggle_mute(self) -> bool:
        muted = winaudio.master_muted()
        if muted is not None and winaudio.set_master_muted(not muted):
            return True
        return press(VirtualKey.VOLUME_MUTE)

    # ------------------------------------------------------------- spotify
    def _spotify_set(self, level: int) -> bool:
        if winaudio.set_app_percent(self._process_names, level):
            return True
        return self._spotify is not None and self._spotify.set_volume_percent(level)

    def _spotify_step(self, delta: int) -> bool:
        current = winaudio.app_percent(self._process_names)
        if current is not None:
            return winaudio.set_app_percent(self._process_names, current + delta)
        if self._spotify is None:
            return False
        remote = self._spotify.get_volume_percent()
        return remote is not None and self._spotify.set_volume_percent(remote + delta)

    def _spotify_toggle_mute(self) -> bool:
        muted = winaudio.app_muted(self._process_names)
        if muted is not None:
            return winaudio.set_app_muted(self._process_names, not muted)
        # La Web API no tiene "mute": silenciar es poner el volumen a 0. No se
        # intenta deshacerlo porque no hay donde guardar el valor anterior de
        # forma fiable entre reinicios; para eso ya esta "pon spotify al 50".
        return self._spotify is not None and self._spotify.set_volume_percent(0)

    # -------------------------------------------------------------- publico
    def set_level(self, target: VolumeTarget, level: int) -> bool:
        if target is VolumeTarget.SPOTIFY:
            return self._spotify_set(level)
        return self._system_set(level)

    def step(self, target: VolumeTarget, delta: int) -> bool:
        if target is VolumeTarget.SPOTIFY:
            return self._spotify_step(delta)
        return self._system_step(delta)

    def toggle_mute(self, target: VolumeTarget) -> bool:
        if target is VolumeTarget.SPOTIFY:
            return self._spotify_toggle_mute()
        return self._system_toggle_mute()

    def current(self, target: VolumeTarget) -> int | None:
        if target is VolumeTarget.SPOTIFY:
            local = winaudio.app_percent(self._process_names)
            if local is not None:
                return local
            return self._spotify.get_volume_percent() if self._spotify else None
        return winaudio.master_percent()


def _failure(target: VolumeTarget, verb: str) -> SkillResult:
    if target is VolumeTarget.SPOTIFY:
        return SkillResult.failed("No encontré Spotify sonando.")
    return SkillResult.failed(f"No pude {verb} el volumen.")


class _TargetedArgs(BaseModel):
    """Base con el destino. Llega como texto crudo del slot ('la musica',
    'spotify') y lo coacciona la skill, igual que con `level`: el router extrae,
    la skill valida."""

    model_config = ConfigDict(extra="forbid")

    target: str = Field(default="system", max_length=40)

    @property
    def resolved_target(self) -> VolumeTarget:
        return resolve_target(self.target)


class VolumeStepArgs(_TargetedArgs):
    #: Positivo sube, negativo baja. Viene de `fixed_args` en el catalogo.
    delta: int = Field(ge=-100, le=100)


class VolumeStepSkill(Skill):
    name = "volume.step"
    args_model = VolumeStepArgs
    description = (
        "Sube o baja el volumen una cantidad relativa. "
        "target='spotify' para la musica, 'system' para el PC."
    )

    def __init__(self, controller: VolumeController) -> None:
        self._controller = controller

    def execute(self, args: BaseModel) -> SkillResult:
        assert isinstance(args, VolumeStepArgs)
        target = args.resolved_target
        if not self._controller.step(target, args.delta):
            return _failure(target, "cambiar")
        return SkillResult.silent()


class VolumeSetArgs(_TargetedArgs):
    #: Llega como texto porque el router extrae la frase cruda: puede ser "40"
    #: o "cuarenta" segun como lo transcriba Whisper.
    level: str = Field(min_length=1, max_length=40)


class VolumeSetSkill(Skill):
    name = "volume.set"
    args_model = VolumeSetArgs
    description = (
        "Fija el volumen a un porcentaje concreto (0-100). "
        "target='spotify' para la musica, 'system' para el PC."
    )

    def __init__(self, controller: VolumeController) -> None:
        self._controller = controller

    def execute(self, args: BaseModel) -> SkillResult:
        assert isinstance(args, VolumeSetArgs)
        level = parse_spanish_number(args.level)
        if level is None:
            return SkillResult.failed("No entendí a qué volumen lo pongo.")
        target = args.resolved_target
        if not self._controller.set_level(target, level):
            return _failure(target, "cambiar")
        return SkillResult.silent()


class VolumeMuteArgs(_TargetedArgs):
    pass


class MuteSkill(Skill):
    name = "volume.mute"
    args_model = VolumeMuteArgs
    description = (
        "Silencia o quita el silencio. "
        "target='spotify' para la musica, 'system' para todo el PC."
    )

    def __init__(self, controller: VolumeController) -> None:
        self._controller = controller

    def execute(self, args: BaseModel) -> SkillResult:
        assert isinstance(args, VolumeMuteArgs)
        target = args.resolved_target
        if not self._controller.toggle_mute(target):
            return _failure(target, "silenciar")
        return SkillResult.silent()


class VolumeQuerySkill(Skill):
    name = "volume.query"
    args_model = VolumeMuteArgs
    description = "Dice a que volumen esta el PC o Spotify."

    def __init__(self, controller: VolumeController) -> None:
        self._controller = controller

    def execute(self, args: BaseModel) -> SkillResult:
        assert isinstance(args, VolumeMuteArgs)
        target = args.resolved_target
        level = self._controller.current(target)
        if level is None:
            return _failure(target, "leer")
        que = "Spotify" if target is VolumeTarget.SPOTIFY else "El volumen"
        return SkillResult.says(f"{que} está al {level} por ciento.")
