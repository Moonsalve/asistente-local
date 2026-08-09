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
from dataclasses import dataclass
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


@dataclass(frozen=True, slots=True)
class VolumeChange:
    """Que paso al ejecutar un comando de volumen.

    Devolver esto en vez de un `bool` resuelve dos problemas a la vez. Uno es de
    diagnostico: hay cuatro mecanismos y sin saber cual actuo, "el volumen no
    funciona" no se puede depurar sin ir probando a ciegas. El otro es de uso:
    subir el volumen cuando ya esta al 100% "funciona" —nadie falla— pero no
    hace nada, y el usuario solo oye silencio por respuesta.
    """

    ok: bool
    via: str = ""
    before: int | None = None
    after: int | None = None

    @property
    def at_limit(self) -> bool:
        """Se ejecuto sin error pero el valor no se movio: ya estaba al tope."""
        return self.ok and self.before is not None and self.before == self.after

    def log(self, target: VolumeTarget, accion: str) -> None:
        log.info(
            "volumen %s (%s): %s vía %s (%s -> %s)",
            target.value,
            accion,
            "ok" if self.ok else "FALLO",
            self.via or "ninguna",
            self.before if self.before is not None else "?",
            self.after if self.after is not None else "?",
        )


_FALLO = VolumeChange(ok=False)


class VolumeController:
    """Un unico sitio donde vive la logica de 'a que mando le hablo'.

    Se inyecta en las skills para que no repitan la cascada de fallbacks y para
    poder testear la resolucion sin Windows.
    """

    def __init__(self, spotify: SpotifyClient | None, process_names: tuple[str, ...]) -> None:
        self._spotify = spotify
        self._process_names = process_names

    # ------------------------------------------------------------- sistema
    def _system_set(self, level: int) -> VolumeChange:
        before = winaudio.master_percent()
        if not winaudio.set_master_percent(level):
            return _FALLO
        return VolumeChange(True, "api-maestra", before, winaudio.master_percent())

    def _system_step(self, delta: int) -> VolumeChange:
        before = winaudio.master_percent()
        if before is not None and winaudio.set_master_percent(before + delta):
            return VolumeChange(True, "api-maestra", before, winaudio.master_percent())

        # Sin API de audio: aproximar con pulsaciones. Se redondea al alza para
        # que "sube el volumen" siempre haga algo perceptible.
        key = VirtualKey.VOLUME_UP if delta > 0 else VirtualKey.VOLUME_DOWN
        presses = max(1, round(abs(delta) / _KEY_STEP_PERCENT))
        # Se cuentan las que salen en vez de cortar en la primera que falle: si
        # una se pierde, mover el volumen la mitad es mejor que no moverlo, y el
        # log dice cuantas llegaron.
        enviadas = sum(1 for _ in range(presses) if press(key))
        if enviadas == 0:
            return _FALLO
        return VolumeChange(True, f"teclas ({enviadas}/{presses})", before, None)

    def _system_toggle_mute(self) -> VolumeChange:
        muted = winaudio.master_muted()
        if muted is not None and winaudio.set_master_muted(not muted):
            return VolumeChange(True, "api-maestra")
        if press(VirtualKey.VOLUME_MUTE):
            return VolumeChange(True, "teclas")
        return _FALLO

    # ------------------------------------------------------------- spotify
    def _spotify_set(self, level: int) -> VolumeChange:
        before = winaudio.app_percent(self._process_names)
        if winaudio.set_app_percent(self._process_names, level):
            after = winaudio.app_percent(self._process_names)
            return VolumeChange(True, "mezclador", before, after)
        if self._spotify is None:
            return _FALLO
        remoto = self._spotify.get_volume_percent()
        if not self._spotify.set_volume_percent(level):
            return _FALLO
        return VolumeChange(True, "spotify-api", remoto, level)

    def _spotify_step(self, delta: int) -> VolumeChange:
        before = winaudio.app_percent(self._process_names)
        if before is not None:
            if not winaudio.set_app_percent(self._process_names, before + delta):
                return _FALLO
            after = winaudio.app_percent(self._process_names)
            return VolumeChange(True, "mezclador", before, after)

        if self._spotify is None:
            return _FALLO
        remoto = self._spotify.get_volume_percent()
        if remoto is None:
            return _FALLO
        objetivo = max(0, min(100, remoto + delta))
        if not self._spotify.set_volume_percent(objetivo):
            return _FALLO
        return VolumeChange(True, "spotify-api", remoto, objetivo)

    def _spotify_toggle_mute(self) -> VolumeChange:
        muted = winaudio.app_muted(self._process_names)
        if muted is not None:
            if not winaudio.set_app_muted(self._process_names, not muted):
                return _FALLO
            return VolumeChange(True, "mezclador")
        # La Web API no tiene "mute": silenciar es poner el volumen a 0. No se
        # intenta deshacerlo porque no hay donde guardar el valor anterior de
        # forma fiable entre reinicios; para eso ya esta "pon spotify al 50".
        if self._spotify is not None and self._spotify.set_volume_percent(0):
            return VolumeChange(True, "spotify-api")
        return _FALLO

    # -------------------------------------------------------------- publico
    def set_level(self, target: VolumeTarget, level: int) -> VolumeChange:
        outcome = (
            self._spotify_set(level)
            if target is VolumeTarget.SPOTIFY
            else self._system_set(level)
        )
        outcome.log(target, f"fijar a {level}")
        return outcome

    def step(self, target: VolumeTarget, delta: int) -> VolumeChange:
        outcome = (
            self._spotify_step(delta)
            if target is VolumeTarget.SPOTIFY
            else self._system_step(delta)
        )
        outcome.log(target, f"paso {delta:+d}")
        return outcome

    def toggle_mute(self, target: VolumeTarget) -> VolumeChange:
        outcome = (
            self._spotify_toggle_mute()
            if target is VolumeTarget.SPOTIFY
            else self._system_toggle_mute()
        )
        outcome.log(target, "silencio")
        return outcome

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
    #: Nivel absoluto, si la frase lo llevaba. Manda sobre `delta`; ver la nota
    #: en `VolumeStepSkill.execute`.
    level: str | None = Field(default=None, max_length=40)


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

        # Un numero explicito manda sobre el paso: "sube el volumen al 50" es
        # fijar, no subir.
        #
        # Medido: "sube el volumen al 50" ganaba `volume.up` (0.635) en vez de
        # `volume.set`, porque el verbo pesa mas que el numero y los embeddings
        # tratan las cifras como ruido. Se puede empujar el catalogo, pero no se
        # puede garantizar que gane siempre. Asi que en vez de pelear con el
        # router se hace que su duda deje de importar: ambos intents aceptan el
        # nivel y hacen lo mismo cuando la frase lo lleva. Es el mismo criterio
        # que `open.target`: cuando la ambiguedad no es semantica, se resuelve
        # con datos y no con similitud.
        if args.level is not None and (level := parse_spanish_number(args.level)) is not None:
            outcome = self._controller.set_level(target, level)
            return SkillResult.silent() if outcome.ok else _failure(target, "cambiar")

        outcome = self._controller.step(target, args.delta)
        if not outcome.ok:
            return _failure(target, "cambiar")
        if outcome.at_limit:
            # Silencio aqui seria indistinguible de que no hubiera entendido.
            que = "Spotify" if target is VolumeTarget.SPOTIFY else "El volumen"
            tope = "al máximo" if args.delta > 0 else "al mínimo"
            return SkillResult.says(f"{que} ya está {tope}.")
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
        if not self._controller.set_level(target, level).ok:
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
        if not self._controller.toggle_mute(target).ok:
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
