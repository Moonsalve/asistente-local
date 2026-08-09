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
- **Sistema** (`target=system`, el que manda cuando la frase no dice otra cosa):
  el volumen maestro de Windows por `IAudioEndpointVolume`, que permite fijar un
  porcentaje exacto. Si la API de audio no esta disponible, teclas multimedia,
  que solo se mueven a saltos de 2%.
- **Spotify** (`target=spotify`): **siempre la Web API**, es decir el mando de
  dentro de la aplicacion. Nunca el mezclador de volumen de Windows: ese es otro
  mando distinto, solo afecta a lo que sale por los altavoces de este PC y no se
  refleja en ninguna parte de Spotify.

CON VALOR O SIN VALOR
---------------------
Sin numero en la frase se mueve un paso relativo; con numero se fija ese valor
exacto. Esto vale para los dos destinos y no depende de que intent gane el
router: ver la nota en `VolumeStepSkill.execute`.
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

    def __init__(self, spotify: SpotifyClient | None) -> None:
        self._spotify = spotify
        #: Volumen previo al silenciar, para poder deshacerlo. La Web API no
        #: tiene "mute": silenciar es poner 0, y sin recordar de donde venias no
        #: hay forma de volver. Vive en memoria y no en disco a proposito:
        #: silenciar y quitar el silencio pasan en la misma sesion, y persistir
        #: un valor entre reinicios daria sorpresas mucho peores que olvidarlo.
        self._spotify_volume_before_mute: int | None = None

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
    #
    # SIEMPRE la Web API, nunca el mezclador de Windows. Son dos mandos
    # distintos y el que la gente quiere decir con "el volumen de Spotify" es el
    # de dentro de la aplicacion: es el que se ve en la barra de Spotify y el
    # que se sincroniza con el movil y con los altavoces de Connect. El del
    # mezclador solo afecta a lo que este PC saca por los altavoces y no se ve
    # desde ninguna parte de Spotify.
    #
    # Se paga en latencia -un viaje de red frente a unos milisegundos- y en que
    # deja de funcionar si no hay dispositivo activo. A cambio, el comando hace
    # lo que dice. No hay degradacion al mezclador: seria hacer justo lo que no
    # se pidio, y en silencio.

    def _spotify_set(self, level: int) -> VolumeChange:
        if self._spotify is None:
            return _FALLO
        state = self._spotify.volume_state()
        if state is None:
            return _FALLO
        device, before = state
        if not self._spotify.set_volume_percent(level, device_id=device):
            return _FALLO
        return VolumeChange(True, "spotify-api", before, max(0, min(100, level)))

    def _spotify_step(self, delta: int) -> VolumeChange:
        if self._spotify is None:
            return _FALLO
        state = self._spotify.volume_state()
        if state is None:
            return _FALLO
        device, before = state
        objetivo = max(0, min(100, before + delta))
        if not self._spotify.set_volume_percent(objetivo, device_id=device):
            return _FALLO
        return VolumeChange(True, "spotify-api", before, objetivo)

    def _spotify_toggle_mute(self) -> VolumeChange:
        if self._spotify is None:
            return _FALLO
        state = self._spotify.volume_state()
        if state is None:
            return _FALLO
        device, before = state

        if before > 0:
            objetivo = 0
            self._spotify_volume_before_mute = before
        else:
            # Ya estaba en 0: esto es "quita el silencio". Si no sabemos de
            # donde venia -el asistente arranco con Spotify ya en 0- se deja a
            # la mitad, que es audible sin asustar.
            objetivo = self._spotify_volume_before_mute or 50

        if not self._spotify.set_volume_percent(objetivo, device_id=device):
            return _FALLO
        return VolumeChange(True, "spotify-api", before, objetivo)

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
