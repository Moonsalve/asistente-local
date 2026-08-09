"""Grabacion de una orden: desde la wake word hasta que dejas de hablar.

Implementa el endpointing sobre el VAD. La maquina de estados es deliberadamente
simple:

    esperando voz  --(prob > umbral)-->  hablando  --(silencio > silence_s)-->  fin

El preroll se antepone siempre: cuando el VAD confirma que hay voz ya se han
perdido 100-200 ms de la primera silaba, y Whisper transcribe mucho peor una
palabra truncada que una con algo de silencio delante.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from asistente.audio.vad import SileroVad
from asistente.config import VadConfig

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Utterance:
    """Lo grabado, mas lo que el VAD vio mientras lo grababa.

    `speech_s` es la clave para filtrar ruido barato: un ventilador que roza el
    umbral produce grabaciones largas con muy poca voz dentro. Distinguir
    "cuanto duro" de "cuanto de eso fue voz" descarta esas frases antes de
    gastar el STT, que cuesta cien veces mas que este contador.
    """

    audio: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    speech_s: float = 0.0
    total_s: float = 0.0

    @property
    def empty(self) -> bool:
        return self.audio.size == 0


class UtteranceRecorder:
    def __init__(self, vad: SileroVad, config: VadConfig, sample_rate: int = 16_000) -> None:
        self._vad = vad
        self._config = config
        self._sample_rate = sample_rate

    def record(self, blocks: object, preroll: np.ndarray) -> Utterance:
        """Consume bloques hasta detectar el final de la frase.

        Devuelve una `Utterance` vacia si nunca se detecto voz (activacion en
        falso de la wake word).
        """
        self._vad.reset()
        collected: list[np.ndarray] = [preroll] if preroll.size else []
        speaking = False
        silence_s = 0.0
        total_s = 0.0
        speech_s = 0.0

        for block in blocks:  # type: ignore[attr-defined]
            block_s = len(block) / self._sample_rate
            total_s += block_s
            collected.append(block)

            # `None` significa que aun no hay un frame completo para el modelo
            # (los bloques del microfono y los frames del VAD no miden lo
            # mismo). Ese tiempo SIGUE contando como silencio: descartarlo hacia
            # que el corte llegase mas tarde de lo configurado.
            prob = self._vad.speech_probability(block)
            if prob is not None and prob >= self._config.speech_threshold:
                speaking = True
                silence_s = 0.0
                speech_s += block_s
            elif speaking:
                silence_s += block_s

            if speaking and silence_s >= self._config.silence_s:
                log.debug("fin de frase tras %.2f s de silencio", silence_s)
                break

            if total_s >= self._config.max_utterance_s:
                log.warning("corte por duracion maxima (%.1f s)", total_s)
                break

            # Activacion en falso: la wake word disparo pero nadie hablo.
            if not speaking and total_s >= 2.0:
                log.debug("no se detecto voz tras la wake word")
                return Utterance(total_s=total_s)

        audio = np.concatenate(collected) if collected else np.zeros(0, dtype=np.float32)
        return Utterance(audio=audio, speech_s=speech_s, total_s=total_s)
