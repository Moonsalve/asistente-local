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

import numpy as np

from asistente.audio.vad import SileroVad
from asistente.config import VadConfig

log = logging.getLogger(__name__)


class UtteranceRecorder:
    def __init__(self, vad: SileroVad, config: VadConfig, sample_rate: int = 16_000) -> None:
        self._vad = vad
        self._config = config
        self._sample_rate = sample_rate

    def record(self, blocks: object, preroll: np.ndarray) -> np.ndarray:
        """Consume bloques hasta detectar el final de la frase.

        Devuelve el audio completo, o un array vacio si nunca se detecto voz
        (activacion en falso de la wake word).
        """
        self._vad.reset()
        collected: list[np.ndarray] = [preroll] if preroll.size else []
        speaking = False
        silence_s = 0.0
        total_s = 0.0

        for block in blocks:  # type: ignore[attr-defined]
            block_s = len(block) / self._sample_rate
            total_s += block_s
            collected.append(block)

            prob = self._vad.speech_probability(block)
            if prob is None:
                continue

            if prob >= self._config.speech_threshold:
                speaking = True
                silence_s = 0.0
            elif speaking:
                silence_s += block_s
                if silence_s >= self._config.silence_s:
                    log.debug("fin de frase tras %.2f s de silencio", silence_s)
                    break

            if total_s >= self._config.max_utterance_s:
                log.warning("corte por duracion maxima (%.1f s)", total_s)
                break

            # Activacion en falso: la wake word disparo pero nadie hablo.
            if not speaking and total_s >= 2.0:
                log.debug("no se detecto voz tras la wake word")
                return np.zeros(0, dtype=np.float32)

        return np.concatenate(collected) if collected else np.zeros(0, dtype=np.float32)
