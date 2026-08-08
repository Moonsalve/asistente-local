"""Transcripcion con faster-whisper.

POR QUE NO MOONSHINE: el stack inicial contemplaba Moonshine tiny (~50 ms), pero
solo tiene modelos en ingles. No existe variante en espanol, asi que no era una
opcion.

POR QUE large-v3-turbo Y NO small: turbo ocupa ~1.6 GB en int8_float16 frente a
~0.5 GB de small, y en una tarjeta de 8 GB eso sobra. La precision en espanol es
notablemente mejor, y la precision del STT es la que marca el techo de todo lo
que viene despues: una palabra mal transcrita es un comando mal enrutado.

AJUSTES DE LATENCIA
-------------------
- `beam_size=1` (greedy): 30-40% mas rapido, perdida despreciable en frases
  cortas de comando.
- `condition_on_previous_text=False`: cada orden es independiente. Arrastrar la
  anterior invita a alucinaciones cuando el audio es corto.
- `vad_filter=False`: el endpointing ya lo hace `recorder.py`; repetirlo aqui
  solo anade trabajo.
"""

from __future__ import annotations

import logging
from time import perf_counter

import numpy as np

from asistente.config import SttConfig

log = logging.getLogger(__name__)

#: Por debajo de esto no hay palabra que transcribir, solo un chasquido.
MIN_AUDIO_S = 0.2


class Transcriber:
    def __init__(self, config: SttConfig) -> None:
        from faster_whisper import WhisperModel

        self._config = config
        started = perf_counter()
        self._model = WhisperModel(
            config.model,
            device=config.device,
            compute_type=config.compute_type,
        )
        log.info("whisper %s cargado en %.1f s", config.model, perf_counter() - started)

    def warmup(self) -> None:
        """Primera inferencia con audio sintetico.

        En CUDA la primera pasada compila kernels y cuesta segundos. Pagarlo al
        arrancar evita que el primer comando real parezca que se ha colgado.
        """
        started = perf_counter()
        self.transcribe(np.zeros(16_000, dtype=np.float32))
        log.info("whisper calentado en %.2f s", perf_counter() - started)

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16_000) -> str:
        if audio.size < MIN_AUDIO_S * sample_rate:
            return ""

        segments, _ = self._model.transcribe(
            audio.astype(np.float32),
            language=self._config.language,
            beam_size=self._config.beam_size,
            condition_on_previous_text=self._config.condition_on_previous_text,
            vad_filter=False,
        )
        return " ".join(segment.text.strip() for segment in segments).strip()
