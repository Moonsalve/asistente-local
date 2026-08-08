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
from asistente.cuda_setup import ensure_cuda_dlls

# Tiene que ejecutarse antes de que CTranslate2 intente cargar sus DLL, es
# decir antes del primer `import faster_whisper` (que es perezoso, mas abajo).
ensure_cuda_dlls()

log = logging.getLogger(__name__)

#: Por debajo de esto no hay palabra que transcribir, solo un chasquido.
MIN_AUDIO_S = 0.2

#: Compute type al degradar a CPU. float16 no existe en CPU; int8 es lo unico
#: con una velocidad tolerable.
_CPU_COMPUTE_TYPE = "int8"


class Transcriber:
    def __init__(self, config: SttConfig) -> None:
        self._config = config
        self._device = config.device
        self._compute_type = config.compute_type
        self._model = self._load(self._device, self._compute_type)

    def _load(self, device: str, compute_type: str) -> object:
        from faster_whisper import WhisperModel

        started = perf_counter()
        model = WhisperModel(self._config.model, device=device, compute_type=compute_type)
        log.info(
            "whisper %s cargado en %s (%s) en %.1f s",
            self._config.model,
            device,
            compute_type,
            perf_counter() - started,
        )
        return model

    def warmup(self) -> None:
        """Primera inferencia con audio sintetico.

        En CUDA la primera pasada compila kernels y cuesta segundos. Pagarlo al
        arrancar evita que el primer comando real parezca que se ha colgado.

        Tambien es donde se detecta que CUDA no es utilizable: CTranslate2 carga
        el modelo sin quejarse y solo falla al ejecutar el encoder. Antes que
        abortar el arranque, se degrada a CPU: el asistente va mas lento pero
        funciona, y asi se puede probar el resto del pipeline mientras se
        arregla la instalacion de CUDA.
        """
        started = perf_counter()
        try:
            self.transcribe(np.zeros(16_000, dtype=np.float32))
        except RuntimeError as exc:
            if self._device == "cpu":
                raise
            log.error("CUDA no utilizable (%s)", exc)
            log.warning(
                "degradando el STT a CPU. Para recuperar la GPU: "
                "pip install nvidia-cublas-cu12 nvidia-cudnn-cu12"
            )
            self._device, self._compute_type = "cpu", _CPU_COMPUTE_TYPE
            self._model = self._load(self._device, self._compute_type)
            self.transcribe(np.zeros(16_000, dtype=np.float32))

        log.info("whisper calentado en %.2f s (%s)", perf_counter() - started, self._device)

    @property
    def device(self) -> str:
        """Donde corre realmente, que puede no ser lo que pedia la config si
        hubo degradacion a CPU."""
        return self._device

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16_000) -> str:
        if audio.size < MIN_AUDIO_S * sample_rate:
            return ""

        segments, _ = self._model.transcribe(  # type: ignore[attr-defined]
            audio.astype(np.float32),
            language=self._config.language,
            beam_size=self._config.beam_size,
            condition_on_previous_text=self._config.condition_on_previous_text,
            vad_filter=False,
        )
        return " ".join(segment.text.strip() for segment in segments).strip()
