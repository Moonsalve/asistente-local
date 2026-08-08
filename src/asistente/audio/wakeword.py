"""Deteccion de palabra clave con openWakeWord.

Corre permanentemente en CPU (1-2% de un nucleo) sobre modelos ONNX. Es la unica
parte del sistema siempre activa, por eso no toca ni GPU ni red.

PERIODO REFRACTARIO: tras una activacion se ignoran nuevas detecciones durante
unos segundos. Sin esto, una sola pronunciacion de la palabra clave dispara
varias veces seguidas, porque el modelo puntua alto en varios frames
consecutivos de la misma palabra.

WAKE WORD PROPIA: v1 usa un modelo preentrenado (`hey_jarvis`). Para uno propio,
openWakeWord trae un pipeline que genera miles de muestras sinteticas con Piper
mas augmentacion; el .onnx resultante se apunta en `config.yaml` sin tocar codigo.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def models_dir() -> Path:
    import openwakeword

    return Path(openwakeword.__file__).parent / "resources" / "models"


def models_are_installed() -> bool:
    """openWakeWord NO incluye los .onnx en el paquete pip.

    El wheel trae solo el codigo; los modelos (el de la palabra clave, el
    melspectrogram, el embedding y el VAD de Silero) hay que descargarlos una
    vez. Si no, el sintoma es un NO_SUCHFILE de onnxruntime sobre un fichero
    dentro del propio site-packages, que despista mucho porque parece una
    instalacion corrupta.
    """
    directory = models_dir()
    # melspectrogram y embedding_model los necesita CUALQUIER wake word;
    # silero_vad lo usa ademas nuestro endpointing.
    required = ("melspectrogram.onnx", "embedding_model.onnx", "silero_vad.onnx")
    return all((directory / name).is_file() for name in required)


def download_models() -> None:
    """Descarga los modelos de openWakeWord (~30 MB). Idempotente."""
    import openwakeword.utils

    log.info("descargando los modelos de openWakeWord (solo la primera vez)...")
    openwakeword.utils.download_models()
    log.info("modelos de openWakeWord listos en %s", models_dir())


class WakeWordDetector:
    def __init__(self, model: str = "hey_jarvis", threshold: float = 0.5, refractory_s: float = 2.0) -> None:
        from openwakeword.model import Model

        if not models_are_installed():
            download_models()

        self._model = Model(wakeword_models=[model], inference_framework="onnx")
        self._key = model
        self._threshold = threshold
        self._refractory_s = refractory_s
        self._last_trigger = 0.0

    def reset(self) -> None:
        """Limpia el estado interno. Se llama tras procesar una orden para que
        el audio de la orden anterior no contamine la siguiente deteccion."""
        self._model.reset()
        self._last_trigger = time.monotonic()

    def detected(self, block: np.ndarray) -> bool:
        if time.monotonic() - self._last_trigger < self._refractory_s:
            return False

        # openWakeWord espera int16, no float32.
        scores = self._model.predict((block * 32767).astype(np.int16))
        score = max(scores.values()) if scores else 0.0
        if score < self._threshold:
            return False

        log.info("wake word '%s' detectada (%.2f)", self._key, score)
        self._last_trigger = time.monotonic()
        return True
