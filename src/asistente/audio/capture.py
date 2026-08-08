"""Captura de microfono con buffer circular.

El callback de `sounddevice` corre en un hilo de tiempo real del driver de
audio: si se bloquea, el audio se corta. Por eso lo unico que hace es copiar el
bloque a una cola y volver. Todo el procesamiento (wake word, VAD, STT) ocurre
en el hilo principal leyendo de esa cola.

El buffer circular guarda los ultimos segundos de audio para el *preroll*: cuando
el VAD detecta habla ya te has comido la primera silaba, asi que se recupera de
aqui. Sin esto, "abre spotify" llega a Whisper como "bre spotify".
"""

from __future__ import annotations

import logging
import queue
from collections import deque
from types import TracebackType
from typing import Self

import numpy as np

log = logging.getLogger(__name__)


class MicrophoneStream:
    def __init__(
        self,
        sample_rate: int = 16_000,
        block_size: int = 1280,
        device: int | None = None,
        preroll_s: float = 0.3,
        gain: float = 1.0,
    ) -> None:
        self._sample_rate = sample_rate
        self._block_size = block_size
        self._device = device
        self._gain = gain
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=64)
        self._preroll = deque(maxlen=max(1, int(preroll_s * sample_rate / block_size)))
        self._stream: object | None = None
        self._clipped_blocks = 0
        self._total_blocks = 0

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def __enter__(self) -> Self:
        import sounddevice as sd

        self._stream = sd.InputStream(
            samplerate=self._sample_rate,
            blocksize=self._block_size,
            device=self._device,
            channels=1,
            dtype="float32",
            callback=self._on_audio,
        )
        self._stream.start()  # type: ignore[attr-defined]
        log.info("microfono abierto a %d Hz (bloques de %d)", self._sample_rate, self._block_size)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._stream is not None:
            self._stream.stop()  # type: ignore[attr-defined]
            self._stream.close()  # type: ignore[attr-defined]

    def _on_audio(self, indata: np.ndarray, frames: int, time_info: object, status: object) -> None:
        if status:
            log.warning("estado del stream de audio: %s", status)

        block = indata[:, 0].copy()
        if self._gain != 1.0:
            block = self._amplify(block)

        try:
            self._queue.put_nowait(block)
        except queue.Full:
            # Preferimos perder audio a bloquear el hilo del driver. Si esto
            # aparece en el log de forma sostenida, el consumidor va demasiado
            # lento y hay que revisar que se ejecuta entre bloques.
            log.warning("cola de audio llena; se descarta un bloque")

    def blocks(self) -> object:
        """Generador infinito de bloques de audio."""
        while True:
            block = self._queue.get()
            self._preroll.append(block)
            yield block

    def preroll_audio(self) -> np.ndarray:
        """Audio inmediatamente anterior al momento actual."""
        return np.concatenate(list(self._preroll)) if self._preroll else np.zeros(0, np.float32)

    def _amplify(self, block: np.ndarray) -> np.ndarray:
        """Aplica la ganancia recortando a [-1, 1].

        Se recorta en vez de dejar que desborde porque un float fuera de rango
        produce artefactos que el VAD interpreta como habla, justo lo contrario
        de lo que se busca.

        Se lleva la cuenta de cuanto se recorta: si pasa a menudo, la ganancia
        esta demasiado alta y estamos distorsionando la voz, que transcribe
        PEOR que una senal floja.
        """
        self._total_blocks += 1
        amplified = block * self._gain

        if np.max(np.abs(amplified)) > 1.0:
            self._clipped_blocks += 1
            # Avisar de forma periodica, no en cada bloque: son 12 por segundo.
            if self._clipped_blocks % 50 == 1:
                ratio = self._clipped_blocks / self._total_blocks
                log.warning(
                    "el audio satura con gain=%.1f (%.0f%% de los bloques). "
                    "Bajalo en config.yaml: la voz distorsionada transcribe peor "
                    "que la floja.",
                    self._gain,
                    ratio * 100,
                )
            amplified = np.clip(amplified, -1.0, 1.0)

        return amplified.astype(np.float32)
