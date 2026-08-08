"""Deteccion de voz y endpointing (saber cuando has terminado de hablar).

ESTE MODULO CONTIENE EL PARAMETRO MAS CARO DEL SISTEMA. El silencio que hay que
esperar para dar la frase por terminada es el 40-50% de la latencia total
percibida: mas que el STT y el router juntos. Todo lo demas se optimiza en
milisegundos; aqui se juegan cientos.

    silence_s = 0.35  ->  punto de partida seguro
    silence_s = 0.25  ->  se nota mucho, rara vez corta comandos cortos

Se usa Silero VAD (ONNX, ~1 MB, <1 ms por frame) en vez de un umbral de energia
porque este ultimo confunde el ruido de fondo con voz y, en una habitacion con
musica sonando -que es precisamente nuestro caso de uso-, nunca detecta silencio.

Silero exige frames de EXACTAMENTE 512 muestras a 16 kHz. Como los bloques del
microfono no tienen por que medir eso, aqui se reagrupan.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

#: Impuesto por el modelo Silero a 16 kHz. No es configurable.
FRAME_SAMPLES = 512


def _default_model_path() -> Path:
    """Silero viene incluido en openWakeWord, que ya es una dependencia."""
    import openwakeword

    return Path(openwakeword.__file__).parent / "resources" / "models" / "silero_vad.onnx"


class SileroVad:
    def __init__(self, sample_rate: int = 16_000, model_path: Path | None = None) -> None:
        import onnxruntime as ort

        path = model_path or _default_model_path()
        opts = ort.SessionOptions()
        # Un solo hilo: el modelo es diminuto y el paralelismo solo anade
        # contencion con el hilo de audio.
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self._session = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        self._sample_rate = sample_rate
        self._pending = np.zeros(0, dtype=np.float32)
        self.reset()

    def reset(self) -> None:
        """Reinicia el estado recurrente. Obligatorio entre frases: el LSTM
        arrastra contexto y sin resetear la deteccion se degrada."""
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._pending = np.zeros(0, dtype=np.float32)

    def speech_probability(self, block: np.ndarray) -> float | None:
        """Probabilidad de voz del bloque, o None si aun no hay un frame completo.

        Con varios frames en el bloque devuelve el maximo: dentro de una ventana
        de 80 ms, que haya voz en cualquier parte significa que estas hablando.
        """
        self._pending = np.concatenate([self._pending, block.astype(np.float32)])
        if len(self._pending) < FRAME_SAMPLES:
            return None

        best: float | None = None
        while len(self._pending) >= FRAME_SAMPLES:
            frame = self._pending[:FRAME_SAMPLES]
            self._pending = self._pending[FRAME_SAMPLES:]
            out, self._state = self._session.run(
                None,
                {
                    "input": frame.reshape(1, -1),
                    "state": self._state,
                    "sr": np.array(self._sample_rate, dtype=np.int64),
                },
            )
            prob = float(out[0][0])
            best = prob if best is None else max(best, prob)
        return best
