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

Silero exige frames de un tamano EXACTO. Como los bloques del microfono no
tienen por que medirlo, aqui se reagrupan.

DOS VERSIONES DEL MODELO, DOS APIs
----------------------------------
Silero cambio de firma entre v4 y v5 y ninguna es compatible con la otra:

    v4: inputs (input, sr, h, c)   estados LSTM separados, (2, batch, 64)
        frames de 1536 muestras a 16 kHz
    v5: inputs (input, sr, state)  estado unificado, (2, batch, 128)
        frames de 512 muestras a 16 kHz

El que trae openWakeWord es el v4. Como el modelo se puede sustituir por otro
descargado aparte, aqui se detecta la version leyendo los nombres de las
entradas en vez de asumir una: pasar el feed equivocado da un ValueError
("Required inputs (['h', 'c']) are missing") que no dice nada de versiones.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

#: Muestras por frame a 16 kHz, impuesto por cada version del modelo.
FRAME_SAMPLES_V4 = 1536
FRAME_SAMPLES_V5 = 512


def _default_model_path() -> Path:
    """Silero sale del directorio de modelos de openWakeWord, que ya es una
    dependencia. Si no esta descargado, se descarga.

    OJO CON EL "VIENE INCLUIDO": el paquete pip de openWakeWord trae solo el
    codigo. Los .onnx se bajan aparte, y quien los baja es `WakeWordDetector`
    al construirse... que en modo `transcript` no se construye nunca. Este VAD
    en cambio se usa en LOS DOS modos, asi que no puede dar por hecho que otro
    componente haya pasado por aqui antes: se asegura el mismo.
    """
    from asistente.audio.wakeword import download_models, models_dir

    path = models_dir() / "silero_vad.onnx"
    if not path.is_file():
        log.warning("falta %s; se descarga ahora", path.name)
        download_models()
    return path


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

        input_names = {i.name for i in self._session.get_inputs()}
        self._is_v4 = "h" in input_names and "c" in input_names
        self.frame_samples = FRAME_SAMPLES_V4 if self._is_v4 else FRAME_SAMPLES_V5
        log.debug(
            "Silero VAD %s cargado (%s, frames de %d muestras)",
            "v4" if self._is_v4 else "v5",
            path.name,
            self.frame_samples,
        )

        self._pending = np.zeros(0, dtype=np.float32)
        self.reset()

    def reset(self) -> None:
        """Reinicia el estado recurrente. Obligatorio entre frases: el LSTM
        arrastra contexto y sin resetear la deteccion se degrada."""
        if self._is_v4:
            self._h = np.zeros((2, 1, 64), dtype=np.float32)
            self._c = np.zeros((2, 1, 64), dtype=np.float32)
        else:
            self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._pending = np.zeros(0, dtype=np.float32)

    def speech_probability(self, block: np.ndarray) -> float | None:
        """Probabilidad de voz del bloque, o None si aun no hay un frame completo.

        Con varios frames en el bloque devuelve el maximo: dentro de una ventana
        de 80 ms, que haya voz en cualquier parte significa que estas hablando.
        """
        self._pending = np.concatenate([self._pending, block.astype(np.float32)])
        if len(self._pending) < self.frame_samples:
            return None

        best: float | None = None
        sr = np.array(self._sample_rate, dtype=np.int64)
        while len(self._pending) >= self.frame_samples:
            frame = self._pending[: self.frame_samples].reshape(1, -1)
            self._pending = self._pending[self.frame_samples :]

            if self._is_v4:
                out, self._h, self._c = self._session.run(
                    None, {"input": frame, "sr": sr, "h": self._h, "c": self._c}
                )
            else:
                out, self._state = self._session.run(
                    None, {"input": frame, "sr": sr, "state": self._state}
                )

            prob = float(out[0][0])
            best = prob if best is None else max(best, prob)
        return best
