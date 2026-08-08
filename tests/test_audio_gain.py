"""Tests de la amplificacion de entrada y la normalizacion previa al STT.

Son dos mecanismos distintos que resuelven problemas distintos, y es facil
confundirlos:

  - `audio.gain` amplifica los bloques segun llegan, para que el VAD y la
    palabra clave detecten una senal floja.
  - `stt.normalize_audio` escala la frase ya grabada, para que Whisper la reciba
    al nivel con el que fue entrenado.

Aplicar solo uno deja el otro problema sin resolver.
"""

from __future__ import annotations

import numpy as np
import pytest

from asistente.audio.capture import MicrophoneStream
from asistente.stt.transcriber import normalize_peak


def _stream(gain: float) -> MicrophoneStream:
    return MicrophoneStream(gain=gain)


def test_gain_amplifies_a_quiet_signal() -> None:
    mic = _stream(gain=8.0)
    quiet = np.full(100, 0.01, dtype=np.float32)
    assert np.allclose(mic._amplify(quiet), 0.08)  # noqa: SLF001


def test_gain_clips_instead_of_overflowing() -> None:
    """Un float fuera de [-1, 1] produce artefactos que el VAD lee como habla:
    justo lo contrario de lo que se busca al subir la ganancia."""
    mic = _stream(gain=20.0)
    loud = np.full(100, 0.5, dtype=np.float32)
    out = mic._amplify(loud)  # noqa: SLF001
    assert out.max() <= 1.0
    assert out.min() >= -1.0


def test_clipping_is_counted_for_the_warning() -> None:
    """Saturar constantemente distorsiona la voz, que transcribe PEOR que una
    senal floja. Hay que poder avisar de ello."""
    mic = _stream(gain=20.0)
    for _ in range(10):
        mic._amplify(np.full(100, 0.5, dtype=np.float32))  # noqa: SLF001
    assert mic._clipped_blocks == 10  # noqa: SLF001


def test_gain_of_one_preserves_dtype() -> None:
    mic = _stream(gain=3.0)
    out = mic._amplify(np.full(100, 0.1, dtype=np.float32))  # noqa: SLF001
    assert out.dtype == np.float32


@pytest.mark.parametrize("level", [0.001, 0.01, 0.1, 0.5])
def test_normalize_brings_any_level_to_target(level: float) -> None:
    audio = np.full(1000, level, dtype=np.float32)
    assert float(np.max(np.abs(normalize_peak(audio, 0.95)))) == pytest.approx(0.95, rel=1e-5)


def test_normalize_leaves_silence_alone() -> None:
    """Escalar silencio amplificaria el ruido de fondo hasta que suene a voz."""
    silence = np.zeros(1000, dtype=np.float32)
    assert np.array_equal(normalize_peak(silence), silence)

    casi_silencio = np.full(1000, 1e-6, dtype=np.float32)
    assert np.array_equal(normalize_peak(casi_silencio), casi_silencio)


def test_normalize_preserves_signal_shape() -> None:
    """Normalizar no 'limpia' nada: solo escala. La relacion senal/ruido y la
    forma de onda quedan intactas."""
    rng = np.random.default_rng(0)
    audio = (rng.standard_normal(1000) * 0.01).astype(np.float32)
    out = normalize_peak(audio, 0.95)
    factor = out[0] / audio[0]
    assert np.allclose(out, audio * factor, rtol=1e-4)
