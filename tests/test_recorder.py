"""Tests del endpointing.

`silence_s` es el 40-50% de la latencia percibida del asistente, asi que la
logica que decide cuando has terminado de hablar merece pruebas exactas.

Se usa un VAD guionizado en vez del real: aqui se prueba la MAQUINA DE ESTADOS
del recorder, no el juicio acustico de Silero. Mezclar las dos cosas daria un
test que falla por razones equivocadas (un tono sintetico no es voz, y Silero
hace bien en decir que no lo es).
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest

from asistente.audio.recorder import Utterance, UtteranceRecorder
from asistente.config import VadConfig

SAMPLE_RATE = 16_000
BLOCK = 1280  # 80 ms
BLOCK_S = BLOCK / SAMPLE_RATE


class ScriptedVad:
    """Devuelve probabilidades prefijadas, una por bloque.

    `None` simula el caso real en que aun no hay un frame completo para el
    modelo: los bloques del microfono (80 ms) y los frames de Silero v4 (96 ms)
    no miden lo mismo, asi que uno de cada seis bloques no produce lectura.
    """

    def __init__(self, script: list[float | None]) -> None:
        self._script = list(script)
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1

    def speech_probability(self, block: np.ndarray) -> float | None:
        return self._script.pop(0) if self._script else 0.0


def _blocks(count: int) -> Iterator[np.ndarray]:
    for _ in range(count):
        yield np.full(BLOCK, 0.1, dtype=np.float32)


def _record(script: list[float | None], **overrides: float) -> np.ndarray:
    """Devuelve solo el audio: la mayoria de estos tests miran la maquina de
    estados, no la contabilidad de voz. Para esa esta `_grabar`."""
    return _grabar(script, **overrides).audio


def _grabar(script: list[float | None], **overrides: float) -> Utterance:
    config = VadConfig(silence_s=0.32, speech_threshold=0.5, **overrides)  # type: ignore[arg-type]
    recorder = UtteranceRecorder(ScriptedVad(script), config, SAMPLE_RATE)  # type: ignore[arg-type]
    return recorder.record(_blocks(200), np.zeros(0, dtype=np.float32))


def test_stops_after_configured_silence() -> None:
    """4 bloques de voz y luego silencio: debe cortar a los 0.32 s (4 bloques)."""
    audio = _record([0.9] * 4 + [0.0] * 50)
    # 4 de voz + 4 de silencio = 8 bloques
    assert audio.size == 8 * BLOCK


def test_none_readings_still_count_as_silence() -> None:
    """El bug que corrigio este test: cuando el VAD devolvia None, el `continue`
    se saltaba la cuenta del silencio, asi que el corte llegaba tarde."""
    # Voz, luego alternancia None/silencio como en la realidad.
    audio = _record([0.9] * 4 + [None, 0.0] * 25)
    assert audio.size == 8 * BLOCK, "los bloques sin lectura deben contar como silencio"


def test_speech_resets_the_silence_counter() -> None:
    """Una pausa corta a mitad de frase no debe cortar."""
    # 2 voz, 2 silencio (0.16 s < 0.32 s), 2 voz, luego silencio hasta cortar.
    audio = _record([0.9, 0.9, 0.0, 0.0, 0.9, 0.9] + [0.0] * 50)
    assert audio.size == 10 * BLOCK


def test_false_wakeword_returns_empty() -> None:
    """La wake word disparo pero nadie hablo: no se transcribe nada."""
    assert _record([0.0] * 100).size == 0


def test_max_duration_cuts_the_recording() -> None:
    """Un microfono ruidoso no puede dejar el pipeline grabando indefinidamente."""
    audio = _record([0.9] * 200, max_utterance_s=1.0)
    assert audio.size == pytest.approx(1.0 * SAMPLE_RATE, abs=BLOCK)


def test_preroll_is_prepended() -> None:
    """Sin preroll, Whisper recibe la primera silaba truncada."""
    config = VadConfig(silence_s=0.32, speech_threshold=0.5)
    recorder = UtteranceRecorder(ScriptedVad([0.9] * 2 + [0.0] * 50), config, SAMPLE_RATE)  # type: ignore[arg-type]
    preroll = np.full(4800, 0.5, dtype=np.float32)  # 0.3 s
    audio = recorder.record(_blocks(200), preroll).audio
    assert audio.size == preroll.size + 6 * BLOCK
    assert np.allclose(audio[: preroll.size], 0.5)


def test_vad_state_is_reset_between_utterances() -> None:
    """El LSTM de Silero arrastra contexto; sin reset la deteccion se degrada."""
    vad = ScriptedVad([0.9] * 2 + [0.0] * 50)
    recorder = UtteranceRecorder(vad, VadConfig(), SAMPLE_RATE)  # type: ignore[arg-type]
    recorder.record(_blocks(60), np.zeros(0, dtype=np.float32))
    assert vad.resets == 1


def test_speech_time_is_counted_separately_from_duration() -> None:
    """La puerta de ruido necesita saber CUANTA voz hubo, no cuanto duro.

    Un ventilador que roza el umbral produce grabaciones largas con casi nada de
    voz dentro; sin este contador no habria forma barata de distinguirlas.
    """
    grabacion = _grabar([0.9] * 5 + [0.0] * 50)
    assert grabacion.speech_s == pytest.approx(5 * BLOCK_S)
    # 5 de voz + 4 de silencio hasta cortar
    assert grabacion.total_s == pytest.approx(9 * BLOCK_S)


def test_intermittent_noise_records_long_but_counts_little_speech() -> None:
    """El caso que motiva la puerta: picos sueltos que superan el umbral.

    La grabacion dura, pero la voz acumulada es minima y `min_speech_s` la
    descarta antes de gastar Whisper.
    """
    grabacion = _grabar([0.9] + [0.0, 0.0, 0.0, 0.9] * 6 + [0.0] * 50)
    assert grabacion.speech_s < 0.6
    assert grabacion.total_s > grabacion.speech_s * 2


def test_filtro_de_alucinaciones_de_whisper() -> None:
    """Whisper no falla al transcribir ruido: INVENTA texto de los subtitulos
    con los que se entreno, y lo emite con toda confianza. El repertorio en
    espanol es corto, asi que una lista cerrada lo quita sin riesgo de tragarse
    un comando: nadie le dice "suscribete al canal" a un asistente."""
    from asistente.stt.transcriber import is_hallucination

    inventadas = [
        "Subtítulos realizados por la comunidad de Amara.org",
        "subtitulos realizados por la comunidad de amara.org.",
        "¡Gracias por ver el vídeo!",
        "Suscríbete al canal",
        "Más información en www.alimmenta.com",
    ]
    for texto in inventadas:
        assert is_hallucination(texto), texto

    reales = [
        "apolo sube el volumen",
        "pon música de los ochenta",
        "gracias",              # solo, no es la alucinacion
        "busca vídeos de gatos",
        "",
    ]
    for texto in reales:
        assert not is_hallucination(texto), texto
