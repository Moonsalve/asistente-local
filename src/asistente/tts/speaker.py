"""Sintesis de voz con Piper y reproduccion por frases.

LA TECNICA QUE MAS APORTA A LA SENSACION DE INMEDIATEZ: no esperar a tener la
respuesta completa. El texto se parte por frases y cada una se sintetiza y
reproduce mientras el LLM todavia esta generando el resto. Asi el tiempo hasta
el primer audio deja de depender de lo larga que sea la respuesta.

Sintetizar y reproducir ocurren en un hilo aparte para que el bucle principal
pueda volver a escuchar mientras el asistente habla.
"""

from __future__ import annotations

import logging
import queue
import re
import threading
from collections.abc import Iterator
from pathlib import Path
from time import perf_counter

import numpy as np

log = logging.getLogger(__name__)

#: Corta tras el signo de puntuacion, conservandolo: Piper entona mejor con el.
_SENTENCE_END = re.compile(r"(?<=[.!?;:,])\s+")


def split_sentences(text: str, min_chars: int = 12) -> Iterator[str]:
    """Trocea texto en frases pronunciables.

    `min_chars` evita fragmentos ridiculos: sintetizar "Si," por separado suena
    entrecortado y cuesta mas que decirlo junto con lo que sigue.
    """
    buffer = ""
    for chunk in _SENTENCE_END.split(text):
        buffer = f"{buffer} {chunk}".strip() if buffer else chunk
        if len(buffer) >= min_chars:
            yield buffer
            buffer = ""
    if buffer:
        yield buffer


class Speaker:
    def __init__(self, voice_model: Path, speed: float = 1.0) -> None:
        from piper.voice import PiperVoice

        self._voice = PiperVoice.load(str(voice_model))
        self._speed = speed
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def warmup(self) -> None:
        """Sintetiza una frase sin reproducirla, para pagar la primera vez ahora.

        Medido: la primera sintesis cuesta ~555 ms y las siguientes ~65 ms. Un
        factor de 9. Sin esto, la primera respuesta hablada del asistente llega
        medio segundo tarde justo cuando mas se nota.
        """
        started = perf_counter()
        self._synthesize("Hola.")
        log.info("TTS calentado en %.0f ms", (perf_counter() - started) * 1000)

    def say(self, text: str) -> None:
        """Encola texto para hablar. No bloquea."""
        if text and text.strip():
            self._queue.put(text.strip())

    def say_stream(self, chunks: Iterator[str]) -> str:
        """Consume un stream de texto y va hablando por frases.

        Devuelve el texto completo acumulado, por si hay que registrarlo.
        """
        full = ""
        buffer = ""
        for chunk in chunks:
            full += chunk
            buffer += chunk
            *ready, buffer = _SENTENCE_END.split(buffer) or [buffer]
            for sentence in ready:
                self.say(sentence)
        if buffer.strip():
            self.say(buffer)
        return full

    def wait_until_done(self) -> None:
        self._queue.join()

    def close(self) -> None:
        self._queue.put(None)
        self._thread.join(timeout=5)

    def _worker(self) -> None:
        try:
            import sounddevice as sd
        except Exception:
            # Si el import falla y el hilo muriera aqui, `say()` seguiria
            # encolando y `wait_until_done()` colgaria para siempre. En vez de
            # eso el hilo sigue vivo vaciando la cola: el asistente se queda
            # mudo, pero las acciones se siguen ejecutando.
            log.exception("no hay salida de audio; el asistente funcionara sin voz")
            self._drain()
            return

        while True:
            item = self._queue.get()
            if item is None:
                self._queue.task_done()
                return
            try:
                audio = self._synthesize(item)
                sd.play(audio, self._voice.config.sample_rate)
                sd.wait()
            except Exception:
                # Que falle la voz no puede tumbar el asistente: la accion ya
                # se ejecuto, quedarse mudo es preferible a morir.
                log.exception("fallo la sintesis de %r", item)
            finally:
                self._queue.task_done()

    def _drain(self) -> None:
        """Consume la cola sin reproducir, para que nadie se quede esperando."""
        while True:
            item = self._queue.get()
            self._queue.task_done()
            if item is None:
                return

    def _synthesize(self, text: str) -> np.ndarray:
        chunks = [
            np.frombuffer(audio.audio_int16_bytes, dtype=np.int16)
            for audio in self._voice.synthesize(text)
        ]
        if not chunks:
            return np.zeros(0, dtype=np.int16)
        return np.concatenate(chunks)
