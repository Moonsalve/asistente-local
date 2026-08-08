"""Bucle principal: wake word -> grabar -> transcribir -> enrutar -> ejecutar -> hablar.

Es sincrono a proposito. Las etapas son estrictamente secuenciales -no puedes
enrutar lo que aun no se ha transcrito- asi que asyncio solo anadiria overhead y
puntos donde equivocarse. La unica concurrencia real que hace falta es que el
asistente pueda hablar mientras vuelve a escuchar, y de eso se encarga el hilo
interno de `Speaker`.

Cada turno registra la latencia de cada etapa. Ese log es la materia prima de
`scripts/benchmark.py` y de todo el tuning de la Fase 5.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from time import perf_counter

from asistente.audio.capture import MicrophoneStream
from asistente.audio.keyphrase import KeyphraseGate
from asistente.audio.recorder import UtteranceRecorder
from asistente.audio.wakeword import WakeWordDetector
from asistente.router.engine import Router
from asistente.router.schema import Stage
from asistente.skills.registry import SkillRegistry
from asistente.stt.transcriber import Transcriber
from asistente.tts.speaker import Speaker

log = logging.getLogger(__name__)


@dataclass(slots=True)
class TurnMetrics:
    """Latencias de un turno. `stage` es la metrica que gobierna la media:
    si el porcentaje de turnos con stage=llm sube del 15%, al catalogo le
    faltan parafrasis."""

    stt_s: float = 0.0
    route_s: float = 0.0
    execute_s: float = 0.0
    stage: str = ""
    text: str = ""
    extra: dict[str, float] = field(default_factory=dict)

    @property
    def total_s(self) -> float:
        return self.stt_s + self.route_s + self.execute_s

    def log(self) -> None:
        log.info(
            "turno: %r | stage=%s | stt=%.3fs route=%.3fs exec=%.3fs total=%.3fs",
            self.text,
            self.stage,
            self.stt_s,
            self.route_s,
            self.execute_s,
            self.total_s,
        )


class Assistant:
    def __init__(
        self,
        mic: MicrophoneStream,
        recorder: UtteranceRecorder,
        transcriber: Transcriber,
        router: Router,
        registry: SkillRegistry,
        speaker: Speaker,
        wake_word: WakeWordDetector | None = None,
        keyphrase: KeyphraseGate | None = None,
    ) -> None:
        if (wake_word is None) == (keyphrase is None):
            raise ValueError("hay que dar exactamente uno: wake_word o keyphrase")
        self._mic = mic
        self._wake_word = wake_word
        self._keyphrase = keyphrase
        self._recorder = recorder
        self._transcriber = transcriber
        self._router = router
        self._registry = registry
        self._speaker = speaker
        self.metrics: list[TurnMetrics] = []

    def run_forever(self) -> None:
        if self._keyphrase is not None:
            self._run_transcript_mode()
        else:
            self._run_wakeword_mode()

    def _run_wakeword_mode(self) -> None:
        """Modelo dedicado siempre escuchando; solo transcribe tras activarse."""
        assert self._wake_word is not None
        blocks = self._mic.blocks()
        log.info("escuchando. Di la palabra clave para empezar.")
        for block in blocks:  # type: ignore[attr-defined]
            if not self._wake_word.detected(block):
                continue
            try:
                self._handle_turn(blocks)
            except Exception:
                # Un turno que revienta no puede matar el bucle: el usuario
                # sigue ahi y el asistente tiene que seguir escuchando.
                log.exception("el turno fallo")
            finally:
                self._wake_word.reset()

    def _run_transcript_mode(self) -> None:
        """Transcribe cada frase y actua solo si empieza por la palabra clave.

        Cuando dices la clave y la orden del tiron ("Apolo, pon musica") basta
        UNA transcripcion: la orden ya viene en el mismo texto. Solo si dices la
        clave a secas se graba una segunda vez para escuchar la orden.
        """
        assert self._keyphrase is not None
        blocks = self._mic.blocks()
        log.info("escuchando. Di la palabra clave para empezar.")

        while True:
            try:
                audio = self._recorder.record(blocks, self._mic.preroll_audio())
                if audio.size == 0:
                    continue

                started = perf_counter()
                text = self._transcriber.transcribe(audio, self._mic.sample_rate)
                stt_s = perf_counter() - started
                if not text:
                    continue

                command = self._keyphrase.match(text)
                if command is None:
                    # No iba dirigido al asistente. A DEBUG y no a INFO: en modo
                    # transcripcion esto pasa con cualquier conversacion ajena.
                    # Si SI le hablabas y salio por aqui, arranca con -v: veras
                    # como transcribio tu palabra clave y podras anadir esa
                    # forma a `wake_word.phrases`.
                    log.debug("ignorado (sin palabra clave): %r", text)
                    continue

                if not command:
                    # Dijo solo la clave: escuchar ahora la orden.
                    log.info("palabra clave detectada; te escucho")
                    self._handle_turn(blocks)
                    continue

                log.info("palabra clave + orden en una frase: %r", command)
                self._complete_turn(command, stt_s)
            except Exception:
                log.exception("el turno fallo")

    def _handle_turn(self, blocks: object) -> None:
        audio = self._recorder.record(blocks, self._mic.preroll_audio())
        if audio.size == 0:
            return

        started = perf_counter()
        text = self._transcriber.transcribe(audio, self._mic.sample_rate)
        stt_s = perf_counter() - started
        if not text:
            return

        self._complete_turn(text, stt_s)

    def _complete_turn(self, text: str, stt_s: float) -> None:
        """Enruta y ejecuta un texto ya transcrito, midiendo cada etapa."""
        metrics = TurnMetrics(stt_s=stt_s, text=text)

        started = perf_counter()
        result = self._router.route(text)
        metrics.route_s = perf_counter() - started
        metrics.stage = result.stage.value

        started = perf_counter()
        self._act(result)
        metrics.execute_s = perf_counter() - started

        metrics.log()
        self.metrics.append(metrics)

        if result.stage is Stage.LLM:
            # A INFO y bien visible: cada frase que llega aqui es una parafrasis
            # que le falta al catalogo. Es la unica forma practica de saber que
            # anadir, porque nadie recuerda despues como lo dijo exactamente.
            log.info(
                "AL CATALOGO LE FALTA ESTA FRASE: %r  "
                "-> anadela a `examples` del intent correspondiente en commands.yaml",
                text,
            )

    def _act(self, result: object) -> None:
        from asistente.router.schema import RouteResult

        assert isinstance(result, RouteResult)

        if result.reply is not None:
            self._speaker.say(result.reply.text)
            return

        if result.tool_call is None:
            # Llego al LLM y ni siquiera el supo que hacer. Decirlo es mejor que
            # el silencio: si no, no sabes si te oyo.
            if result.stage is Stage.LLM:
                self._speaker.say("No entendí qué querías.")
            return

        outcome = self._registry.dispatch(result.tool_call)
        if outcome.speech:
            self._speaker.say(outcome.speech)
