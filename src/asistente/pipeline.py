"""Bucle principal: wake word -> grabar -> transcribir -> enrutar -> ejecutar -> hablar.

Es sincrono a proposito. Las etapas son estrictamente secuenciales -no puedes
enrutar lo que aun no se ha transcrito- asi que asyncio solo anadiria overhead y
puntos donde equivocarse. La unica concurrencia real que hace falta es que el
asistente pueda hablar mientras vuelve a escuchar, y de eso se encarga el hilo
interno de `Speaker`.

Cada turno registra la latencia de cada etapa. Ese log es la materia prima de
`scripts/benchmark.py` y de todo el tuning de la Fase 5.

VENTANA DE SEGUIMIENTO
----------------------
Tras ejecutar una orden el bucle no vuelve directo a esperar la palabra clave:
sigue escuchando unos segundos por si viene otra. Es lo que permite decir
"sube el volumen" / "mas" / "mas" en vez de repetir "Apolo" tres veces.

Dentro de esa ventana la palabra clave no protege nada, asi que solo se acepta
un subconjunto de ordenes -reversibles y resueltas sin LLM-. El razonamiento
completo esta en `FollowUpConfig`; aqui esta la implementacion.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from time import perf_counter

import numpy as np

from asistente.audio.capture import MicrophoneStream
from asistente.audio.denoise import snr_db
from asistente.audio.keyphrase import KeyphraseGate
from asistente.audio.recorder import DEFAULT_WAIT_S, Utterance, UtteranceRecorder
from asistente.audio.speaker import SpeakerGate
from asistente.audio.wakeword import WakeWordDetector
from asistente.config import FollowUpConfig, VadConfig
from asistente.router.engine import Router
from asistente.router.schema import RouteResult, Stage
from asistente.skills.registry import SkillRegistry
from asistente.stt.transcriber import Transcriber
from asistente.tts.speaker import Speaker

log = logging.getLogger(__name__)

#: Por debajo de esto no merece la pena volver a grabar dentro de la ventana:
#: no da tiempo ni a empezar una palabra.
_MIN_LISTEN_S = 0.5


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
        vad_config: VadConfig | None = None,
        speaker_gate: SpeakerGate | None = None,
        stop: threading.Event | None = None,
        follow_up: FollowUpConfig | None = None,
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
        self._vad_config = vad_config or VadConfig()
        self._speaker_gate = speaker_gate
        # Corriendo de fondo no hay Ctrl-C: la salida llega por este evento,
        # que pone el icono de la bandeja desde su propio hilo. Se comprueba
        # ENTRE frases y no dentro de `record()` para no partir una grabacion
        # a medias; el retardo maximo son los ~2 s que `record()` tarda en
        # rendirse cuando nadie habla.
        self._stop = stop
        self._follow_up = follow_up or FollowUpConfig()
        self._announce_follow_up(registry)
        self.metrics: list[TurnMetrics] = []

    def _announce_follow_up(self, registry: SkillRegistry) -> None:
        """Deja en el log que se aceptara sin palabra clave, y avisa de erratas.

        Un nombre mal escrito en `follow_up.tools` no da error: simplemente esa
        orden nunca se acepta dentro de la ventana. Es un fallo silencioso que
        llega al usuario como "a veces no me hace caso", que es de lo mas caro
        de diagnosticar. Mejor decirlo al arrancar.
        """
        if not self._follow_up.enabled:
            log.info("ventana de seguimiento desactivada")
            return

        if desconocidas := sorted(self._follow_up.tools - registry.names):
            log.warning(
                "follow_up.tools nombra skills que no existen: %s. Se ignoran, "
                "y esas ordenes nunca se aceptaran en la ventana de seguimiento",
                ", ".join(desconocidas),
            )
        log.info(
            "ventana de seguimiento: %.0f s tras cada orden, para %s",
            self._follow_up.window_s,
            ", ".join(sorted(self._follow_up.tools & registry.names)) or "nada",
        )

    @property
    def _stopping(self) -> bool:
        return self._stop is not None and self._stop.is_set()

    def _passes_noise_gates(self, utterance: Utterance) -> bool:
        """Descarta ruido ANTES de gastar el STT.

        El orden es por coste creciente, que es la misma idea que gobierna el
        router de tres etapas: primero un contador que ya tenemos, luego una FFT
        de unos milisegundos, y solo entonces los ~120 ms de Whisper.

        Todo se registra a DEBUG con sus numeros: cuando el asistente "no oye",
        arrancar con -v dice exactamente que puerta lo paro y por cuanto.
        """
        if utterance.speech_s < self._vad_config.min_speech_s:
            log.debug(
                "descartado: solo %.2f s de voz en %.2f s (minimo %.2f s)",
                utterance.speech_s,
                utterance.total_s,
                self._vad_config.min_speech_s,
            )
            return False

        if self._vad_config.min_snr_db > 0:
            snr = snr_db(utterance.audio)
            if snr < self._vad_config.min_snr_db:
                log.debug(
                    "descartado: SNR %.1f dB por debajo de %.1f dB",
                    snr,
                    self._vad_config.min_snr_db,
                )
                return False

        return True

    def _is_my_voice(self, audio: np.ndarray) -> bool:
        """Verificacion de locutor, si esta activada.

        Va antes del STT porque asi la voz de otra persona no llega a costar una
        transcripcion. El coseno se registra SIEMPRE, tambien cuando acepta: es
        el unico dato con el que ajustar el umbral sin adivinar.
        """
        if self._speaker_gate is None:
            return True

        verdict = self._speaker_gate.check(audio, self._mic.sample_rate)
        if not verdict.accepted:
            log.info("ignorado, no es tu voz: %s", verdict.reason)
            return False
        log.debug("locutor: %s", verdict.reason)
        return True

    def run_forever(self) -> None:
        if self._keyphrase is not None:
            self._run_transcript_mode()
        else:
            self._run_wakeword_mode()
        if self._stopping:
            log.info("bucle detenido a peticion")

    def _run_wakeword_mode(self) -> None:
        """Modelo dedicado siempre escuchando; solo transcribe tras activarse."""
        assert self._wake_word is not None
        blocks = self._mic.blocks()
        log.info("escuchando. Di la palabra clave para empezar.")
        for block in blocks:  # type: ignore[attr-defined]
            if self._stopping:
                break
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

        while not self._stopping:
            try:
                utterance = self._recorder.record(blocks, self._mic.preroll_audio())
                if utterance.empty:
                    continue
                if not self._passes_noise_gates(utterance):
                    continue
                if not self._is_my_voice(utterance.audio):
                    continue

                started = perf_counter()
                text = self._transcriber.transcribe(utterance.audio, self._mic.sample_rate)
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
                result = self._complete_turn(command, stt_s)
                self._follow_up_window(blocks, result)
            except Exception:
                log.exception("el turno fallo")

    def _handle_turn(self, blocks: object) -> None:
        utterance = self._recorder.record(blocks, self._mic.preroll_audio())
        if utterance.empty:
            return

        # Aqui NO se verifica el locutor: ya dijiste la palabra clave y el
        # asistente esta esperando tu orden. Rechazarla ahora dejaria el turno
        # colgado sin explicacion.
        if not self._passes_noise_gates(utterance):
            return

        started = perf_counter()
        text = self._transcriber.transcribe(utterance.audio, self._mic.sample_rate)
        stt_s = perf_counter() - started
        if not text:
            return

        result = self._complete_turn(text, stt_s)
        self._follow_up_window(blocks, result)

    def _complete_turn(self, text: str, stt_s: float) -> RouteResult:
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

        return result

    def _follow_up_window(self, blocks: object, result: RouteResult) -> None:
        """Sigue escuchando unos segundos mas, sin exigir la palabra clave.

        Solo se abre si el asistente HIZO algo. Si el turno no acabo en accion
        ni en respuesta, lo que llego al microfono probablemente no iba
        dirigido a el, y esa es la peor situacion para bajar la guardia.

        La cuenta se reinicia con cada orden aceptada: encadenar cinco ordenes
        no necesita una ventana de veinticinco segundos, necesita cinco de
        cinco. Y la cierra la primera frase que NO se acepta: si lo que suena
        delante del microfono ya no son ordenes, se vuelve a la palabra clave.
        """
        if not self._follow_up.enabled or self._stopping:
            return
        if result.tool_call is None and result.reply is None:
            return

        deadline = self._reopen_window()
        log.debug("ventana de seguimiento abierta (%.1f s)", self._follow_up.window_s)

        while not self._stopping:
            remaining = deadline - perf_counter()
            if remaining < _MIN_LISTEN_S:
                break

            # Trozos de como mucho `DEFAULT_WAIT_S` en vez de una sola espera
            # larga: asi lo que llega a Whisper no arrastra cuatro segundos de
            # silencio por delante, que es material de alucinacion.
            utterance = self._recorder.record(
                blocks, self._mic.preroll_audio(), wait_s=min(DEFAULT_WAIT_S, remaining)
            )
            if utterance.empty:
                continue
            if not self._passes_noise_gates(utterance):
                continue
            # Aqui SI se verifica el locutor, al reves que en `_handle_turn`:
            # alli acababas de decir la palabra clave, aqui no ha dicho nada
            # nadie. Es donde esta verificacion se gana el sitio.
            if not self._is_my_voice(utterance.audio):
                continue

            started = perf_counter()
            text = self._transcriber.transcribe(utterance.audio, self._mic.sample_rate)
            stt_s = perf_counter() - started
            if not text:
                continue

            if not self._follow_up_turn(text, stt_s):
                break
            deadline = self._reopen_window()

        log.debug("ventana de seguimiento cerrada")
        # Lo grabado mientras se ejecutaba lo ultimo ya no vale: sin esto, el
        # bucle principal empezaria mordiendo audio viejo.
        self._mic.discard_pending()

    def _reopen_window(self) -> float:
        """Espera a que el asistente acabe de hablar y devuelve el nuevo plazo.

        El plazo cuenta desde que CALLA, no desde que termina de ejecutar: si
        no, una confirmacion de dos segundos se comeria media ventana. Y el
        microfono se vacia despues de esperar, porque lo que se ha estado
        encolando mientras hablaba es su propia voz.
        """
        self._speaker.wait_until_done()
        if descartado := self._mic.discard_pending():
            log.debug("descartados %.1f s de audio propio", descartado)
        return perf_counter() + self._follow_up.window_s

    def _follow_up_turn(self, text: str, stt_s: float) -> bool:
        """Enruta una frase oida dentro de la ventana. `True` si se ejecuto."""
        started = perf_counter()
        result = self._router.route(text)
        route_s = perf_counter() - started

        if motivo := self._rejects_follow_up(result):
            # A INFO y con la frase entera: es el unico dato con el que decidir
            # si a `follow_up.tools` le falta algo, o si la ventana esta
            # recogiendo conversacion ajena y hay que acortarla.
            log.info("seguimiento ignorado (%s): %r", motivo, text)
            return False

        metrics = TurnMetrics(stt_s=stt_s, route_s=route_s, text=text)
        metrics.stage = result.stage.value
        started = perf_counter()
        self._act(result)
        metrics.execute_s = perf_counter() - started
        metrics.log()
        self.metrics.append(metrics)
        return True

    def _rejects_follow_up(self, result: RouteResult) -> str | None:
        """Motivo por el que NO se ejecuta, o `None` si se acepta.

        Devolver el motivo en vez de un booleano es lo que hace que el log
        sirva para algo: "ignorado" a secas no distingue si sobraba la frase o
        si falta un nombre en la allowlist.
        """
        if result.stage is Stage.LLM:
            # Si hizo falta el modelo, no era una orden simple. Ademas es la
            # etapa mas facil de disparar con charla: la clase `_fallback` del
            # catalogo manda ahi todo lo que no se parece a un comando.
            return "hubo que escalar al LLM"
        if result.tool_call is None:
            return "no es una orden"
        if result.tool_call.name not in self._follow_up.tools:
            return f"{result.tool_call.name} no esta en follow_up.tools"
        return None

    def _act(self, result: object) -> None:
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
