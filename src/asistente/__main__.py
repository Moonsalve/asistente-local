"""Arranque del asistente.

EL ORDEN DE ARRANQUE IMPORTA. Todos los modelos se cargan y se calientan ANTES
de empezar a escuchar. La primera inferencia en CUDA compila kernels y cuesta
segundos; si se paga con el primer comando real, el asistente parece colgado
justo cuando mas atencion le prestas.

Modo `--text` para desarrollo: salta microfono, STT y TTS, y deja probar el
router y las skills escribiendo. Es lo unico que se puede ejecutar fuera de
Windows, y sirve para iterar el catalogo sin hablarle al PC.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from asistente.config import Config, Secrets
from asistente.preflight import run_checks
from asistente.router.catalog import load_catalog
from asistente.router.embedder import OnnxEmbedder
from asistente.router.engine import Router
from asistente.router.llm import OllamaFallback
from asistente.router.semantic import SemanticMatcher
from asistente.skills.factory import build_registry

log = logging.getLogger("asistente")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="asistente")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--commands", type=Path, default=Path("commands.yaml"))
    parser.add_argument(
        "--text",
        action="store_true",
        help="modo texto: escribe comandos en vez de hablarlos (sin microfono ni voz)",
    )
    parser.add_argument("--no-llm", action="store_true", help="desactiva la etapa 3")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # httpx registra en INFO cada peticion que hacen huggingface_hub y ollama.
    # En el arranque son decenas de lineas que tapan lo que si importa.
    for noisy in ("httpx", "httpcore", "huggingface_hub", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    config = Config.load(args.config)
    secrets = Secrets()

    # Antes de cargar nada pesado: Whisper tarda 25 s y el encoder otros 5, asi
    # que descubrir despues de todo eso que falta la voz de Piper es tiempo
    # tirado. En modo texto no hay TTS, asi que no aplica.
    if not args.text and not run_checks(config):
        log.error("arranque abortado: corrige lo marcado como error y vuelve a intentarlo")
        return 1

    registry, spotify = build_registry(config, secrets)

    log.info("cargando encoder de embeddings...")
    embedder = OnnxEmbedder(
        repo_id=config.router.embedding_model,
        onnx_file=config.router.embedding_onnx_file,
        use_gpu=config.router.embedding_on_gpu,
    )
    embedder.warmup()

    catalog = load_catalog(args.commands, embedder)
    registry.verify_catalog({
        spec.tool for spec in catalog.intents.values() if spec.tool is not None
    })
    log.info(
        "catalogo: %d intents, %d anclas semanticas, %d frases literales",
        len(catalog.intents),
        catalog.embeddings.shape[0],
        len(catalog.literal_index),
    )

    llm = None
    if not args.no_llm:
        log.info("conectando con Ollama (%s)...", config.llm.model)
        llm = OllamaFallback(config.llm, registry.describe())
        llm.warmup()

    router = Router(catalog, SemanticMatcher(catalog, embedder, config.router.threshold), llm)

    if config.spotify.enabled and spotify.available:
        log.info("conectando con Spotify...")
        spotify.connect()

    if args.text:
        return _run_text_mode(router, registry)

    return _run_voice_mode(config, router, registry)


def _run_text_mode(router: object, registry: object) -> int:
    """Bucle de teclado. Util para iterar el catalogo sin hablar."""
    from asistente.router.engine import Router
    from asistente.skills.registry import SkillRegistry

    assert isinstance(router, Router) and isinstance(registry, SkillRegistry)

    print("\nModo texto. Escribe un comando (Ctrl-C para salir).\n")
    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not text:
            continue

        result = router.route(text)
        print(f"  stage={result.stage.value} score={result.score:.3f} ({result.latency_s * 1000:.1f} ms)")
        if result.reply is not None:
            print(f"  habla: {result.reply.text}")
        elif result.tool_call is not None:
            print(f"  accion: {result.tool_call.name} {result.tool_call.args}")
            outcome = registry.dispatch(result.tool_call)
            print(f"  resultado: ok={outcome.ok} {outcome.speech or ''}")
        else:
            print("  sin accion")


def _run_voice_mode(config: Config, router: object, registry: object) -> int:
    from asistente.audio.capture import MicrophoneStream
    from asistente.audio.recorder import UtteranceRecorder
    from asistente.audio.vad import SileroVad
    from asistente.audio.wakeword import WakeWordDetector
    from asistente.pipeline import Assistant
    from asistente.router.engine import Router
    from asistente.skills.registry import SkillRegistry
    from asistente.stt.transcriber import Transcriber
    from asistente.tts.speaker import Speaker

    assert isinstance(router, Router) and isinstance(registry, SkillRegistry)

    log.info("cargando whisper (%s)...", config.stt.model)
    transcriber = Transcriber(config.stt)
    transcriber.warmup()

    speaker = Speaker(config.tts.voice_model, config.tts.speed)
    speaker.warmup()
    wake_word = WakeWordDetector(
        config.wake_word.model,
        config.wake_word.threshold,
        config.wake_word.refractory_s,
    )
    recorder = UtteranceRecorder(SileroVad(config.audio.sample_rate), config.vad, config.audio.sample_rate)

    with MicrophoneStream(
        sample_rate=config.audio.sample_rate,
        block_size=config.audio.block_size,
        device=config.audio.input_device,
        preroll_s=config.vad.preroll_s,
    ) as mic:
        assistant = Assistant(mic, wake_word, recorder, transcriber, router, registry, speaker)
        try:
            assistant.run_forever()
        except KeyboardInterrupt:
            log.info("adios")
        finally:
            speaker.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
