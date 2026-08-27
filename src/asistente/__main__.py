"""Arranque del asistente.

EL ORDEN DE ARRANQUE IMPORTA. Todos los modelos se cargan y se calientan ANTES
de empezar a escuchar. La primera inferencia en CUDA compila kernels y cuesta
segundos; si se paga con el primer comando real, el asistente parece colgado
justo cuando mas atencion le prestas.

Modo `--text` para desarrollo: salta microfono, STT y TTS, y deja probar el
router y las skills escribiendo. Es lo unico que se puede ejecutar fuera de
Windows, y sirve para iterar el catalogo sin hablarle al PC.

ARRANQUE DE FONDO
-----------------
Lanzado con `pythonw.exe` no hay consola, y eso invalida tres supuestos del
codigo: no hay `stdout`, el directorio de trabajo es arbitrario y no hay
Ctrl-C. `runtime.py` explica por que; aqui se aplican en este orden, que no es
casual:

    1. rutas de los argumentos a absolutas   <- mientras el cwd sigue siendo el suyo
    2. chdir a la raiz del proyecto          <- arregla TODAS las relativas de golpe
    3. logging (a fichero siempre)           <- ya hay donde escribir los errores
    4. instancia unica                       <- antes de reservar 1.6 GB de VRAM
    5. icono de bandeja                      <- antes de los 30 s de carga, para
                                                que el doble clic de senales de vida
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
from pathlib import Path

from asistente.config import Config, Secrets
from asistente.discovery.auto import augment_config
from asistente.logsetup import configure as configure_logging
from asistente.preflight import run_checks
from asistente.router.catalog import load_catalog
from asistente.router.embedder import OnnxEmbedder
from asistente.router.engine import Router
from asistente.router.llm import OllamaFallback
from asistente.router.semantic import SemanticMatcher
from asistente.runtime import APP_NAME, anchor_working_directory, has_console, notify
from asistente.singleton import SingleInstance
from asistente.skills.factory import build_registry
from asistente.tray import Tray

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
    parser.add_argument(
        "--tray",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "icono en la bandeja del sistema. Por defecto se activa solo cuando no hay "
            "consola, que es cuando hace falta: con terminal ya tienes Ctrl-C"
        ),
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args()


def _keep_relative_if_missing(path: Path) -> Path:
    """Absolutiza la ruta solo si el fichero existe donde estamos ahora.

    Sutil pero necesario. `--config otro.yaml` se refiere al directorio desde el
    que lanzaste el comando, asi que hay que fijarlo ANTES del `chdir`. Pero el
    valor por defecto -`config.yaml`- se refiere al del PROYECTO, y absolutizarlo
    contra el cwd de un acceso directo lanzado desde el Inicio de Windows daria
    `C:\\Windows\\System32\\config.yaml`, que no existe.

    La existencia del fichero distingue los dos casos sin tener que adivinar la
    intencion: si esta aqui, era este; si no, que lo resuelva la raiz.
    """
    return path.resolve() if path.exists() else path


def main() -> int:
    """Punto de entrada. Ninguna excepcion puede salir de aqui sin dejar rastro.

    Con consola, una traza sin capturar se ve y ya esta. De fondo, el proceso
    desaparece sin mas: haces doble clic, el icono no llega a aparecer y no hay
    ni error ni pista. Por eso el cuerpo va envuelto: la traza al registro y una
    linea al usuario diciendole donde mirar.
    """
    args = _parse_args()

    config_path = _keep_relative_if_missing(args.config)
    commands_path = _keep_relative_if_missing(args.commands)
    root = anchor_working_directory()
    log_file = configure_logging(args.verbose)
    log.info("%s arrancando desde %s", APP_NAME, root)

    try:
        return _run(args, config_path, commands_path, log_file, root)
    except Exception:
        log.exception("el asistente termino por un error inesperado")
        _fatal("Error inesperado al arrancar.", log_file)
        return 1


def _fatal(message: str, log_file: Path | None) -> None:
    """Cuenta un fallo fatal por un canal que el usuario vaya a ver."""
    if log_file is not None:
        message = f"{message}\n\nDetalles en:\n{log_file}"
    log.error("%s", message.replace("\n", " "))
    if not has_console():
        notify(f"{APP_NAME} no pudo arrancar", message)


def _run(
    args: argparse.Namespace,
    config_path: Path,
    commands_path: Path,
    log_file: Path | None,
    root: Path,
) -> int:
    # El guardia de instancia unica va aqui, antes de cargar nada: rechazar el
    # arranque cuesta milisegundos, y hacerlo despues significaria haber pedido
    # ya 1.6 GB de VRAM que hay que devolver. Ver `singleton.py` para el
    # desastre concreto que evita.
    #
    # NO APLICA AL MODO TEXTO. Lo que el guardia protege es el microfono y la
    # VRAM de Whisper, y el modo texto no toca ninguno de los dos. Bloquearlo
    # ademas romperia el uso normal durante el desarrollo: iterar el catalogo
    # mientras el Apolo de verdad sigue escuchando de fondo.
    lock = SingleInstance()
    if not args.text and not lock.acquire():
        _fatal(f"{APP_NAME} ya se esta ejecutando. Mira el icono de la bandeja.", log_file)
        return 1

    tray = None
    stop = threading.Event()
    want_tray = args.tray if args.tray is not None else (not has_console() and not args.text)
    if want_tray:
        tray = Tray(on_quit=stop.set, log_file=log_file, project_dir=root)
        if tray.start():
            tray.set_title("arrancando…")
        else:
            tray = None

    try:
        return _start(args, config_path, commands_path, log_file, stop, tray)
    finally:
        if tray is not None:
            tray.stop()
        lock.release()


def _start(
    args: argparse.Namespace,
    config_path: Path,
    commands_path: Path,
    log_file: Path | None,
    stop: threading.Event,
    tray: Tray | None,
) -> int:
    config = Config.load(config_path)
    secrets = Secrets()

    # Antes de cargar nada pesado: Whisper tarda 25 s y el encoder otros 5, asi
    # que descubrir despues de todo eso que falta la voz de Piper es tiempo
    # tirado. En modo texto no hay TTS, asi que no aplica.
    if not args.text and not run_checks(config):
        _fatal("Falta algo para poder arrancar. Revisa el registro.", log_file)
        return 1

    # Autodescubrimiento: anade lo instalado a la allowlist. Con cache, asi que
    # solo la primera ejecucion (o una vez al dia) paga el coste de enumerar.
    config = augment_config(config)

    registry, spotify = build_registry(config, secrets)

    log.info("cargando encoder de embeddings...")
    embedder = OnnxEmbedder(
        repo_id=config.router.embedding_model,
        onnx_file=config.router.embedding_onnx_file,
        use_gpu=config.router.embedding_on_gpu,
    )
    embedder.warmup()

    catalog = load_catalog(commands_path, embedder)
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
        if spotify.connect() and not args.text:
            # Indexar la biblioteca son hasta 20 peticiones; se pagan aqui por
            # la misma razon que se calientan los modelos. En modo texto no
            # merece la pena: no hay STT que sesgar ni latencia que esconder.
            spotify.warmup()

    if args.text:
        return _run_text_mode(router, registry)

    if tray is not None:
        tray.set_title("escuchando")
    return _run_voice_mode(config, router, registry, spotify, stop=stop)


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


def _bias_stt_with_your_music(config: Config, transcriber: object, spotify: object | None) -> None:
    """Le pasa al STT los nombres de tu biblioteca de Spotify.

    Desactivado por defecto: con `large-v3-turbo` empeora la transcripcion. Ver
    la tabla en `SttConfig.hotwords_from_spotify`.

    Cualquier problema aqui degrada a "sin sesgo" y sigue: el asistente tiene
    que arrancar aunque Spotify no responda.
    """
    from asistente.skills.spotify import SpotifyClient
    from asistente.stt.transcriber import Transcriber

    if not config.stt.hotwords_from_spotify or not isinstance(spotify, SpotifyClient):
        return
    assert isinstance(transcriber, Transcriber)
    try:
        if terminos := spotify.hotwords(limit=config.stt.hotwords_limit):
            transcriber.set_hotwords(terminos)
    except Exception:
        log.exception("no se pudo sesgar el STT con tu biblioteca; se sigue sin ello")


def _build_speaker_gate(config: Config) -> object | None:
    """Verificacion de locutor, o None si no esta activada o no se puede cargar.

    Cualquier problema aqui degrada a "sin verificacion" y sigue. Es deliberado:
    un perfil corrupto o un modelo que no descarga no pueden dejar el asistente
    sordo — atender de mas es recuperable, no atender no se nota hasta que te
    has cansado de repetir la orden.
    """
    if not config.speaker.enabled:
        return None

    from asistente.audio.speaker import SpeakerEmbedder, SpeakerGate, download_model

    profile = Path(config.speaker.profile)
    if not profile.is_file():
        log.warning(
            "speaker.enabled=true pero no existe %s. Ejecuta: "
            "python scripts/enroll_voice.py    (se sigue sin verificar locutor)",
            profile,
        )
        return None

    try:
        embedder = SpeakerEmbedder(download_model())
        gate = SpeakerGate.from_profile(embedder, profile, config.speaker.threshold)
    except Exception:
        log.exception("no se pudo cargar la verificacion de locutor; se sigue sin ella")
        return None

    log.info("verificacion de locutor activa (umbral %.2f)", config.speaker.threshold)
    return gate


def _run_voice_mode(
    config: Config,
    router: object,
    registry: object,
    spotify: object | None = None,
    *,
    stop: threading.Event | None = None,
) -> int:
    from asistente.audio.capture import MicrophoneStream
    from asistente.audio.keyphrase import KeyphraseGate
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
    _bias_stt_with_your_music(config, transcriber, spotify)
    transcriber.warmup()

    speaker = Speaker(config.tts.voice_model, config.tts.speed)
    speaker.warmup()

    wake_word = keyphrase = None
    if config.wake_word.mode == "openwakeword":
        wake_word = WakeWordDetector(
            config.wake_word.model,
            config.wake_word.threshold,
            config.wake_word.refractory_s,
        )
        log.info("activacion por wake word: '%s'", config.wake_word.model)
    else:
        keyphrase = KeyphraseGate(config.wake_word.phrases, config.wake_word.phrase_threshold)
        log.info("activacion por transcripcion: %s", " / ".join(config.wake_word.phrases))

    recorder = UtteranceRecorder(SileroVad(config.audio.sample_rate), config.vad, config.audio.sample_rate)
    speaker_gate = _build_speaker_gate(config)

    with MicrophoneStream(
        sample_rate=config.audio.sample_rate,
        block_size=config.audio.block_size,
        device=config.audio.input_device,
        preroll_s=config.vad.preroll_s,
        gain=config.audio.gain,
    ) as mic:
        assistant = Assistant(
            mic,
            recorder,
            transcriber,
            router,
            registry,
            speaker,
            wake_word=wake_word,
            keyphrase=keyphrase,
            vad_config=config.vad,
            speaker_gate=speaker_gate,
            stop=stop,
        )
        try:
            assistant.run_forever()
        except KeyboardInterrupt:
            log.info("adios")
        finally:
            speaker.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
