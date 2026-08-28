"""Comprobaciones previas al arranque.

POR QUE EXISTE
--------------
Cargar Whisper cuesta 25 s y el encoder de embeddings otros 5. Descubrir DESPUES
de todo eso que falta la voz de Piper -o que Ollama no tiene el modelo- es una
perdida de tiempo evitable: son cosas que se comprueban en milisegundos.

Todo lo que se pueda verificar barato se verifica primero, y se informa de TODOS
los problemas a la vez en vez de uno por ejecucion.

Las comprobaciones distinguen dos niveles:
  - ERROR: impide arrancar (falta la voz del TTS).
  - AVISO: degrada pero deja funcionar (sin Ollama no hay etapa 3, pero el
    router sigue resolviendo la inmensa mayoria de comandos).

UNA COMPROBACION QUE MIENTE ES PEOR QUE NO TENERLA. `ensure_wakeword_models`
descarga en vez de avisar precisamente por eso: avisaba de que los modelos se
descargarian solos al arrancar, cosa que no pasaba en el modo por defecto, y el
asistente moria 30 segundos despues con un error que apuntaba a otro sitio.
"""

from __future__ import annotations

import logging
import site
import sys
from dataclasses import dataclass
from pathlib import Path

from asistente.config import Config

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Problem:
    fatal: bool
    title: str
    fix: str


def _in_virtualenv() -> bool:
    return sys.prefix != getattr(sys, "base_prefix", sys.prefix)


def describe_interpreter() -> str:
    kind = "venv" if _in_virtualenv() else "PYTHON DEL SISTEMA"
    return f"{sys.executable} ({kind}, {sys.version.split()[0]})"


def check_interpreter() -> Problem | None:
    """Avisa si no se esta usando el entorno virtual del proyecto.

    Es la causa mas comun de "instale la dependencia y sigue sin encontrarla":
    `pip install` fue a un interprete y `python -m asistente` corre en otro.
    """
    if _in_virtualenv():
        return None
    return Problem(
        fatal=False,
        title=f"No estas en un entorno virtual: {sys.executable}",
        fix=(
            "Si instalaste las dependencias en .venv, este proceso no las ve.\n"
            "     Activalo primero:  .venv\\Scripts\\activate"
        ),
    )


def check_cuda_packages() -> Problem | None:
    """Comprueba que las DLL de CUDA estan instaladas en ESTE interprete."""
    roots = [Path(p) for p in site.getsitepackages()]
    roots.append(Path(sys.prefix) / "Lib" / "site-packages")
    if any((root / "nvidia" / "cublas" / "bin").is_dir() for root in roots):
        return None
    return Problem(
        fatal=False,
        title="Faltan las librerias CUDA (cuBLAS/cuDNN) en este interprete",
        fix=(
            'El STT ira en CPU (varias veces mas lento). Para usar la GPU:\n'
            '     pip install -e ".[gpu]"\n'
            "     ...con el MISMO python con el que arrancas el asistente."
        ),
    )


def check_tts_voice(config: Config) -> Problem | None:
    """La voz de Piper son DOS archivos: el .onnx y su .json de configuracion.

    Piper falla con un FileNotFoundError sobre el .json aunque falte el .onnx,
    lo que despista: por eso aqui se comprueban los dos por separado.
    """
    model = config.tts.voice_model
    missing = [p for p in (model, model.with_suffix(model.suffix + ".json")) if not p.exists()]
    if not missing:
        return None
    return Problem(
        fatal=True,
        title=f"Falta la voz de Piper: {', '.join(str(p) for p in missing)}",
        fix="Descargala con:  python scripts/download_voice.py",
    )


def ensure_wakeword_models() -> Problem | None:
    """Descarga los modelos de openWakeWord si faltan. Fatal si no lo consigue.

    ESTO ANTES SOLO AVISABA, Y EL AVISO ERA FALSO. Decia "se descargaran solos
    al arrancar", que solo es cierto en modo `openwakeword`: ahi los descarga
    `WakeWordDetector` al construirse. En modo `transcript` -el de por defecto,
    y el unico que admite "Apolo"- ese detector no se construye nunca, asi que
    no los descargaba nadie.

    Pero `silero_vad.onnx` vive en ese mismo directorio y lo necesita el
    endpointing en LOS DOS MODOS. Resultado: el asistente cargaba Whisper (25
    s) y Piper, y solo entonces reventaba con un NO_SUCHFILE de onnxruntime
    apuntando a un fichero dentro de site-packages, que parece una instalacion
    corrupta y no lo es.

    Se descarga aqui, en el segundo cero, en vez de avisar: la promesa ya
    estaba escrita, lo que faltaba era cumplirla. Son ~30 MB una sola vez.
    """
    from asistente.audio.wakeword import download_models, models_are_installed, models_dir

    if models_are_installed():
        return None

    try:
        download_models()
    except Exception as exc:
        return Problem(
            fatal=True,
            title=(
                "No se pudieron descargar los modelos de openWakeWord "
                f"({type(exc).__name__}: {exc})"
            ),
            fix=(
                "Sin `silero_vad.onnx` no hay deteccion de voz y el asistente no\n"
                "     arranca en ningun modo. Reintentalo con:\n"
                "     python scripts/download_models.py"
            ),
        )

    if models_are_installed():
        return None
    return Problem(
        fatal=True,
        title=f"La descarga de openWakeWord no dejo todos los modelos en {models_dir()}",
        fix=(
            "Quedo a medias. Borra ese directorio y reintenta:\n"
            "     python scripts/download_models.py"
        ),
    )


def check_ollama(config: Config) -> Problem | None:
    """Comprueba que Ollama responde y tiene descargado el modelo."""
    try:
        from ollama import Client

        # Timeout corto: esto es un ping, no una inferencia.
        installed = Client(host=config.llm.host, timeout=5.0).list()
    except Exception as exc:
        return Problem(
            fatal=False,
            title=f"Ollama no responde en {config.llm.host} ({type(exc).__name__})",
            fix=(
                "Sin el no hay etapa 3, pero el router resuelve igual la mayoria\n"
                "     de comandos. Arrancalo con:  ollama serve"
            ),
        )

    wanted = config.llm.model
    names = {str(m.get("model") or m.get("name") or "") for m in installed.get("models", [])}
    # Ollama normaliza "qwen2.5:3b" a "qwen2.5:3b-...:latest" segun version.
    if any(name == wanted or name.startswith(f"{wanted}:") for name in names):
        return None
    return Problem(
        fatal=False,
        title=f"Ollama no tiene el modelo '{wanted}'",
        fix=f"Descargalo con:  ollama pull {wanted}",
    )


def run_checks(config: Config) -> bool:
    """Ejecuta todo y reporta. Devuelve False si hay algun problema fatal."""
    log.info("interprete: %s", describe_interpreter())

    problems = [
        problem
        for problem in (
            check_interpreter(),
            check_tts_voice(config),
            ensure_wakeword_models(),
            check_cuda_packages(),
            check_ollama(config),
        )
        if problem is not None
    ]

    for problem in problems:
        if problem.fatal:
            log.error("%s", problem.title)
        else:
            log.warning("%s", problem.title)
        for line in problem.fix.splitlines():
            log.info("  -> %s", line)

    return not any(problem.fatal for problem in problems)
