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


def check_wakeword_models() -> Problem | None:
    """Los modelos de openWakeWord se descargan aparte del paquete pip.

    No es fatal porque el detector los descarga solo la primera vez que se
    construye; esto solo sirve para avisar de que habra una descarga.
    """
    from asistente.audio.wakeword import models_are_installed

    if models_are_installed():
        return None
    return Problem(
        fatal=False,
        title="Faltan los modelos de openWakeWord (no vienen en el paquete pip)",
        fix="Se descargaran solos al arrancar (~30 MB). Manual: python scripts/download_models.py",
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
            check_wakeword_models(),
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
