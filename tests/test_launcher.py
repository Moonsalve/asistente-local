"""Tests del lanzador de aplicaciones y del logging resistente."""

from __future__ import annotations

import io
import logging
import sys
from pathlib import Path

import pytest

from asistente.logsetup import RobustStreamHandler
from asistente.skills.launcher import resolve_executable


def test_resolves_an_absolute_path(tmp_path: Path) -> None:
    exe = tmp_path / "app.exe"
    exe.write_bytes(b"")
    assert resolve_executable(str(exe)) == exe


def test_resolves_from_path_env() -> None:
    """Lo que si esta en el PATH se encuentra por la via barata."""
    resolved = resolve_executable("python" if sys.platform == "win32" else "sh")
    assert resolved is not None
    assert resolved.exists()


def test_unknown_command_returns_none() -> None:
    """None y no una excepcion: 'no instalado' es una condicion ordinaria que
    la skill traduce a un mensaje para el usuario."""
    assert resolve_executable("programa-que-no-existe-12345") is None


def _capture(encoding: str) -> tuple[logging.Logger, io.TextIOWrapper]:
    stream = io.TextIOWrapper(io.BytesIO(), encoding=encoding, errors="strict")
    handler = RobustStreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger = logging.getLogger(f"test-{encoding}")
    logger.handlers = [handler]
    logger.propagate = False
    logger.setLevel(logging.INFO)
    return logger, stream


def _read(stream: io.TextIOWrapper, encoding: str) -> str:
    stream.flush()
    stream.buffer.seek(0)  # type: ignore[attr-defined]
    return stream.buffer.read().decode(encoding)  # type: ignore[attr-defined]


def test_ipa_symbols_do_not_lose_the_message() -> None:
    """El bug real: Piper registra fonemas IPA al sintetizar y la consola de
    Windows (cp1252) no puede codificarlos. El mensaje se perdia entero y solo
    quedaba un '--- Logging error ---' que no decia nada."""
    logger, stream = _capture("cp1252")
    logger.info("fonemas: ˈapolo θeta ə")
    salida = _read(stream, "cp1252")
    assert "fonemas:" in salida, "el mensaje debe sobrevivir aunque se pierdan simbolos"
    assert "apolo" in salida


def test_spanish_accents_survive_intact() -> None:
    """Degradar no puede significar romper lo que la consola SI sabe escribir."""
    logger, stream = _capture("cp1252")
    logger.info("canción, ñ, ¿qué?")
    assert "canción, ñ, ¿qué?" in _read(stream, "cp1252")


def test_utf8_console_keeps_everything() -> None:
    logger, stream = _capture("utf-8")
    logger.info("fonemas ˈapolo y emoji \U0001f3b5")
    salida = _read(stream, "utf-8")
    assert "ˈapolo" in salida
    assert "\U0001f3b5" in salida


@pytest.mark.parametrize("encoding", ["cp1252", "cp850", "ascii", "utf-8"])
def test_never_raises_whatever_the_console_encoding(encoding: str) -> None:
    logger, stream = _capture(encoding)
    logger.info("mezcla ˈ θ ñ é \U0001f3b5 texto")
    assert "texto" in _read(stream, encoding)
