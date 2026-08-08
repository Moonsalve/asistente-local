"""Tests de las comprobaciones previas.

Existen porque los preflight fallan de la peor forma posible: en silencio.
Un check que devuelve None cuando deberia avisar no rompe ningun test a menos
que se compruebe explicitamente.
"""

from __future__ import annotations

from pathlib import Path

from asistente.config import Config
from asistente.preflight import check_tts_voice, describe_interpreter, run_checks

ROOT = Path(__file__).resolve().parents[1]


def _config_with_voice(path: Path) -> Config:
    config = Config.load(ROOT / "config.yaml")
    return config.model_copy(update={"tts": config.tts.model_copy(update={"voice_model": path})})


def test_missing_voice_is_fatal(tmp_path: Path) -> None:
    problem = check_tts_voice(_config_with_voice(tmp_path / "no-existe.onnx"))
    assert problem is not None
    assert problem.fatal is True


def test_voice_needs_both_files(tmp_path: Path) -> None:
    """El .onnx solo no basta: Piper necesita tambien su .json y falla con un
    error confuso que senala al .json aunque falte el otro."""
    onnx = tmp_path / "voz.onnx"
    onnx.write_bytes(b"fake")
    problem = check_tts_voice(_config_with_voice(onnx))
    assert problem is not None
    assert ".json" in problem.title


def test_complete_voice_passes(tmp_path: Path) -> None:
    onnx = tmp_path / "voz.onnx"
    onnx.write_bytes(b"fake")
    (tmp_path / "voz.onnx.json").write_text("{}")
    assert check_tts_voice(_config_with_voice(onnx)) is None


def test_run_checks_returns_false_when_fatal(tmp_path: Path) -> None:
    assert run_checks(_config_with_voice(tmp_path / "no-existe.onnx")) is False


def test_describe_interpreter_reports_venv_state() -> None:
    described = describe_interpreter()
    assert "venv" in described or "SISTEMA" in described
