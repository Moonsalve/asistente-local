"""Tests de las comprobaciones previas.

Existen porque los preflight fallan de la peor forma posible: en silencio.
Un check que devuelve None cuando deberia avisar no rompe ningun test a menos
que se compruebe explicitamente.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from asistente.config import Config
from asistente.preflight import (
    check_tts_voice,
    describe_interpreter,
    ensure_wakeword_models,
    run_checks,
)

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


# --------------------------------------------------------------------------
# los modelos de openWakeWord: el aviso que era falso
# --------------------------------------------------------------------------
#
# EL FALLO QUE MOTIVA ESTOS TESTS. Esto antes solo avisaba, y decia "se
# descargaran solos al arrancar". Solo era cierto en modo `openwakeword`, donde
# los baja `WakeWordDetector` al construirse. En modo `transcript` -el de por
# defecto, el unico que admite "Apolo"- ese detector no se construye nunca.
#
# Pero `silero_vad.onnx` esta en ese mismo directorio y lo necesita el
# endpointing en LOS DOS modos. El asistente cargaba Whisper (25 s) y Piper, y
# solo entonces moria con un NO_SUCHFILE de onnxruntime apuntando dentro de
# site-packages, que parece una instalacion corrupta y no lo es.


def test_missing_models_are_downloaded_not_just_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """La promesa estaba escrita; lo que faltaba era cumplirla."""
    descargas = []
    instalados = iter([False, True])
    monkeypatch.setattr(
        "asistente.audio.wakeword.models_are_installed", lambda: next(instalados)
    )
    monkeypatch.setattr(
        "asistente.audio.wakeword.download_models", lambda: descargas.append(1)
    )

    assert ensure_wakeword_models() is None
    assert descargas == [1]


def test_present_models_are_not_downloaded_again(monkeypatch: pytest.MonkeyPatch) -> None:
    """Idempotente: son ~30 MB y esto corre en cada arranque."""
    descargas = []
    monkeypatch.setattr("asistente.audio.wakeword.models_are_installed", lambda: True)
    monkeypatch.setattr(
        "asistente.audio.wakeword.download_models", lambda: descargas.append(1)
    )

    assert ensure_wakeword_models() is None
    assert descargas == []


def test_a_failed_download_is_fatal(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sin `silero_vad.onnx` no hay VAD, y sin VAD no arranca en ningun modo.
    Decirlo aqui cuesta un segundo; descubrirlo despues cuesta treinta."""

    def _boom() -> None:
        raise OSError("no hay red")

    monkeypatch.setattr("asistente.audio.wakeword.models_are_installed", lambda: False)
    monkeypatch.setattr("asistente.audio.wakeword.download_models", _boom)

    problem = ensure_wakeword_models()

    assert problem is not None
    assert problem.fatal is True
    assert "download_models" in problem.fix


def test_a_half_finished_download_is_fatal(monkeypatch: pytest.MonkeyPatch) -> None:
    """Que la descarga no lance no significa que dejara los cuatro ficheros.
    Sin esta segunda comprobacion volveriamos al NO_SUCHFILE de siempre."""
    monkeypatch.setattr("asistente.audio.wakeword.models_are_installed", lambda: False)
    monkeypatch.setattr("asistente.audio.wakeword.download_models", lambda: None)

    problem = ensure_wakeword_models()

    assert problem is not None
    assert problem.fatal is True


def test_the_vad_downloads_its_own_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """El VAD no puede dar por hecho que otro componente paso antes por aqui.

    Es la mitad del arreglo que importa: el preflight se salta en modo texto y
    no existe en los tests, pero `SileroVad` se usa en los dos modos de
    activacion.
    """
    from asistente.audio import vad

    modelo = tmp_path / "silero_vad.onnx"

    def _descargar() -> None:
        modelo.write_bytes(b"fake")

    monkeypatch.setattr("asistente.audio.wakeword.models_dir", lambda: tmp_path)
    monkeypatch.setattr("asistente.audio.wakeword.download_models", _descargar)

    assert vad._default_model_path() == modelo
    assert modelo.is_file()
