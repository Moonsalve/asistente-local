"""Tests de la superposicion de `config.local.yaml`.

Existe para que los ajustes propios de una maquina (indice del microfono,
ganancia, rutas de apps) no vivan en un fichero versionado. Sin esto,
`config.yaml` choca en cada `git pull` con cambios que no son compartidos.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from asistente.config import Config, _deep_merge

BASE = """
audio:
  sample_rate: 16000
  block_size: 1280
  gain: 8.0
stt:
  device: cuda
  language: es
"""


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    (tmp_path / "config.yaml").write_text(BASE)
    return tmp_path


def test_without_local_file_uses_base(config_dir: Path) -> None:
    config = Config.load(config_dir / "config.yaml")
    assert config.audio.gain == 8.0
    assert config.stt.device == "cuda"


def test_local_overrides_only_the_keys_it_declares(config_dir: Path) -> None:
    """Lo esencial: cambiar `gain` no puede borrar el resto de `audio`."""
    (config_dir / "config.local.yaml").write_text("audio:\n  gain: 15.0\n")

    config = Config.load(config_dir / "config.yaml")
    assert config.audio.gain == 15.0
    assert config.audio.sample_rate == 16000, "las claves no declaradas se heredan"
    assert config.audio.block_size == 1280
    assert config.stt.device == "cuda", "las secciones no tocadas quedan intactas"


def test_local_can_override_several_sections(config_dir: Path) -> None:
    (config_dir / "config.local.yaml").write_text(
        "audio:\n  input_device: 3\nstt:\n  device: cpu\n  compute_type: int8\n"
    )
    config = Config.load(config_dir / "config.yaml")
    assert config.audio.input_device == 3
    assert config.audio.gain == 8.0
    assert config.stt.device == "cpu"
    assert config.stt.language == "es"


def test_empty_local_file_is_harmless(config_dir: Path) -> None:
    (config_dir / "config.local.yaml").write_text("")
    assert Config.load(config_dir / "config.yaml").audio.gain == 8.0


def test_deep_merge_recurses_into_nested_dicts() -> None:
    base = {"a": {"b": {"c": 1, "d": 2}}, "e": 3}
    assert _deep_merge(base, {"a": {"b": {"c": 9}}}) == {"a": {"b": {"c": 9, "d": 2}}, "e": 3}


def test_deep_merge_does_not_mutate_the_base() -> None:
    base = {"a": {"b": 1}}
    _deep_merge(base, {"a": {"b": 2}})
    assert base == {"a": {"b": 1}}
