"""Tests de la frontera de seguridad.

El registro es el unico punto por el que se llega a ejecutar algo. Estos tests
comprueban que rechaza lo que debe rechazar, porque es exactamente lo que un LLM
de 3B produce cuando alucina: nombres de herramienta inventados y argumentos que
no existen.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from asistente.config import Config, Secrets
from asistente.router.schema import ToolCall
from asistente.skills.factory import build_registry
from asistente.skills.registry import SkillRegistry

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def registry() -> SkillRegistry:
    config = Config.load(ROOT / "config.yaml")
    reg, _ = build_registry(config, Secrets())
    return reg


def test_catalog_tools_all_have_skills(registry: SkillRegistry) -> None:
    """Un typo en commands.yaml se manifestaria como un comando que no hace nada."""
    raw = yaml.safe_load((ROOT / "commands.yaml").read_text(encoding="utf-8"))
    referenced = {
        spec["tool"] for spec in raw["intents"].values() if not spec.get("fallback", False)
    }
    registry.verify_catalog(referenced)


def test_unknown_tool_is_rejected(registry: SkillRegistry) -> None:
    """Nombre fuera de la allowlist: no hay resolucion dinamica que explotar."""
    result = registry.dispatch(ToolCall(name="system.shutdown", args={}))
    assert result.ok is False


def test_extra_args_are_rejected(registry: SkillRegistry) -> None:
    """`extra='forbid'` en cada modelo: el LLM no puede colar parametros.

    OJO al elegir la clave: `target` SI es un campo valido de volume.mute desde
    que el volumen distingue PC de Spotify. Este test necesita un nombre que de
    verdad no exista, o dejaria de comprobar nada.
    """
    result = registry.dispatch(ToolCall(name="volume.mute", args={"ruta": "/etc/passwd"}))
    assert result.ok is False


def test_free_text_target_never_escapes_the_enum(registry: SkillRegistry) -> None:
    """`target` es texto libre que puede venir del LLM, asi que la skill lo
    coacciona: cualquier cosa que no suene a musica cae en el sistema. Nunca se
    usa como ruta, comando ni nombre de proceso."""
    from asistente.skills.volume import VolumeTarget, resolve_target

    for hostile in ("/etc/passwd", "; rm -rf /", "../../windows", "", "lo que sea"):
        assert resolve_target(hostile) is VolumeTarget.SYSTEM


def test_wrong_arg_type_is_rejected(registry: SkillRegistry) -> None:
    result = registry.dispatch(ToolCall(name="volume.step", args={"delta": "muchisimo"}))
    assert result.ok is False


def test_out_of_range_arg_is_rejected(registry: SkillRegistry) -> None:
    result = registry.dispatch(ToolCall(name="volume.step", args={"delta": 9999}))
    assert result.ok is False


def test_close_rejects_app_outside_allowlist(registry: SkillRegistry) -> None:
    """No se mata un proceso que no este declarado: es una accion destructiva."""
    result = registry.dispatch(ToolCall(name="app.close", args={"app": "lsass"}))
    assert result.ok is False


def test_tool_name_pattern_blocks_malformed_names() -> None:
    """El propio ToolCall rechaza lo que no tenga forma `dominio.accion`."""
    for bad in ("../../etc/passwd", "SYSTEM.MUTE", "rm -rf", "sinpunto", ""):
        with pytest.raises(ValueError):
            ToolCall(name=bad, args={})


def test_wellformed_but_unregistered_name_still_rejected(registry: SkillRegistry) -> None:
    """`os.system` pasa el patron pero no esta registrado.

    Es la razon de que la allowlist sea la defensa real y el patron solo higiene:
    un nombre puede tener la forma correcta y no ser una skill.
    """
    assert ToolCall(name="os.system", args={}).name == "os.system"
    assert registry.dispatch(ToolCall(name="os.system", args={})).ok is False
