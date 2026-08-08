"""Tests de configuracion.

Cubren errores que solo se manifiestan en runtime contra un servicio externo y
que por tanto no se detectan hasta que el asistente ya esta arrancando.
"""

from __future__ import annotations

from pathlib import Path

from asistente.config import Config

ROOT = Path(__file__).resolve().parents[1]


def test_keep_alive_is_numeric_not_string() -> None:
    """Ollama parsea las cadenas como duraciones con unidad.

    `keep_alive: "-1"` produce HTTP 400 `missing unit in duration "-1"`, y el
    fallo solo aparece en la primera consulta al LLM. Como numero se interpreta
    en segundos, y -1 significa "no descargar nunca de VRAM".
    """
    config = Config.load(ROOT / "config.yaml")
    assert isinstance(config.llm.keep_alive, int)
    assert config.llm.keep_alive == -1


def test_real_config_file_is_valid() -> None:
    """El config.yaml del repo tiene que validar: extra='forbid' en todos los
    modelos hace que una clave mal escrita reviente aqui y no al arrancar."""
    config = Config.load(ROOT / "config.yaml")
    assert config.apps, "la allowlist de apps no deberia estar vacia"
    assert config.sites, "el mapa de sitios no deberia estar vacio"
