"""Tests del router.

El valor de este archivo esta en `CORPUS`: frases que NO aparecen en
`commands.yaml`. Probar el router con las frases del propio catalogo no
demuestra nada (la etapa literal las resolveria trivialmente); lo que hay que
demostrar es que reconoce formas que nunca se escribieron.

Este corpus tambien es la red de seguridad al anadir intents: uno nuevo puede
robarle frases a uno existente, y aqui se ve inmediatamente.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from asistente.router.catalog import load_catalog
from asistente.router.embedder import OnnxEmbedder
from asistente.router.engine import Router
from asistente.router.schema import Stage
from asistente.router.semantic import SemanticMatcher

CATALOG_PATH = Path(__file__).resolve().parents[1] / "commands.yaml"

# (frase, intent esperado, slots esperados o None si no aplica)
CORPUS: list[tuple[str, str, dict[str, str] | None]] = [
    # --- media.next: el caso que motivo toda la etapa semantica
    ("pásala", "media.next", None),
    ("esta no me gusta", "media.next", None),
    ("cámbiame la canción", "media.next", None),
    ("ponme la que sigue", "media.next", None),
    ("no quiero escuchar esta", "media.next", None),
    ("brinca esta canción", "media.next", None),
    ("dale a la siguiente", "media.next", None),
    ("siguiente por favor", "media.next", None),
    # --- media.previous
    ("regrésala", "media.previous", None),
    ("vuelve a la de antes", "media.previous", None),
    ("pon otra vez la canción pasada", "media.previous", None),
    ("échale para atrás", "media.previous", None),
    # --- media.play_pause
    ("párale a la música", "media.play_pause", None),
    ("detén eso", "media.play_pause", None),
    ("ponle pausa", "media.play_pause", None),
    ("sigue reproduciendo", "media.play_pause", None),
    ("dale continuar", "media.play_pause", None),
    # --- volumen
    ("súbele", "volume.up", None),
    ("no se escucha nada", "volume.up", None),
    ("ponlo más fuerte", "volume.up", None),
    ("bájale tantito", "volume.down", None),
    ("está muy fuerte", "volume.down", None),
    ("ponlo más suave", "volume.down", None),
    ("quita el audio", "volume.mute", None),
    ("sin sonido", "volume.mute", None),
    ("volumen al 40", "volume.set", {"level": "40"}),
    ("pon el volumen en 75", "volume.set", {"level": "75"}),
    # --- spotify.play
    ("reproduce la playlist de estudio", "spotify.play", {"query": "estudio"}),
    ("pon música de los 80", "spotify.play", {"query": "los 80"}),
    ("quiero escuchar a shakira", "spotify.play", {"query": "shakira"}),
    ("ponme el álbum de pink floyd", "spotify.play", {"query": "pink floyd"}),
    # --- open.target: apps y webs comparten intent a proposito
    ("ábreme el chrome", "open.target", {"target": "chrome"}),
    ("inicia el spotify", "open.target", {"target": "spotify"}),
    ("corre la calculadora", "open.target", {"target": "calculadora"}),
    ("lanza el word", "open.target", {"target": "word"}),
    ("entra a youtube", "open.target", {"target": "youtube"}),
    ("llévame a netflix", "open.target", {"target": "netflix"}),
    ("visita wikipedia", "open.target", {"target": "wikipedia"}),
    # --- app.close
    ("cierra el chrome", "app.close", {"app": "chrome"}),
    ("mata el discord", "app.close", {"app": "discord"}),
    ("termina el spotify", "app.close", {"app": "spotify"}),
    # --- web.search
    ("búscame el precio del dólar", "web.search", {"query": "el precio del dolar"}),
    ("googlea restaurantes cerca", "web.search", {"query": "restaurantes cerca"}),
    ("busca en internet noticias de hoy", "web.search", {"query": "noticias de hoy"}),
    ("investiga quién ganó el mundial", "web.search", {"query": "quien gano el mundial"}),
    # --- system.time
    ("qué hora tienes", "system.time", None),
    ("dime qué hora es", "system.time", None),
]


@pytest.fixture(scope="session")
def router() -> Router:
    embedder = OnnxEmbedder()
    catalog = load_catalog(CATALOG_PATH, embedder)
    return Router(catalog, SemanticMatcher(catalog, embedder, threshold=0.25), llm=None)


@pytest.mark.parametrize(("phrase", "expected_intent", "expected_slots"), CORPUS)
def test_corpus_routes_without_llm(
    router: Router,
    phrase: str,
    expected_intent: str,
    expected_slots: dict[str, str] | None,
) -> None:
    """Cada frase debe resolver a la tool correcta sin escalar al LLM."""
    result = router.route(phrase)
    catalog = router._catalog  # noqa: SLF001 - acceso deliberado en test
    expected_tool = catalog.intents[expected_intent].tool

    assert result.stage is not Stage.LLM, f"{phrase!r} escalo al LLM (score {result.score:.3f})"
    assert result.tool_call is not None
    assert result.tool_call.name == expected_tool, (
        f"{phrase!r} -> {result.tool_call.name}, se esperaba {expected_tool} "
        f"(score {result.score:.3f})"
    )
    if expected_slots is not None:
        for key, value in expected_slots.items():
            assert result.tool_call.args.get(key) == value, (
                f"{phrase!r} slot {key}={result.tool_call.args.get(key)!r}, "
                f"se esperaba {value!r}"
            )


def test_literal_stage_is_used_for_catalog_phrases(router: Router) -> None:
    """Las frases declaradas como literales no deben pagar el encoder."""
    assert router.route("siguiente canción").stage is Stage.LITERAL
    assert router.route("Sube el volumen.").stage is Stage.LITERAL


def test_fixed_args_are_applied(router: Router) -> None:
    """volume.up y volume.down comparten skill y se diferencian por fixed_args."""
    up = router.route("súbele al volumen")
    down = router.route("bájale al volumen")
    assert up.tool_call is not None and up.tool_call.args["delta"] == 10
    assert down.tool_call is not None and down.tool_call.args["delta"] == -10


def test_unknown_phrase_escalates_to_llm(router: Router) -> None:
    """Lo que no esta en el catalogo debe caer a la etapa 3, no inventar un intent."""
    result = router.route("explícame la teoría de la relatividad en dos frases")
    assert result.stage is Stage.LLM
    assert result.tool_call is None  # llm=None en el fixture
