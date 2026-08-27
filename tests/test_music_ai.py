"""Tests de la correccion de titulos con LLM.

DOS COSAS DISTINTAS SE PRUEBAN AQUI, y la primera importa mas que la segunda:

1. **LA CASCADA.** Que el LLM no se llame cuando no hace falta. Es la mitad del
   diseno que se rompe sin que nadie se entere: si un dia el resolvedor pasa a
   invocarse en cada peticion de musica, todo sigue funcionando —solo que cada
   "pon musica" cuesta un segundo mas. No hay sintoma, solo lentitud.

2. **LA VEROSIMILITUD.** Que una invencion del modelo no llegue a sonar. Los
   casos son transcripciones reales de Whisper y las puntuaciones estan medidas,
   no supuestas.

No se habla con Ollama: se sustituye el modulo entero por un doble. En la Mac de
desarrollo `ollama` ni siquiera esta instalado, que es justo el escenario que
`build_resolver` tiene que sobrevivir.
"""

from __future__ import annotations

import json
import sys
import types
from typing import Any

import pytest

from asistente.config import Config, LlmConfig, Secrets
from asistente.skills.music_ai import (
    PLAUSIBLE_THRESHOLD,
    MusicQuery,
    MusicResolver,
    build_resolver,
)
from asistente.skills.spotify import PlayArgs, PlayOutcome, SpotifyClient, SpotifyPlaySkill

_OK = PlayOutcome(True)
_NO_ESTA = PlayOutcome(False, "no_encontrado")


class FakeOllama:
    """Doble de `ollama.Client` con lo justo: devuelve lo que se le diga."""

    def __init__(self, *respuestas: dict[str, Any], error: Exception | None = None) -> None:
        self._respuestas = list(respuestas)
        self._error = error
        #: El texto del prompt de cada llamada, para poder afirmar sobre el.
        self.prompts: list[str] = []

    def chat(self, **kwargs: Any) -> dict[str, Any]:
        self.prompts.append(kwargs["messages"][-1]["content"])
        if self._error is not None:
            raise self._error
        payload = self._respuestas.pop(0) if self._respuestas else {}
        return {"message": {"content": json.dumps(payload)}}


def _resolver(monkeypatch: pytest.MonkeyPatch, fake: FakeOllama) -> MusicResolver:
    """Un resolvedor cuyo Ollama es `fake`.

    Se pone un modulo `ollama` falso en `sys.modules` en vez de parchear el
    atributo despues de construir: asi el `from ollama import Client` del
    constructor —que es codigo de produccion— se ejecuta de verdad.
    """
    modulo = types.ModuleType("ollama")
    modulo.Client = lambda **_: fake  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ollama", modulo)
    return MusicResolver(LlmConfig())


class FakePlayer:
    """Doble de `SpotifyClient` a la altura de `play_query`.

    Se dobla aqui y no en spotipy a proposito: lo que se prueba en este fichero
    es CUANDO se decide reintentar, no como busca Spotify. Eso ya tiene sus
    tests en `test_spotify.py`.
    """

    def __init__(self, *resultados: PlayOutcome) -> None:
        self._resultados = list(resultados)
        #: (query, artist, kind) de cada busqueda, en orden.
        self.busquedas: list[tuple[str, str | None, str | None]] = []

    def play_query(
        self,
        query: str,
        artist: str | None = None,
        kind: str | None = None,
    ) -> PlayOutcome:
        self.busquedas.append((query, artist, kind))
        return self._resultados.pop(0) if self._resultados else _NO_ESTA


class FakeResolver:
    """Doble del resolvedor, para probar la cascada sin pasar por el prompt."""

    def __init__(self, correccion: MusicQuery | None = None) -> None:
        self._correccion = correccion
        self.llamadas = 0

    def resolve(self, query: str, artist: str | None = None) -> MusicQuery | None:
        self.llamadas += 1
        return self._correccion


def _skill(player: FakePlayer, resolver: FakeResolver | None = None) -> SpotifyPlaySkill:
    return SpotifyPlaySkill(player, resolver)  # type: ignore[arg-type]


# ------------------------------------------------------- la cascada por coste
#
# El LLM cuesta 600-900 ms. Todo esto existe para que solo los pague quien no
# tenia otra salida.


def test_the_llm_is_not_called_when_the_cheap_path_works() -> None:
    """EL TEST QUE PROTEGE EL DISENO.

    Si esto se rompe no falla nada visible: simplemente cada peticion de musica
    que ya funcionaba empieza a costar casi un segundo mas.
    """
    player, resolver = FakePlayer(_OK), FakeResolver(MusicQuery(titulo="lo que sea"))
    resultado = _skill(player, resolver).execute(PlayArgs(query="rock"))

    assert resultado.ok
    assert resolver.llamadas == 0
    assert len(player.busquedas) == 1


@pytest.mark.parametrize("motivo", ["sin_spotify", "sin_dispositivo", "error"])
def test_the_llm_is_not_called_for_problems_a_better_name_cannot_fix(motivo: str) -> None:
    """Sin Spotify abierto, escribir mejor el titulo no arregla nada.

    Reintentar aqui seria pagar el LLM para volver a chocar con la misma pared,
    y ademas retrasaria el aviso que si es util ("abrelo y te lo pongo").
    """
    player = FakePlayer(PlayOutcome(False, motivo))
    resolver = FakeResolver(MusicQuery(titulo="Loving Machine", artista="TV Girl"))

    resultado = _skill(player, resolver).execute(PlayArgs(query="lovin machin"))

    assert not resultado.ok
    assert resolver.llamadas == 0
    assert len(player.busquedas) == 1


def test_a_correction_earns_a_second_search() -> None:
    player = FakePlayer(_NO_ESTA, _OK)
    resolver = FakeResolver(MusicQuery(titulo="Loving Machine", artista="TV Girl"))

    resultado = _skill(player, resolver).execute(
        PlayArgs(query="lovin machin", artist="tibi guerl")
    )

    assert resultado.ok
    assert resolver.llamadas == 1
    assert player.busquedas == [
        ("lovin machin", "tibi guerl", None),
        ("Loving Machine", "TV Girl", None),
    ]


def test_a_correction_that_changes_nothing_is_not_searched_again() -> None:
    """El modelo devuelve lo mismo que se le dio: no habia nada que corregir.

    Es la respuesta correcta por su parte, pero repetir la busqueda daria
    exactamente el mismo `no_encontrado` y una peticion de mas a Spotify.
    """
    player = FakePlayer(_NO_ESTA)
    resolver = FakeResolver(MusicQuery(titulo="Rock Del Bueno"))

    resultado = _skill(player, resolver).execute(PlayArgs(query="rock del bueno"))

    assert not resultado.ok
    assert resolver.llamadas == 1
    assert len(player.busquedas) == 1


def test_only_the_artist_changing_is_enough_to_retry() -> None:
    """El titulo estaba bien y el grupo mal. Es media correccion y vale igual."""
    player = FakePlayer(_NO_ESTA, _OK)
    resolver = FakeResolver(MusicQuery(titulo="Loving Machine", artista="TV Girl"))

    _skill(player, resolver).execute(PlayArgs(query="loving machine", artist="tv gery"))

    assert player.busquedas[1] == ("Loving Machine", "TV Girl", None)


def test_the_kind_survives_the_correction() -> None:
    """"Pon el album X" sigue buscando un album despues de corregir el nombre.

    El LLM corrige COMO se escribe, no QUE se pidio: esa parte ya la decidio el
    router y no tiene por que volver a jugarse.
    """
    player = FakePlayer(_NO_ESTA, _OK)
    resolver = FakeResolver(MusicQuery(titulo="Nevermind", artista="Nirvana"))

    _skill(player, resolver).execute(
        PlayArgs(query="neverman", artist="nirvana", kind="disco")
    )

    assert player.busquedas[1] == ("Nevermind", "Nirvana", "album")


def test_a_correction_without_an_artist_does_not_invent_one() -> None:
    """`artista=""` tiene que llegar a `play_query` como None, no como "".

    Con cadena vacia, `play_query` creeria que se pidio una cancion concreta y
    se saltaria la cascada de playlists y albumes.
    """
    player = FakePlayer(_NO_ESTA, _OK)
    resolver = FakeResolver(MusicQuery(titulo="Lovers Rock"))

    _skill(player, resolver).execute(PlayArgs(query="ponlobes rock"))

    assert player.busquedas[1] == ("Lovers Rock", None, None)


def test_without_a_resolver_it_behaves_exactly_as_before() -> None:
    player = FakePlayer(_NO_ESTA)
    resultado = _skill(player).execute(PlayArgs(query="lovin machin"))

    assert not resultado.ok
    assert len(player.busquedas) == 1


def test_the_failure_names_what_was_asked_not_what_the_llm_proposed() -> None:
    """Si la correccion tampoco suena, se repite lo que dijo el usuario.

    Nombrarle un titulo que el no dijo —y que el modelo pudo inventarse— haria
    creer que se busco otra cosa.
    """
    player = FakePlayer(_NO_ESTA, _NO_ESTA)
    resolver = FakeResolver(MusicQuery(titulo="Loving Machine", artista="TV Girl"))

    resultado = _skill(player, resolver).execute(
        PlayArgs(query="lovin machin", artist="tibi guerl")
    )

    assert not resultado.ok
    assert resultado.speech is not None
    assert "lovin machin de tibi guerl" in resultado.speech
    assert "Loving Machine" not in resultado.speech


def test_a_song_that_plays_says_nothing_even_after_a_correction() -> None:
    """El silencio de `spotify.play` no cambia porque haya habido correccion:
    la cancion empezando ya es la confirmacion, y hablar se pisa con ella."""
    player = FakePlayer(_NO_ESTA, _OK)
    resolver = FakeResolver(MusicQuery(titulo="Loving Machine", artista="TV Girl"))

    resultado = _skill(player, resolver).execute(PlayArgs(query="lovin machin"))

    assert resultado.ok and resultado.speech is None


# ------------------------------------------------- corregir no es inventar
#
# Puntuaciones MEDIDAS con `similitud` sobre transcripciones reales de Whisper.
# Correcciones de verdad: 73.9-92.0. Invenciones: 35.9-63.4. El umbral (72) cae
# en el hueco.


@pytest.mark.parametrize(
    ("oido", "artista_oido", "titulo", "artista"),
    [
        ("lovin machin", "tibi guerl", "Loving Machine", "TV Girl"),          # 80.0
        ("no tinelsmatters", "metalica", "Nothing Else Matters", "Metallica"),  # 92.0
        ("esmels laik tin espirit", "nirvana", "Smells Like Teen Spirit", "Nirvana"),  # 85.2
        ("ponlobes rock", "tv girl", "Lovers Rock", "TV Girl"),               # 85.0
        ("blain ding lights", "uiquen", "Blinding Lights", "The Weeknd"),     # 73.9
    ],
)
def test_a_real_correction_is_accepted(
    monkeypatch: pytest.MonkeyPatch,
    oido: str,
    artista_oido: str,
    titulo: str,
    artista: str,
) -> None:
    fake = FakeOllama({"titulo": titulo, "artista": artista})
    corregido = _resolver(monkeypatch, fake).resolve(oido, artista_oido)

    assert corregido == MusicQuery(titulo=titulo, artista=artista)


@pytest.mark.parametrize(
    ("oido", "artista_oido", "titulo", "artista"),
    [
        # El modelo conoce OTRA cancion parecida y la ofrece con la misma
        # seguridad. Esto es exactamente lo que habia que impedir.
        ("loving machine", "tv gery", "Love Machine", "The Miracles"),   # 63.4
        ("lovin machin", "tibi guerl", "Loving You", "Minnie Riperton"),  # 55.8
        # Rellenar el hueco con lo primero que se le ocurre.
        ("cancion de la playa", None, "Despacito", "Luis Fonsi"),         # 35.9
    ],
)
def test_an_invention_is_refused(
    monkeypatch: pytest.MonkeyPatch,
    oido: str,
    artista_oido: str | None,
    titulo: str,
    artista: str,
) -> None:
    """Se prefiere fallar a poner una cancion que nadie pidio.

    Rechazar una correccion buena cuesta un "no lo encontre", que es lo que
    habria pasado sin el LLM. Aceptar una mala pone otra cancion a sonar y
    parece que funciono.
    """
    fake = FakeOllama({"titulo": titulo, "artista": artista})

    assert _resolver(monkeypatch, fake).resolve(oido, artista_oido) is None


def test_an_artist_that_was_never_said_is_not_held_against_the_correction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """"Pon creep" -> "Creep de Radiohead" es perfecto y puntuaria 52.6.

    Si nadie dijo el grupo, el que anade el modelo es informacion NUEVA: no hay
    nada oido contra lo que contrastarla, y meterla en la comparacion solo
    penaliza a los titulos cortos con grupos de nombre largo.
    """
    fake = FakeOllama({"titulo": "Creep", "artista": "Radiohead"})
    corregido = _resolver(monkeypatch, fake).resolve("creep")

    assert corregido == MusicQuery(titulo="Creep", artista="Radiohead")


def test_the_threshold_sits_between_the_two_measurements() -> None:
    """Fija el hueco medido: mejor invencion 63.4, peor correccion 73.9.

    Si alguien mueve el umbral, este test dice contra que se estaba midiendo.
    """
    assert 63.4 < PLAUSIBLE_THRESHOLD <= 73.9


# ---------------------------------------------------- lo que el modelo devuelve


def test_the_prompt_carries_the_whole_phrase(monkeypatch: pytest.MonkeyPatch) -> None:
    """Titulo y artista vuelven a juntarse: separados, el modelo pierde contexto.

    El router los partio por el ultimo "de" y esa particion puede estar mal; el
    modelo tiene que ver la frase como se dijo.
    """
    fake = FakeOllama({"titulo": "Loving Machine", "artista": "TV Girl"})
    _resolver(monkeypatch, fake).resolve("lovin machin", "tibi guerl")

    assert "lovin machin de tibi guerl" in fake.prompts[0]


def test_ollama_being_down_is_not_a_crash(monkeypatch: pytest.MonkeyPatch) -> None:
    """Un fallo de servicio deja el comando como estaba, no lo rompe."""
    fake = FakeOllama(error=ConnectionError("connection refused"))

    assert _resolver(monkeypatch, fake).resolve("lovin machin") is None


def test_extra_fields_from_the_llm_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """`extra="forbid"`, igual que en `ToolCall`: el modelo emite datos
    validados contra un modelo cerrado, nunca campos libres."""
    fake = FakeOllama({"titulo": "Creep", "artista": "Radiohead", "uri": "spotify:track:x"})

    assert _resolver(monkeypatch, fake).resolve("creep") is None


@pytest.mark.parametrize("payload", [{}, {"titulo": "", "artista": "Metallica"}])
def test_an_answer_without_a_title_is_rejected(
    monkeypatch: pytest.MonkeyPatch, payload: dict[str, Any]
) -> None:
    fake = FakeOllama(payload)

    assert _resolver(monkeypatch, fake).resolve("no tinelsmatters") is None


def test_a_missing_artist_defaults_to_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """El esquema pide las dos claves, pero el decodificado restringido guia, no
    garantiza. Sin artista se sigue pudiendo buscar por titulo."""
    fake = FakeOllama({"titulo": "Lovers Rock"})
    corregido = _resolver(monkeypatch, fake).resolve("ponlobes rock")

    assert corregido == MusicQuery(titulo="Lovers Rock", artista="")


def test_without_ollama_installed_there_is_simply_no_resolver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """En la Mac no hay Ollama, y el asistente tiene que arrancar igual.

    Degradar en silencio es correcto AQUI: sin resolvedor la busqueda de musica
    funciona como funcionaba antes de que existiera este modulo.
    """
    monkeypatch.setitem(sys.modules, "ollama", None)

    assert build_resolver(LlmConfig()) is None


# ---------------------------------------------------- el interruptor del indice


def test_the_library_index_can_be_switched_off() -> None:
    """`index_library: false` quita el paso 2 de la cascada, no los me gusta.

    "Pon mis me gusta" lee de la API en el momento y no depende del indice: lo
    que se apaga es emparejar titulos contra el, que son hasta 20 peticiones al
    arrancar.
    """
    config = Config.model_validate({"spotify": {"index_library": False}})
    client = SpotifyClient(config, Secrets())
    client._client = object()  # inyeccion deliberada: basta con que no sea None

    assert client.liked_tracks() == ()
