"""Tests del cliente de Spotify.

No se habla con la Web API: se sustituye spotipy por un doble que devuelve lo
que devolveria la API de verdad. Lo que se prueba es lo que estaba mal, que no
era la red sino QUE SE LE PIDE:

  - buscar una playlist tuya en el catalogo publico (donde no esta),
  - tratar los Me Gusta como si fueran una playlist (no lo son),
  - meter el nombre del grupo dentro del titulo de la cancion.
"""

from __future__ import annotations

from typing import Any

import pytest

from asistente.config import Config, Secrets
from asistente.skills.base import NoArgs
from asistente.skills.spotify import (
    MIN_COBERTURA,
    LikedSkill,
    PlayArgs,
    SpotifyClient,
    SpotifyPlaySkill,
    cobertura,
)


class FakeSpotify:
    """Doble de `spotipy.Spotify` con lo justo para estos tests."""

    def __init__(
        self,
        *,
        devices: list[dict[str, Any]] | None = None,
        catalogo: dict[str, list[dict[str, Any]]] | None = None,
        playlists: list[dict[str, Any]] | None = None,
        guardadas: list[str] | None = None,
    ) -> None:
        self._devices = devices if devices is not None else [{"id": "PC", "is_active": True}]
        #: consulta -> items que devuelve la busqueda, por tipo
        self._catalogo = catalogo or {}
        self._playlists = playlists or []
        self._guardadas = guardadas or []
        #: lo que se llego a reproducir, para poder afirmar sobre ello
        self.reproducido: list[dict[str, Any]] = []
        self.busquedas: list[tuple[str, str]] = []
        self.paginas_guardadas: list[int] = []

    def devices(self) -> dict[str, Any]:
        return {"devices": self._devices}

    def search(self, q: str, type: str, limit: int = 1) -> dict[str, Any]:  # noqa: A002
        self.busquedas.append((q, type))
        clave = {"playlist": "playlists", "album": "albums", "track": "tracks",
                 "artist": "artists"}[type]
        return {clave: {"items": self._catalogo.get(f"{type}:{q}", [])}}

    def current_user_playlists(self, limit: int = 50, offset: int = 0) -> dict[str, Any]:
        return {"items": self._playlists[offset : offset + limit]}

    def current_user_saved_tracks(self, limit: int = 50, offset: int = 0) -> dict[str, Any]:
        self.paginas_guardadas.append(offset)
        ventana = self._guardadas[offset : offset + limit]
        return {
            "items": [
                {"track": {"uri": uri, "name": nombre,
                           "artists": [{"name": a} for a in artistas]}}
                for uri, nombre, *artistas in (_guardada(g) for g in ventana)
            ],
            "total": len(self._guardadas),
        }

    def start_playback(self, **kwargs: Any) -> None:
        self.reproducido.append(kwargs)


def _client(fake: FakeSpotify) -> SpotifyClient:
    client = SpotifyClient(Config(), Secrets())
    client._client = fake  # noqa: SLF001 - inyeccion deliberada en test
    return client


def _guardada(entry: str | tuple[str, ...]) -> tuple[str, ...]:
    """Un me gusta del doble: una URI suelta, o (uri, titulo, *artistas)."""
    return (entry, "", ) if isinstance(entry, str) else entry


def _uri(name: str, uri: str, *artistas: str) -> dict[str, Any]:
    """Un item de la API. `artists` solo lo traen las canciones y los albumes, y
    forma parte de lo que se compara: sin el, "la bamba de los lobos" solo
    cubriria la mitad de las palabras que se dijeron."""
    return {"name": name, "uri": uri, "artists": [{"name": a} for a in artistas]}


# --------------------------------------------------------------- tus playlists


def test_your_own_playlist_beats_a_public_one_with_the_same_name() -> None:
    """EL BUG. "pon mi playlist de gym" buscaba "gym" en el catalogo publico y
    reproducia la primera coincidencia del mundo, no la tuya."""
    fake = FakeSpotify(
        playlists=[_uri("Gym", "spotify:playlist:MIA")],
        catalogo={"playlist:gym": [_uri("Gym Hits 2020", "spotify:playlist:AJENA")]},
    )
    outcome = _client(fake).play_query("gym", kind="playlist")

    assert outcome.ok
    assert fake.reproducido == [{"device_id": "PC", "context_uri": "spotify:playlist:MIA"}]


def test_a_public_playlist_still_works_when_you_have_none() -> None:
    """Preferir las tuyas no puede significar renunciar al catalogo."""
    fake = FakeSpotify(catalogo={"playlist:rock": [_uri("Rock Classics", "spotify:playlist:PUB")]})
    outcome = _client(fake).play_query("rock", kind="playlist")

    assert outcome.ok
    assert fake.reproducido[0]["context_uri"] == "spotify:playlist:PUB"


def test_playlists_are_matched_by_whole_words_not_by_substring() -> None:
    """Con `WRatio`, "rock" casaba al 90 con "Rocky soundtrack" por ser
    subcadena y sonaba una banda sonora en vez de rock. `token_set_ratio`
    compara palabras completas."""
    fake = FakeSpotify(
        playlists=[_uri("Rocky soundtrack", "spotify:playlist:PELICULA")],
        catalogo={"playlist:rock": [_uri("Rock Classics", "spotify:playlist:PUB")]},
    )
    _client(fake).play_query("rock", kind="playlist")

    assert fake.reproducido[0]["context_uri"] == "spotify:playlist:PUB"


def test_a_typo_in_the_name_still_finds_your_playlist() -> None:
    """La otra mitad: Whisper transcribe como suena, y el emparejamiento tiene
    que absorber esa diferencia."""
    fake = FakeSpotify(playlists=[_uri("Concentración", "spotify:playlist:MIA")])
    assert _client(fake).find_own_playlist("concentracion") is not None


def test_the_playlist_list_is_cached() -> None:
    """Enumerarlas esta en la ruta critica de un comando hablado."""
    fake = FakeSpotify(playlists=[_uri("Gym", "spotify:playlist:MIA")])
    client = _client(fake)
    llamadas = 0
    original = fake.current_user_playlists

    def contar(**kwargs: Any) -> dict[str, Any]:
        nonlocal llamadas
        llamadas += 1
        return original(**kwargs)

    fake.current_user_playlists = contar  # type: ignore[method-assign]
    client.own_playlists()
    client.own_playlists()
    assert llamadas == 1


# ------------------------------------------------------------------ me gusta


def test_liked_songs_are_played_as_a_list_of_tracks() -> None:
    """Los Me Gusta NO tienen `context_uri`: no son una playlist. Intentar
    arrancarlos como contexto es un 404 garantizado."""
    fake = FakeSpotify(guardadas=["spotify:track:1", "spotify:track:2"])
    outcome = _client(fake).play_liked()

    assert outcome.ok
    assert fake.reproducido == [
        {"device_id": "PC", "uris": ["spotify:track:1", "spotify:track:2"]}
    ]
    assert not fake.busquedas, "los me gusta no se buscan, se leen"


def test_an_empty_library_says_so_instead_of_playing_nothing() -> None:
    fake = FakeSpotify(guardadas=[])
    outcome = _client(fake).play_liked()
    assert not outcome.ok
    assert outcome.reason == "no_encontrado"
    assert not fake.reproducido


# ------------------------------------------------------------ cancion + artista


def test_a_song_with_an_artist_uses_field_filters() -> None:
    """Sin filtros, "labios compartidos de mana" se busca como un titulo entero
    y Spotify devuelve cualquier cosa."""
    fake = FakeSpotify(
        catalogo={
            'track:track:"la bamba" artist:"los lobos"': [
                _uri("La Bamba", "spotify:track:OK", "Los Lobos")
            ]
        }
    )
    outcome = _client(fake).play_query("la bamba", artist="los lobos")

    assert outcome.ok
    assert fake.reproducido == [{"device_id": "PC", "uris": ["spotify:track:OK"]}]
    assert fake.busquedas[0] == ('track:"la bamba" artist:"los lobos"', "track")


def test_free_text_is_the_second_attempt() -> None:
    """Los filtros exigen que el titulo este escrito como en Spotify. Cuando
    Whisper se desvia una letra no devuelven nada, y rendirse ahi seria peor que
    reintentar con la busqueda tolerante."""
    fake = FakeSpotify(
        catalogo={"track:la bamba los lobos": [_uri("La Bamba", "spotify:track:OK", "Los Lobos")]}
    )
    outcome = _client(fake).play_query("la bamba", artist="los lobos")

    assert outcome.ok
    assert len(fake.busquedas) == 2


def test_an_artist_skips_the_playlist_cascade() -> None:
    """"Pon la cancion X de Y" no puede acabar en una playlist llamada X."""
    fake = FakeSpotify(
        playlists=[_uri("La Bamba", "spotify:playlist:MIA")],
        catalogo={
            "track:la bamba los lobos": [_uri("La Bamba", "spotify:track:OK", "Los Lobos")]
        },
    )
    _client(fake).play_query("la bamba", artist="los lobos")
    assert fake.reproducido[0]["uris"] == ["spotify:track:OK"]


# ------------------------------------------------------------------- cascada


def test_without_a_kind_a_playlist_is_preferred_over_a_track() -> None:
    """"Pon rock" casi siempre es una lista, no la primera cancion titulada
    "Rock"."""
    fake = FakeSpotify(
        catalogo={
            "playlist:rock": [_uri("Rock", "spotify:playlist:P")],
            "track:rock": [_uri("Rock", "spotify:track:T")],
        }
    )
    _client(fake).play_query("rock")
    assert fake.reproducido[0]["context_uri"] == "spotify:playlist:P"


def test_an_explicit_kind_searches_only_that() -> None:
    """Si la frase dijo "album", una playlist con ese nombre no vale."""
    fake = FakeSpotify(
        catalogo={
            "playlist:abbey road": [_uri("Abbey Road", "spotify:playlist:P")],
            "album:abbey road": [_uri("Abbey Road", "spotify:album:A")],
        }
    )
    _client(fake).play_query("abbey road", kind="album")
    assert fake.reproducido[0]["context_uri"] == "spotify:album:A"


def test_null_items_from_the_api_do_not_crash() -> None:
    """La Web API devuelve `null` entre los resultados con mas frecuencia de la
    que su documentacion sugiere."""
    fake = FakeSpotify(catalogo={"playlist:rock": [None]})  # type: ignore[list-item]
    outcome = _client(fake).play_query("rock", kind="playlist")
    assert not outcome.ok
    assert outcome.reason == "no_encontrado"


# ----------------------------------------------------- buscar en tus me gusta
#
# Tu biblioteca va PRIMERO y no es solo una preferencia: es donde se absorbe lo
# que Whisper destroza. Un titulo en ingles dicho en espanol sale transcrito
# como suena, y con eso la Web API no encuentra nada; contra unos cientos de
# canciones tuyas, un emparejamiento difuso si acierta.

BIBLIOTECA = [
    ("spotify:track:LM", "Loving Machine", "TV Girl"),
    ("spotify:track:LR", "Lovers Rock", "TV Girl"),
    ("spotify:track:NEM", "Nothing Else Matters", "Metallica"),
    ("spotify:track:BL", "Blinding Lights", "The Weeknd"),
    ("spotify:track:CN", "La Camisa Negra", "Juanes"),
]


def test_your_library_is_searched_before_the_catalog() -> None:
    fake = FakeSpotify(
        guardadas=BIBLIOTECA,
        catalogo={"track:loving machine tv girl": [_uri("Otra", "spotify:track:NO", "Otro")]},
    )
    outcome = _client(fake).play_query("loving machine", artist="tv girl")

    assert outcome.ok
    assert fake.reproducido == [{"device_id": "PC", "uris": ["spotify:track:LM"]}]
    assert not fake.busquedas, "si esta en tu biblioteca, no hace falta buscar fuera"


@pytest.mark.parametrize(
    ("dicho", "esperado"),
    [
        # Transcripciones REALES de `say` en voces españolas, medidas con
        # Whisper small: es exactamente lo que le llega al asistente.
        ("lovin machine de tv girl", "spotify:track:LM"),
        ("lobes rock de tv girl", "spotify:track:LR"),
        ("no tinelsmatters de metalica", "spotify:track:NEM"),
        # Y lo que ya funcionaba tiene que seguir funcionando.
        ("la camisa negra de juanes", "spotify:track:CN"),
    ],
)
def test_a_mangled_english_title_still_finds_the_song(dicho: str, esperado: str) -> None:
    """EL CASO REPORTADO. Ninguna de estas cadenas encuentra nada en la Web API,
    porque no es asi como esta escrito el titulo. Contra la biblioteca de Juan
    sí, porque el espacio de busqueda son sus canciones y no millones."""
    fake = FakeSpotify(guardadas=BIBLIOTECA)
    titulo, _, artista = dicho.partition(" de ")
    outcome = _client(fake).play_query(titulo, artist=artista or None)

    assert outcome.ok, f"{dicho!r} no encontro nada"
    assert fake.reproducido[0]["uris"] == [esperado]


def test_something_you_do_not_have_falls_through_to_the_catalog() -> None:
    """Preferir tu biblioteca no puede significar quedarse encerrado en ella."""
    fake = FakeSpotify(
        guardadas=BIBLIOTECA,
        catalogo={"playlist:jazz": [_uri("Jazz Classics", "spotify:playlist:PUB")]},
    )
    outcome = _client(fake).play_query("jazz")

    assert outcome.ok
    assert fake.reproducido[0]["context_uri"] == "spotify:playlist:PUB"


def test_the_library_match_is_not_a_substring_match() -> None:
    """El riesgo de bajar el umbral: con una biblioteca entera delante,
    cualquier peticion corta casaria con la primera cancion que contenga esa
    palabra. "rock" no puede convertirse en "Lovers Rock"."""
    fake = FakeSpotify(guardadas=BIBLIOTECA)
    assert _client(fake).find_liked("rock") is None


def test_the_library_index_is_cached() -> None:
    """Son hasta 20 peticiones: se pagan al arrancar, no en cada comando."""
    fake = FakeSpotify(guardadas=BIBLIOTECA)
    client = _client(fake)
    client.liked_tracks()
    antes = len(fake.paginas_guardadas)
    client.liked_tracks()
    assert len(fake.paginas_guardadas) == antes


# ---------------------------------------------- sesgo del STT con tu música


def test_hotwords_lead_with_the_artists() -> None:
    """Los artistas se repiten -veinte canciones de un grupo son un solo
    termino- y son el ancla que arrastra al resto de la frase."""
    fake = FakeSpotify(guardadas=BIBLIOTECA)
    terminos = _client(fake).hotwords().split(", ")

    assert terminos[:4] == ["TV Girl", "Metallica", "The Weeknd", "Juanes"]
    assert "Loving Machine" in terminos


def test_hotwords_respect_the_limit() -> None:
    """Whisper acota el contexto del prompt: pasarse desplaza al audio."""
    fake = FakeSpotify(guardadas=BIBLIOTECA)
    assert len(_client(fake).hotwords(limit=2).split(", ")) <= 4


def test_hotwords_without_spotify_are_empty() -> None:
    assert SpotifyClient(Config(), Secrets()).hotwords() == ""


# ------------------------------------------------- el resultado tiene que valer
#
# `search()` SIEMPRE devuelve algo. Quedarse con el primero era lo que hacia que
# "reproduce loving machine de tv girl" acabara poniendo cualquier cosa y
# pareciera que el comando habia funcionado.


def test_a_result_that_does_not_cover_what_was_asked_is_refused() -> None:
    """EL CASO REPORTADO. Se pide una cancion concreta y Spotify devuelve una
    playlist que solo comparte el nombre del grupo: dos palabras de cuatro."""
    fake = FakeSpotify(
        catalogo={"playlist:loving machine de tv girl": [_uri("TV Girl", "spotify:playlist:NO")]}
    )
    outcome = _client(fake).play_query("loving machine de tv girl")

    assert not outcome.ok
    assert outcome.reason == "no_encontrado"
    assert not fake.reproducido, "mejor no sonar nada que sonar otra cosa"


def test_a_broader_result_is_fine_when_you_asked_broadly() -> None:
    """La medida es DIRECCIONAL: lo que se pidio tiene que estar en el
    resultado, no al reves. Pediste jazz y te ponen una lista de jazz: correcto.
    Es la misma comparacion que rechaza el caso de arriba."""
    fake = FakeSpotify(
        catalogo={"playlist:jazz": [_uri("Jazz Classics", "spotify:playlist:SI")]}
    )
    assert _client(fake).play_query("jazz").ok


def test_the_best_of_several_results_wins_not_the_first() -> None:
    """Spotify no siempre pone el bueno arriba; por eso se piden varios."""
    fake = FakeSpotify(
        catalogo={
            "track:loving machine tv girl": [
                _uri("Machine", "spotify:track:NO", "Otra Banda"),
                _uri("Loving Machine", "spotify:track:SI", "TV Girl"),
            ]
        }
    )
    outcome = _client(fake).play_query("loving machine", artist="tv girl")

    assert outcome.ok
    assert fake.reproducido[0]["uris"] == ["spotify:track:SI"]


def test_the_whole_phrase_is_the_last_resort_of_the_artist_guess() -> None:
    """El router entrega `artist` como HIPOTESIS partiendo por el ultimo "de", y
    a veces se parte mal. El ultimo intento busca la frase tal cual se dijo."""
    fake = FakeSpotify(
        catalogo={
            "track:el dia que me quieras de gardel": [
                _uri("El día que me quieras", "spotify:track:SI", "Carlos Gardel")
            ]
        }
    )
    outcome = _client(fake).play_query("el dia que me quieras", artist="gardel")

    assert outcome.ok
    assert len(fake.busquedas) == 3, "filtros, texto libre, y la frase entera"


@pytest.mark.parametrize(
    ("pedido", "nombre", "artista", "vale"),
    [
        ("loving machine de tv girl", "Loving Machine", "TV Girl", True),
        ("loving machine de tv girl", "TV Girl", "", False),
        ("jazz", "Jazz Classics", "", True),
        ("rock", "Rocky soundtrack", "", False),
        # Whisper transcribe como suena: una letra de diferencia no puede
        # tumbar la busqueda.
        ("loving machin de tv girl", "Loving Machine", "TV Girl", True),
        # "de" no cuenta: es la preposicion mas comun del espanol y estaria en
        # todas las peticiones y en ningun titulo.
        ("la bamba", "La Bamba", "Los Lobos", True),
    ],
)
def test_coverage(pedido: str, nombre: str, artista: str, vale: bool) -> None:
    assert (cobertura(pedido, nombre, artista) >= MIN_COBERTURA) is vale


# ------------------------------------------------------------------- motivos


def test_no_device_is_a_different_problem_than_not_finding_it() -> None:
    """Un unico "no pude reproducir X" no distinguia "abre Spotify" de "no
    existe eso", que se arreglan de formas distintas."""
    fake = FakeSpotify(devices=[])
    outcome = _client(fake).play_query("rock")
    assert outcome.reason == "sin_dispositivo"

    skill = SpotifyPlaySkill(_client(fake))
    result = skill.execute(PlayArgs(query="rock"))
    assert not result.ok
    assert "Spotify" in (result.speech or "")


def test_a_disconnected_client_does_not_pretend() -> None:
    client = SpotifyClient(Config(), Secrets())
    assert client.play_query("rock").reason == "sin_spotify"
    assert client.play_liked().reason == "sin_spotify"
    assert client.own_playlists() == ()


def test_the_liked_skill_stays_quiet_when_it_works() -> None:
    """La musica sonando ya es la confirmacion."""
    fake = FakeSpotify(guardadas=["spotify:track:1"])
    assert LikedSkill(_client(fake)).execute(NoArgs()).speech is None


def test_finding_a_song_says_nothing() -> None:
    """Se hablaba para confirmar el titulo, y se pisaba con el principio de la
    propia cancion. La musica sonando ya es la confirmacion; solo se habla
    cuando algo NO sale."""
    fake = FakeSpotify(
        catalogo={"track:la bamba los lobos": [_uri("La Bamba", "spotify:track:OK", "Los Lobos")]}
    )
    result = SpotifyPlaySkill(_client(fake)).execute(PlayArgs(query="la bamba", artist="los lobos"))
    assert result.ok
    assert result.speech is None


# ----------------------------------------------------------- validacion de args


@pytest.mark.parametrize(
    ("hablado", "esperado"),
    [("playlist", "playlist"), ("lista de reproduccion", "playlist"), ("disco", "album"),
     ("tema", "cancion"), ("rola", "cancion"), ("banda", "artista")],
)
def test_the_spoken_word_becomes_an_api_type(hablado: str, esperado: str) -> None:
    assert PlayArgs(query="x", kind=hablado).kind == esperado


def test_an_unknown_kind_falls_back_to_deciding_it() -> None:
    """"Musica" no es un tipo de la API. Rechazar la frase entera por eso seria
    convertir un matiz en un fallo; None significa "usa la cascada"."""
    assert PlayArgs(query="x", kind="musica").kind is None


def test_extra_args_are_rejected() -> None:
    """La frontera de seguridad: el LLM no puede colar campos que no existen."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        PlayArgs(query="x", ruta="C:/algo")  # type: ignore[call-arg]
