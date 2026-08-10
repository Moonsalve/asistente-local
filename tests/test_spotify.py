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
    LikedSkill,
    PlayArgs,
    SpotifyClient,
    SpotifyPlaySkill,
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

    def devices(self) -> dict[str, Any]:
        return {"devices": self._devices}

    def search(self, q: str, type: str, limit: int = 1) -> dict[str, Any]:  # noqa: A002
        self.busquedas.append((q, type))
        clave = {"playlist": "playlists", "album": "albums", "track": "tracks",
                 "artist": "artists"}[type]
        return {clave: {"items": self._catalogo.get(f"{type}:{q}", [])}}

    def current_user_playlists(self, limit: int = 50, offset: int = 0) -> dict[str, Any]:
        return {"items": self._playlists[offset : offset + limit]}

    def current_user_saved_tracks(self, limit: int = 50) -> dict[str, Any]:
        return {"items": [{"track": {"uri": uri}} for uri in self._guardadas[:limit]]}

    def start_playback(self, **kwargs: Any) -> None:
        self.reproducido.append(kwargs)


def _client(fake: FakeSpotify) -> SpotifyClient:
    client = SpotifyClient(Config(), Secrets())
    client._client = fake  # noqa: SLF001 - inyeccion deliberada en test
    return client


def _uri(name: str, uri: str) -> dict[str, Any]:
    return {"name": name, "uri": uri}


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
            'track:track:"la bamba" artist:"los lobos"': [_uri("La Bamba", "spotify:track:OK")]
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
        catalogo={"track:la bamba los lobos": [_uri("La Bamba", "spotify:track:OK")]}
    )
    outcome = _client(fake).play_query("la bamba", artist="los lobos")

    assert outcome.ok
    assert len(fake.busquedas) == 2


def test_an_artist_skips_the_playlist_cascade() -> None:
    """"Pon la cancion X de Y" no puede acabar en una playlist llamada X."""
    fake = FakeSpotify(
        playlists=[_uri("La Bamba", "spotify:playlist:MIA")],
        catalogo={"track:la bamba los lobos": [_uri("La Bamba", "spotify:track:OK")]},
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


def test_a_song_with_an_artist_is_confirmed_out_loud() -> None:
    """Aqui si: es el caso donde puede haber sonado algo que no era, y oir el
    nombre lo delata sin tener que mirar la pantalla."""
    fake = FakeSpotify(catalogo={"track:la bamba los lobos": [_uri("x", "spotify:track:OK")]})
    result = SpotifyPlaySkill(_client(fake)).execute(PlayArgs(query="la bamba", artist="los lobos"))
    assert result.ok
    assert "los lobos" in (result.speech or "")


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
