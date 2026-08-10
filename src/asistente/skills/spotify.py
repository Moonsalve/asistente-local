"""Cliente de Spotify y skill `spotify.play`.

AUTENTICACION
-------------
Se usa el flujo Authorization Code con PKCE. Autorizas UNA vez desde el
navegador; a partir de ahi spotipy guarda un refresh token en disco y lo renueva
solo. El cache va a `%LOCALAPPDATA%\\asistente-local`, nunca al repositorio: es
una credencial de larga duracion con permiso para controlar tu reproduccion.

PKCE en vez del flujo con client secret porque esto es una aplicacion de
escritorio: no hay servidor donde esconder un secreto, y PKCE esta disenado
exactamente para ese caso.

DEGRADACION
-----------
La Web API necesita un dispositivo activo. Si Spotify lleva rato cerrado no hay
ninguno, y la llamada falla. Por eso `media.*` cae a las teclas multimedia: el
comando siempre hace algo. `spotify.play` no puede degradar (buscar por nombre
requiere la API), asi que ahi si se avisa al usuario.

DOS COSAS QUE LA BUSQUEDA PUBLICA NO PUEDE HACER
------------------------------------------------
`search()` mira el catalogo PUBLICO de Spotify, y hay dos cosas tuyas que no
estan ahi:

  - **Tus playlists.** "pon mi playlist de gym" buscaba "gym" en el catalogo y
    reproducia la primera playlist publica que se llamara asi. Las propias hay
    que pedirlas por `current_user_playlists` y emparejarlas por nombre.
  - **Tus Me Gusta.** No son una playlist: no tienen `context_uri` con el que
    arrancar la reproduccion. Hay que leer `current_user_saved_tracks` y pasar
    la lista de URIs.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
from rapidfuzz import fuzz

from asistente.config import Config, Secrets
from asistente.router.text import normalize
from asistente.skills.base import Skill, SkillResult

log = logging.getLogger(__name__)

#: Que buscar cuando la frase lo dice ("pon el ALBUM de X"). None = decidelo tu.
Kind = Literal["playlist", "album", "cancion", "artista"]

#: Como llama la gente a cada cosa, ya normalizado. Lo rellena el slot `kind`
#: del catalogo, que captura la palabra tal cual se dijo.
_KINDS: dict[str, Kind] = {
    "playlist": "playlist",
    "play list": "playlist",
    "lista": "playlist",
    "lista de reproduccion": "playlist",
    "album": "album",
    "disco": "album",
    "cancion": "cancion",
    "tema": "cancion",
    "rola": "cancion",
    "track": "cancion",
    "artista": "artista",
    "grupo": "artista",
    "banda": "artista",
}

#: Similitud minima (0-100) para dar por buena una playlist tuya. Se usa
#: `token_set_ratio` y no `WRatio` porque compara palabras completas: con
#: WRatio, "rock" casaba al 90 con una playlist llamada "Rocky soundtrack" por
#: ser subcadena. Aqui "jazz" casa con "Jazz mix" (comparten la palabra) y no
#: con "Rocky" (no comparten ninguna).
PLAYLIST_THRESHOLD = 82.0

#: Palabras que no aportan nada al comparar lo pedido con lo encontrado.
_VACIAS = frozenset({"de", "del", "la", "el", "los", "las", "un", "una", "y", "a", "en"})

#: Fraccion minima de las palabras dichas que tiene que aparecer en el
#: resultado para darlo por bueno.
#:
#: EL PROBLEMA QUE RESUELVE: `search()` SIEMPRE devuelve algo. Pedir "loving
#: machine de tv girl" y quedarse con el primer resultado significaba reproducir
#: una playlist llamada "TV Girl" —que comparte dos palabras de cinco— y parecer
#: que el comando habia funcionado. Ahora el resultado tiene que CUBRIR lo que
#: se dijo, no simplemente solaparse con ello, y si nada lo cubre se dice que no
#: se encontro en vez de poner otra cosa.
#:
#: La medida es direccional a proposito: "jazz" -> "Jazz Classics" cubre 1/1 y
#: vale (pediste jazz y te ponen jazz), pero "loving machine de tv girl" ->
#: "TV Girl" cubre 2/4 y no vale (pediste algo mucho mas concreto).
#:
#: 0.6 deja pasar 2 de 3 palabras y 3 de 5, y exige las dos cuando solo hay dos.
MIN_COBERTURA = 0.6

#: Similitud por palabra suelta, para absorber como transcribe Whisper
#: ("machin" -> "machine", 92.3). Alto a proposito: medido, a 85 "rock" casaba
#: con "rocky" (88.9) y volvia a colarse la banda sonora de Rocky.
#:
#: Ser estricto aqui es barato porque la COBERTURA ya es una fraccion: una
#: palabra mal transcrita de cuatro deja 0.75 y pasa igual. Esta segunda capa
#: solo tiene que salvar los casos de una o dos palabras, y ahi un falso
#: positivo (sonar otra cosa) es peor que un falso negativo (repetir la orden).
_SIMILITUD_PALABRA = 90

#: Cuantos resultados se piden para poder ELEGIR el que mejor cubre en vez de
#: quedarse con el primero. Cinco caben en la misma peticion y el ranking de
#: Spotify rara vez pone el bueno mas abajo.
_LIMITE_BUSQUEDA = 5

#: Cuantos Me Gusta se ponen en cola. Una sola llamada a la API (el maximo por
#: pagina es 50) y suficiente para un rato largo. Si tienes el aleatorio puesto
#: en Spotify, se aplica a esta cola: no hace falta que lo toquemos nosotros.
LIKED_LIMIT = 50

#: Cuantos Me Gusta se leen para poder BUSCAR dentro de ellos. Nada que ver con
#: el limite de arriba: aquello es lo que se pone en cola, esto es el indice
#: contra el que se emparejan los titulos. 20 peticiones en el peor caso, una
#: sola vez al arrancar.
LIKED_INDEX_LIMIT = 1000

#: Similitud minima (0-100) para dar por buena una cancion de TU biblioteca.
#:
#: Mas permisivo que el catalogo publico a proposito, y por dos razones: el
#: espacio de busqueda es de unos cientos de canciones tuyas en vez de millones,
#: asi que un falso positivo es mucho menos probable; y es justo aqui donde hay
#: que absorber los destrozos de Whisper con los titulos en ingles
#: ("Ponlobes Rock" por "pon Lovers Rock").
LIBRARY_THRESHOLD = 72.0

#: Vida del cache de tus playlists. Enumerarlas cuesta una peticion por cada 50,
#: y eso esta en la ruta critica de un comando de voz. Crear una playlist nueva
#: y quererla reproducir en el mismo minuto es raro; esperar medio segundo de
#: mas en cada "pon mi playlist de X" no lo es.
_PLAYLISTS_TTL_S = 600.0


def cobertura(query: str, *campos: str) -> float:
    """Que fraccion de lo que se pidio aparece en el resultado (0-1).

    Se compara palabra a palabra y no con una similitud global porque la
    pregunta es direccional: importa si el resultado contiene lo que dijiste, no
    si se parecen en conjunto. `token_set_ratio` daba 100 tanto a
    "jazz" -> "Jazz Classics" (correcto) como a
    "loving machine de tv girl" -> "TV Girl" (un desastre): en cuanto uno de los
    dos es subconjunto del otro, puntua perfecto en las dos direcciones.
    """
    pedidas = [p for p in normalize(query).split() if p and p not in _VACIAS]
    if not pedidas:
        return 0.0

    disponibles = {p for p in normalize(" ".join(campos)).split() if p}
    if not disponibles:
        return 0.0

    aciertos = sum(
        1
        for palabra in pedidas
        if palabra in disponibles
        or any(fuzz.ratio(palabra, otra) >= _SIMILITUD_PALABRA for otra in disponibles)
    )
    return aciertos / len(pedidas)


def similitud(pedido: str, candidato: str) -> float:
    """Cuanto se parecen "titulo artista" pedido y candidato (0-100).

    DOS MEDIDAS, y la segunda no es redundante:

    - `token_sort_ratio` ordena las palabras antes de comparar, asi que da igual
      si se dijo el artista primero o si sobra un "de", y sigue siendo sensible
      a las letras.
    - La misma comparacion SIN ESPACIOS. Hace falta porque Whisper no solo
      cambia letras: PEGA PALABRAS entre si. "nothing else matters" salio como
      "no tinelsmatters", y ahi cualquier medida por tokens se hunde
      (`token_sort` 50.9) mientras que sin espacios sube a 92.0.

    Se toma el maximo: son dos formas de romperse distintas y basta con
    sobrevivir a una.

    MEDIDO con transcripciones reales de `say` en voces espanolas contra una
    biblioteca de prueba: el peor acierto puntua 91.9 y el mejor fallo, 65.0.

    `token_set_ratio` NO vale aqui, aunque si valga para las playlists: premia
    las subcadenas, y con una biblioteca entera delante "rock" casaria al 100
    con "Lovers Rock".
    """
    return max(
        fuzz.token_sort_ratio(pedido, candidato),
        fuzz.ratio(pedido.replace(" ", ""), candidato.replace(" ", "")),
    )


@dataclass(frozen=True, slots=True)
class LikedTrack:
    """Una cancion de tus Me Gusta, lista para emparejar."""

    name: str
    artists: str
    uri: str

    @property
    def searchable(self) -> str:
        return normalize(f"{self.name} {self.artists}")

    @property
    def label(self) -> str:
        return f"{self.name}, de {self.artists}" if self.artists else self.name


@dataclass(frozen=True, slots=True)
class PlayOutcome:
    """Por que no sono algo, para poder decirlo con precision.

    Un unico `False` obligaba a responder "no pude reproducir X" en los tres
    casos, y son tres problemas distintos con tres soluciones distintas: abrir
    Spotify, decir otro nombre, o mirar el log.
    """

    ok: bool
    #: "" si ok. Si no: sin_spotify, sin_dispositivo, no_encontrado, error.
    reason: str = ""
    #: Que sono, para poder confirmarlo en voz alta cuando no es evidente.
    label: str = ""


_OK = PlayOutcome(True)


def _cache_path() -> Path:
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~/.cache")
    directory = Path(base) / "asistente-local"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / "spotify-token.json"


class SpotifyClient:
    """Envoltorio fino sobre spotipy con degradacion controlada.

    Ningun metodo lanza: devuelven bool. El asistente tiene que seguir vivo
    aunque la red se caiga a mitad de un comando.
    """

    def __init__(self, config: Config, secrets: Secrets) -> None:
        self._config = config.spotify
        self._client: Any | None = None
        self._secrets = secrets
        #: (nombre, uri) de tus playlists, y cuando se leyeron.
        self._playlists: tuple[tuple[str, str], ...] = ()
        self._playlists_at = 0.0
        #: (texto buscable, nombre, artistas, uri) de tus Me Gusta.
        self._liked: tuple[LikedTrack, ...] = ()
        self._liked_at = 0.0

    @property
    def available(self) -> bool:
        return self._config.enabled and bool(self._secrets.spotify_client_id)

    def connect(self) -> bool:
        """Crea el cliente. Abre el navegador solo la primera vez."""
        if not self.available:
            return False
        try:
            import spotipy
            from spotipy.oauth2 import SpotifyPKCE

            auth = SpotifyPKCE(
                client_id=self._secrets.spotify_client_id,
                redirect_uri=self._config.redirect_uri,
                scope=",".join(self._config.scopes),
                cache_handler=spotipy.CacheFileHandler(cache_path=str(_cache_path())),
                open_browser=True,
            )
            self._client = spotipy.Spotify(auth_manager=auth, requests_timeout=5)
        except Exception:
            log.exception("no se pudo conectar con Spotify")
            self._client = None
            return False
        return True

    def warmup(self) -> None:
        """Trae de tu cuenta lo que hace falta para buscar, al arrancar.

        Indexar los Me Gusta son hasta 20 peticiones y listar las playlists unas
        pocas mas. Pagarlo aqui y no en el primer "pon X" es la misma razon por
        la que se calientan Whisper, Piper y el LLM: el coste existe igual, y en
        mitad de un comando se oye.
        """
        if self._client is None:
            return
        started = time.perf_counter()
        liked, playlists = self.liked_tracks(force=True), self.own_playlists(force=True)
        log.info(
            "spotify: %d me gusta y %d playlists indexados en %.1f s",
            len(liked),
            len(playlists),
            time.perf_counter() - started,
        )

    def _active_device(self) -> str | None:
        if self._client is None:
            return None
        try:
            devices = self._client.devices().get("devices", [])
        except Exception:
            log.exception("no se pudieron listar los dispositivos de Spotify")
            return None
        for device in devices:
            if device.get("is_active"):
                return str(device["id"])
        # Sin dispositivo activo pero con alguno disponible: usar el primero
        # permite que "pon rock" funcione con Spotify abierto pero parado.
        return str(devices[0]["id"]) if devices else None

    # ------------------------------------------------------------- reproducir

    def play_query(
        self,
        query: str,
        artist: str | None = None,
        kind: Kind | None = None,
    ) -> PlayOutcome:
        """Busca y reproduce lo que mejor encaje con lo que se pidio.

        Con `kind` la frase ya dijo QUE es ("pon el album de X") y se busca solo
        eso. Sin `kind` se recorre una cascada, y el orden importa: "pon rock"
        casi siempre significa una lista, no la primera cancion titulada "Rock".

            tus playlists -> playlists publicas -> albumes -> canciones

        Tus playlists van primero porque son las unicas que puedes nombrar de
        memoria: si dices "pon mi playlist de gym" es porque existe y sabes como
        se llama, mientras que una publica que se llame igual es una casualidad.
        """
        if self._client is None:
            return PlayOutcome(False, "sin_spotify")
        if (device := self._active_device()) is None:
            return PlayOutcome(False, "sin_dispositivo")

        # TUS ME GUSTA PRIMERO, y no es solo una preferencia: es tambien donde
        # se absorbe lo que Whisper destroza. Un titulo en ingles dicho en
        # espanol sale transcrito como suena ("Ponlobes Rock" por "pon Lovers
        # Rock"), y con eso la Web API no encuentra nada — pero contra unos
        # cientos de canciones TUYAS, un emparejamiento difuso si acierta.
        pedido = f"{query} de {artist}" if artist else query
        if (mia := self.find_liked(query, artist)) is not None:
            try:
                self._client.start_playback(device_id=device, uris=[mia.uri])
            except Exception:
                log.exception("fallo la reproduccion de %r desde tus me gusta", pedido)
                return PlayOutcome(False, "error")
            log.info("%r -> me gusta %r", pedido, mia.label)
            return _OK

        # Con artista no hay ambiguedad posible: se pidio una cancion concreta.
        if artist:
            return self._play_track(device, query, artist)

        try:
            if kind in (None, "playlist") and (mia := self.find_own_playlist(query)) is not None:
                nombre, uri = mia
                self._client.start_playback(device_id=device, context_uri=uri)
                return PlayOutcome(True, label=nombre)

            for buscar, clave in self._search_order(kind):
                if (hallado := self._best_match(query, query, buscar, clave)) is None:
                    continue
                nombre, uri = hallado
                if buscar == "track":
                    self._client.start_playback(device_id=device, uris=[uri])
                else:
                    self._client.start_playback(device_id=device, context_uri=uri)
                log.info("%r -> %s %r", query, buscar, nombre)
                return _OK
        except Exception:
            log.exception("fallo la reproduccion de %r", query)
            return PlayOutcome(False, "error")
        return PlayOutcome(False, "no_encontrado")

    @staticmethod
    def _search_order(kind: Kind | None) -> tuple[tuple[str, str], ...]:
        """Tipos de la Web API a probar, en orden. `(type, clave del JSON)`."""
        por_tipo: dict[Kind, tuple[tuple[str, str], ...]] = {
            "playlist": (("playlist", "playlists"),),
            "album": (("album", "albums"),),
            "cancion": (("track", "tracks"),),
            "artista": (("artist", "artists"),),
        }
        if kind is not None:
            return por_tipo[kind]
        return (("playlist", "playlists"), ("album", "albums"), ("track", "tracks"))

    def _best_match(
        self,
        consulta: str,
        pedido: str,
        kind: str,
        key: str,
    ) -> tuple[str, str] | None:
        """`(nombre, uri)` del resultado que mejor cubre lo pedido, o None.

        `consulta` es lo que se le manda a la API (puede llevar filtros de
        campo) y `pedido` es lo que dijo el usuario, que es contra lo que hay
        que comprobar. No son lo mismo: `track:"la bamba" artist:"los lobos"` no
        se parece a nada escrito por una persona.

        Quedarse con el primer resultado era el fallo: `search()` siempre
        devuelve algo, asi que "loving machine de tv girl" acababa reproduciendo
        una playlist llamada "TV Girl" y pareciendo que habia funcionado.
        Devolver None y decir "no lo encontre" es mucho mas util que sonar algo
        que no se pidio.
        """
        assert self._client is not None
        results = self._client.search(q=consulta, type=kind, limit=_LIMITE_BUSQUEDA)
        items = (results or {}).get(key, {}).get("items") or []

        mejor: tuple[str, str] | None = None
        mejor_score = MIN_COBERTURA
        for item in items:
            if not item or not item.get("uri"):
                continue
            nombre = str(item.get("name", ""))
            # En una cancion, el artista es parte de lo que se pidio: sin el,
            # "la bamba de los lobos" solo cubriria la mitad de las palabras.
            artistas = " ".join(a.get("name", "") for a in item.get("artists") or [])
            if (score := cobertura(pedido, nombre, artistas)) >= mejor_score:
                mejor, mejor_score = (nombre, str(item["uri"])), score

        if mejor is None and items:
            log.debug("%r: %d resultados de tipo %s, ninguno cubre lo pedido",
                      pedido, len(items), kind)
        return mejor

    def _play_track(self, device: str, title: str, artist: str) -> PlayOutcome:
        """"Pon la cancion X de Y". Tres intentos, y el orden no es cosmetico.

        1. Filtros de campo (`track:"X" artist:"Y"`): lo mas preciso, y lo unico
           que impide que el nombre del grupo se trate como parte del titulo.
        2. Texto libre "X Y": mas tolerante cuando Whisper transcribe el nombre
           algo distinto de como esta escrito en Spotify.
        3. Texto libre "X de Y", la frase tal cual se dijo. Es la red de
           seguridad de la HIPOTESIS del router: partir por el ultimo "de"
           acierta casi siempre, pero un titulo como "el dia que me quieras de
           gardel" cantado por un grupo con "de" en el nombre se parte mal, y
           entonces lo unico que queda es buscar la frase entera.

        Los tres se comprueban contra lo que dijo el usuario: un resultado que
        no cubra sus palabras se descarta, aunque Spotify lo haya devuelto.
        """
        assert self._client is not None
        pedido = f"{title} {artist}"
        intentos = (f'track:"{title}" artist:"{artist}"', pedido, f"{title} de {artist}")
        try:
            for consulta in intentos:
                hallado = self._best_match(consulta, pedido, "track", "tracks")
                if hallado is None:
                    continue
                nombre, uri = hallado
                self._client.start_playback(device_id=device, uris=[uri])
                log.info("%r de %r -> %r", title, artist, nombre)
                return PlayOutcome(True, label=f"{title}, de {artist}")
        except Exception:
            log.exception("fallo la reproduccion de %r de %r", title, artist)
            return PlayOutcome(False, "error")
        return PlayOutcome(False, "no_encontrado")

    def play_liked(self) -> PlayOutcome:
        """Reproduce tus Me Gusta.

        No son una playlist y no tienen `context_uri`: hay que leer las
        canciones y pasarlas como lista de URIs. Se cogen las
        `LIKED_LIMIT` guardadas mas recientemente, que es lo que cabe en una
        peticion.
        """
        if self._client is None:
            return PlayOutcome(False, "sin_spotify")
        if (device := self._active_device()) is None:
            return PlayOutcome(False, "sin_dispositivo")

        try:
            saved = self._client.current_user_saved_tracks(limit=LIKED_LIMIT)
            uris = [
                str(item["track"]["uri"])
                for item in (saved or {}).get("items", [])
                if item.get("track") and item["track"].get("uri")
            ]
            if not uris:
                return PlayOutcome(False, "no_encontrado")
            self._client.start_playback(device_id=device, uris=uris)
        except Exception:
            log.exception("no se pudieron reproducir los me gusta")
            return PlayOutcome(False, "error")
        return PlayOutcome(True, label="tus me gusta")

    # ----------------------------------------------------------- tus me gusta

    def liked_tracks(self, force: bool = False) -> tuple[LikedTrack, ...]:
        """Tus Me Gusta como indice buscable, cacheados `_PLAYLISTS_TTL_S`.

        Se leen hasta `LIKED_INDEX_LIMIT` en paginas de 50. Es la parte cara de
        todo esto (hasta 20 peticiones), por eso `warmup()` la paga al arrancar
        y no en mitad de un comando.
        """
        if self._client is None:
            return ()
        fresco = time.monotonic() - self._liked_at < _PLAYLISTS_TTL_S
        if not force and self._liked and fresco:
            return self._liked

        encontradas: list[LikedTrack] = []
        try:
            for offset in range(0, LIKED_INDEX_LIMIT, 50):
                page = self._client.current_user_saved_tracks(limit=50, offset=offset)
                items = (page or {}).get("items") or []
                for item in items:
                    track = (item or {}).get("track") or {}
                    if not track.get("uri") or not track.get("name"):
                        continue
                    artistas = ", ".join(a.get("name", "") for a in track.get("artists") or [])
                    encontradas.append(
                        LikedTrack(
                            name=str(track["name"]),
                            artists=artistas,
                            uri=str(track["uri"]),
                        )
                    )
                if len(items) < 50:
                    break
        except Exception:
            log.exception("no se pudieron leer tus me gusta")
            return self._liked

        self._liked = tuple(encontradas)
        self._liked_at = time.monotonic()
        log.debug("me gusta indexados: %d", len(self._liked))
        return self._liked

    def find_liked(self, query: str, artist: str | None = None) -> LikedTrack | None:
        """La cancion de tus Me Gusta que mejor case con lo pedido, o None."""
        pedido = normalize(f"{query} {artist}" if artist else query)
        if not pedido:
            return None

        mejor: LikedTrack | None = None
        mejor_score = LIBRARY_THRESHOLD
        for track in self.liked_tracks():
            if (score := similitud(pedido, track.searchable)) >= mejor_score:
                mejor, mejor_score = track, score
        if mejor is not None:
            log.debug("me gusta %r -> %r (%.0f)", pedido, mejor.label, mejor_score)
        return mejor

    def hotwords(self, limit: int = 60) -> str:
        """Vocabulario de TU cuenta para sesgar el decodificador de Whisper.

        MEDIDO (24 clips de titulos en ingles dichos por voces en espanol,
        Whisper small en CPU, que es el caso que fallaba):

            sin hotwords, beam=1   WER 39.6%
            sin hotwords, beam=5   WER 38.4%   <- subir el beam NO sirve
            con hotwords, beam=1   WER 33.4%   al mismo coste

        Y en los titulos concretos la diferencia no es de grado:
            "Lovin Machine"                -> "Loving Machine"
            "Ponlobes Rock"                -> "Pon Lovers Rock"
            "no-tinelsmatters de metalica" -> "Nothing Else Matters de Metallica"

        Tiene sentido: el problema no es que Whisper oiga mal, es que "loving
        machine" no es una secuencia probable en espanol y "lovin machine" si.
        Decirle que existe basta para inclinar la balanza.

        Se priorizan los ARTISTAS sobre los titulos porque se repiten -veinte
        canciones de un grupo son un solo termino- y porque son el ancla que
        arrastra al resto de la frase.
        """
        artistas: list[str] = []
        titulos: list[str] = []
        for track in self.liked_tracks():
            for nombre in track.artists.split(", "):
                if nombre and nombre not in artistas:
                    artistas.append(nombre)
            if track.name not in titulos:
                titulos.append(track.name)

        terminos = artistas[:limit] + titulos[: max(0, limit - len(artistas[:limit]))]
        return ", ".join(terminos)

    # -------------------------------------------------------- tus playlists

    def own_playlists(self, force: bool = False) -> tuple[tuple[str, str], ...]:
        """`(nombre, uri)` de TUS playlists, cacheadas `_PLAYLISTS_TTL_S`.

        Pagina de 50 en 50 hasta agotarlas o llegar a 200, que es donde el coste
        de enumerar empieza a notarse en un comando hablado. Si tienes mas y la
        que buscas se queda fuera, el siguiente escalon de `play_query` la
        buscara en el catalogo publico.
        """
        if self._client is None:
            return ()
        fresco = time.monotonic() - self._playlists_at < _PLAYLISTS_TTL_S
        if not force and self._playlists and fresco:
            return self._playlists

        encontradas: list[tuple[str, str]] = []
        try:
            for offset in range(0, 200, 50):
                page = self._client.current_user_playlists(limit=50, offset=offset)
                items = (page or {}).get("items") or []
                for item in items:
                    if item and item.get("name") and item.get("uri"):
                        encontradas.append((str(item["name"]), str(item["uri"])))
                if len(items) < 50:
                    break
        except Exception:
            log.exception("no se pudieron listar tus playlists")
            return self._playlists

        self._playlists = tuple(encontradas)
        self._playlists_at = time.monotonic()
        log.debug("playlists propias: %d", len(self._playlists))
        return self._playlists

    def find_own_playlist(self, name: str) -> tuple[str, str] | None:
        """`(nombre, uri)` de tu playlist que mejor case, o None.

        Exacta primero y difusa despues, igual que la resolucion de apps: la
        difusa esta para absorber como transcribe Whisper ("gim" por "gym"), no
        para adivinar.
        """
        objetivo = normalize(name)
        if not objetivo:
            return None

        mejor: tuple[str, str] | None = None
        mejor_score = PLAYLIST_THRESHOLD
        for nombre, uri in self.own_playlists():
            candidato = normalize(nombre)
            if candidato == objetivo:
                return nombre, uri
            if (score := fuzz.token_set_ratio(objetivo, candidato)) >= mejor_score:
                mejor, mejor_score = (nombre, uri), score
        if mejor is not None:
            log.debug("playlist propia %r -> %r (%.0f)", name, mejor[0], mejor_score)
        return mejor

    def next_track(self) -> bool:
        return self._simple_action("next_track")

    def previous_track(self) -> bool:
        return self._simple_action("previous_track")

    def toggle_playback(self) -> bool:
        if self._client is None or (device := self._active_device()) is None:
            return False
        try:
            state = self._client.current_playback()
            if state is not None and state.get("is_playing"):
                self._client.pause_playback(device_id=device)
            else:
                self._client.start_playback(device_id=device)
        except Exception:
            log.exception("fallo el play/pause de Spotify")
            return False
        return True

    def current_track(self) -> tuple[str, str] | None:
        """(titulo, artistas) de lo que suena, o None si no suena nada."""
        if self._client is None:
            return None
        try:
            playing = self._client.current_user_playing_track()
        except Exception:
            log.exception("no se pudo consultar la cancion actual")
            return None

        item = (playing or {}).get("item")
        if not item:
            return None
        artistas = ", ".join(a["name"] for a in item.get("artists", [])) or "artista desconocido"
        return str(item.get("name", "")), artistas

    def save_current_track(self) -> str | None:
        """Guarda en favoritos lo que suena. Devuelve el titulo, o None.

        Necesita el permiso `user-library-modify`. Si autorizaste antes de que
        existiera este comando, hay que borrar el token cacheado para que
        Spotify vuelva a pedir permisos (ver la guia del README).
        """
        if self._client is None:
            return None
        try:
            playing = self._client.current_user_playing_track()
            item = (playing or {}).get("item")
            if not item:
                return None
            self._client.current_user_saved_tracks_add([item["id"]])
        except Exception:
            log.exception("no se pudo guardar la cancion")
            return None
        return str(item.get("name", ""))

    def volume_state(self) -> tuple[str, int] | None:
        """`(device_id, volumen)` del dispositivo activo, en UNA sola llamada.

        Es el volumen *del reproductor*: el mando que mueve la barra dentro de
        Spotify y que se sincroniza con el movil y con los altavoces de Connect.
        No tiene nada que ver con el del mezclador de Windows.

        Devuelve las dos cosas juntas a proposito. `current_playback()` ya trae
        el dispositivo entero, asi que preguntar por el volumen y luego por el
        dispositivo serian dos viajes de red para leer un unico objeto — y con
        la Web API en la ruta critica del comando, cada viaje se oye.
        """
        if self._client is None:
            return None
        try:
            state = self._client.current_playback()
        except Exception:
            log.exception("no se pudo leer el volumen de Spotify")
            return None

        device = (state or {}).get("device") or {}
        volume = device.get("volume_percent")
        if not device.get("id") or volume is None:
            return None
        return str(device["id"]), int(volume)

    def get_volume_percent(self) -> int | None:
        state = self.volume_state()
        return None if state is None else state[1]

    def set_volume_percent(self, level: int, device_id: str | None = None) -> bool:
        """Fija el volumen del reproductor. `level` se recorta a 0-100.

        `device_id` se pasa cuando quien llama ya lo sabe (porque acaba de leer
        el estado), y asi se ahorra la consulta de dispositivos.

        Spotify devuelve 403 en dispositivos que no admiten control de volumen
        (bastantes altavoces de Connect, y el reproductor web). Se traga la
        excepcion y devuelve False: quien llama decide que decir.
        """
        if self._client is None:
            return False
        device = device_id or self._active_device()
        if device is None:
            return False
        try:
            self._client.volume(max(0, min(100, level)), device_id=device)
        except Exception:
            log.exception("no se pudo fijar el volumen de Spotify")
            return False
        return True

    def set_shuffle(self, enabled: bool) -> bool:
        if self._client is None or (device := self._active_device()) is None:
            return False
        try:
            self._client.shuffle(enabled, device_id=device)
        except Exception:
            log.exception("no se pudo cambiar el modo aleatorio")
            return False
        return True

    def _simple_action(self, method: str) -> bool:
        if self._client is None or (device := self._active_device()) is None:
            return False
        try:
            getattr(self._client, method)(device_id=device)
        except Exception:
            log.exception("fallo %s en Spotify", method)
            return False
        return True


def _no_sono(outcome: PlayOutcome, que: str) -> SkillResult:
    """Traduce el motivo a algo que se pueda arreglar oyendolo.

    "No pude reproducir X" servia para los tres casos y no ayudaba en ninguno:
    quedarse sin dispositivo se arregla abriendo Spotify, y no encontrar algo,
    diciendo otro nombre.
    """
    mensajes = {
        "sin_spotify": "No estoy conectado a Spotify.",
        "sin_dispositivo": "No hay ningún Spotify abierto donde sonar. Ábrelo y te lo pongo.",
        "no_encontrado": f"No encontré {que} en Spotify.",
    }
    return SkillResult.failed(mensajes.get(outcome.reason, f"No pude reproducir {que}."))


class PlayArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=200)
    #: Nombre del grupo o cantante, cuando la frase lo dice: "pon la cancion X
    #: DE Y". Cambia la busqueda entera, no es un adorno.
    artist: str | None = Field(default=None, max_length=120)
    #: Que clase de cosa se pidio, tal como se dijo ("playlist", "disco"...).
    kind: str | None = Field(default=None, max_length=40)

    @field_validator("kind")
    @classmethod
    def _canonical_kind(cls, value: str | None) -> Kind | None:
        """Mapea la palabra hablada al tipo de la API; lo que no reconozca, None.

        None significa "decidelo tu con la cascada", que es un buen valor por
        defecto. Rechazar la frase entera por una palabra que no esta en la
        tabla seria convertir un matiz en un fallo.
        """
        if value is None:
            return None
        return _KINDS.get(normalize(value))


class SpotifyPlaySkill(Skill):
    name = "spotify.play"
    args_model = PlayArgs
    description = (
        "Reproduce en Spotify una playlist, album, artista o cancion por nombre. "
        "Usar para 'pon X', 'reproduce X', 'quiero escuchar X'. Si la frase dice "
        "de quien es la cancion, va en 'artist'."
    )

    def __init__(self, client: SpotifyClient) -> None:
        self._client = client

    def execute(self, args: BaseModel) -> SkillResult:
        assert isinstance(args, PlayArgs)
        kind: Kind | None = args.kind  # type: ignore[assignment]  # el validador ya lo acoto
        outcome = self._client.play_query(args.query, artist=args.artist, kind=kind)
        if not outcome.ok:
            que = f"{args.query} de {args.artist}" if args.artist else args.query
            return _no_sono(outcome, que)
        # Silencio. La cancion empezando a sonar ya dice que se acerto, y decirlo
        # ademas en voz alta se pisa con el principio de la propia cancion. Solo
        # se habla cuando algo NO sale: `outcome.label` sigue existiendo porque
        # es lo que se registra en el log.
        return SkillResult.silent()


class LikedSkill(Skill):
    name = "spotify.liked"
    description = (
        "Reproduce tus canciones con me gusta (favoritas, guardadas) de Spotify. "
        "Usar para 'pon mis me gusta', 'reproduce mis favoritas'."
    )

    def __init__(self, client: SpotifyClient) -> None:
        self._client = client

    def execute(self, args: BaseModel) -> SkillResult:
        outcome = self._client.play_liked()
        if not outcome.ok:
            return _no_sono(outcome, "tus me gusta")
        return SkillResult.silent()


class WhatSongSkill(Skill):
    name = "spotify.what_song"
    description = "Dice que cancion esta sonando ahora mismo."

    def __init__(self, client: SpotifyClient) -> None:
        self._client = client

    def execute(self, args: BaseModel) -> SkillResult:
        track = self._client.current_track()
        if track is None:
            return SkillResult.failed("No hay nada sonando en Spotify.")
        title, artist = track
        return SkillResult.says(f"{title}, de {artist}.")


class LikeSkill(Skill):
    name = "spotify.like"
    description = "Guarda la cancion actual en tus canciones favoritas de Spotify."

    def __init__(self, client: SpotifyClient) -> None:
        self._client = client

    def execute(self, args: BaseModel) -> SkillResult:
        track = self._client.save_current_track()
        if track is None:
            return SkillResult.failed("No pude guardar la canción.")
        return SkillResult.says(f"Guardada: {track}.")


class ShuffleArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    #: Viene de fixed_args del catalogo: hay un intent para activar y otro
    #: para desactivar, y asi la frase no tiene que llevar el valor.
    enabled: bool = True


class ShuffleSkill(Skill):
    name = "spotify.shuffle"
    args_model = ShuffleArgs
    description = "Activa o desactiva la reproduccion aleatoria."

    def __init__(self, client: SpotifyClient) -> None:
        self._client = client

    def execute(self, args: BaseModel) -> SkillResult:
        assert isinstance(args, ShuffleArgs)
        if not self._client.set_shuffle(args.enabled):
            return SkillResult.failed("No pude cambiar el modo aleatorio.")
        return SkillResult.says("Aleatorio activado." if args.enabled else "Aleatorio desactivado.")
