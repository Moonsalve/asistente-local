"""¿Y si en vez de corregir el titulo, lo buscamos dentro del artista?

POR QUE ESTA SONDA
------------------
Lo que hay hoy (`music_ai.py`) le pide al LLM que corrija {titulo, artista} y
acepta la correccion si se parece a lo que se oyo. **Medido en el PC, no
funciona**: "Nothing Matters" de Metallica (mal, es "Nothing ELSE Matters")
puntuo 87 y "Blinding Lights" (bien) puntuo 72.7. Una mala saco mas nota que
una buena, asi que ningun umbral las separa.

Pero las tandas anteriores enseñaron donde SI hay señal:

  - **El LLM acierta el ARTISTA 4 de 5 veces** y el titulo solo 2 de 5. El
    artista es un nombre que ha visto miles de veces; el titulo es lo que
    alucina.
  - **Spotify NO resuelve el artista fonetico**: "tibi guerl" -> TINI,
    "uiquen" -> ChocQuibTown. Solo acierta cuando Whisper ya lo escribio bien.
    Asi que el LLM no sobra: hace falta, pero para el artista.

La idea es usar el artista —la parte fiable— para acotar la busqueda, y dejar
que el titulo se resuelva dentro de ese espacio pequeño y real.

PRIMERO SE MIDE QUE PUEDE HACER ESTA APLICACION
------------------------------------------------
Dos intentos anteriores de esta sonda murieron por dar por supuesto lo que la
API permite: `artist_top_tracks` devuelve 403 con esta aplicacion, y `search`
rechaza `limit=50` **y tambien `limit=20`** con HTTP 400 "Invalid limit",
aunque la documentacion de Spotify diga que el maximo es 50 y aunque
`current_user_saved_tracks(limit=50)` si funcione.

Asi que ya no se supone nada: al arrancar se prueba de verdad que endpoints
responden y cual es el `limit` mas alto que acepta la busqueda, y el resto de
la sonda usa lo que haya salido. La tabla que imprime es en si misma un
resultado, y vale para todo el proyecto.

LAS VIAS QUE SE COMPARAN
------------------------
    A) HOY        el LLM corrige titulo+artista, guardia, y se busca
    B) CATALOGO   se traen las canciones del artista y se empareja en local
    C) SIN LLM    Spotify resuelve el artista el solo (ya medido: se cae)
    D) UNA SOLA   artista del LLM + titulo TAL COMO SE OYO, en una peticion

**D es la nueva candidata y la barata.** En vez de traerse el catalogo entero
para emparejar en casa, se le pasa a Spotify el artista como filtro de campo y
el titulo destrozado como texto libre: `artist:"TV Girl" ponlobes rock`. El
filtro reduce el espacio a las canciones de ese artista y la busqueda difusa
solo tiene que acertar dentro de ahi. Una peticion en vez de varias paginas.

Si D acierta tanto como B, gana D: menos peticiones, menos codigo y ninguna
lista que mantener en memoria.

Uso (en el PC, con Ollama y Spotify en marcha):

    python scripts/probe_artist_catalog.py
    python scripts/probe_artist_catalog.py "ponlobes rock de tv girl"

**Pon canciones TUYAS, y sobre todo alguna que NO sea un exito del artista**:
ahi es donde se vera si el catalogo llega o se queda corto.

No reproduce nada: solo busca. Y no decide: enseña y decides tu.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asistente.config import Config, Secrets  # noqa: E402
from asistente.router.text import normalize  # noqa: E402
from asistente.skills.music_ai import MusicQuery, build_resolver  # noqa: E402
from asistente.skills.spotify import SpotifyClient, similitud  # noqa: E402

MAL = " MAL "

#: Limites de busqueda que se prueban, de menor a mayor. El mayor que responda
#: es el que se usa para paginar. No se supone ninguno: se mide.
_ESCALERA = (1, 3, 5, 10, 20, 50)

#: Techo de canciones que se traen por artista. Con un `limit` pequeño esto son
#: muchas peticiones, y ese coste es parte de la decision: se cuenta y se
#: enseña. Si hacen falta diez peticiones, la via B no vale para voz.
_MAX_CANCIONES = 60

#: Cuantas peticiones de catalogo se toleran antes de rendirse. Un comando de
#: voz no puede pagar mas.
_MAX_PETICIONES = 6

EJEMPLOS = (
    "ponlobes rock de tv girl",
    "lovin machin de tibi guerl",
    "no tinelsmatters de metalica",
    "blain ding lights de uiquen",
    "esmels laik tin espirit de nirvana",
)


def _http(exc: Exception) -> str:
    """El codigo HTTP de un fallo de spotipy, sin las cuarenta lineas de URL."""
    codigo = getattr(exc, "http_status", None)
    crudo = str(getattr(exc, "msg", "") or exc).replace("\n", " ")
    razon = " ".join(p for p in crudo.split() if "://" not in p)
    razon = razon.removesuffix(", reason: None").strip(" -:")
    return f"HTTP {codigo}: {razon[:50]}" if codigo else razon[:70]


def _partir(frase: str) -> tuple[str, str | None]:
    """Parte "X de Y" por el ULTIMO "de", igual que hace el router."""
    if " de " in frase:
        titulo, _, artista = frase.rpartition(" de ")
        return titulo.strip(), artista.strip()
    return frase.strip(), None


def _pista(item: dict[str, Any]) -> str:
    artistas = ", ".join(a.get("name", "") for a in item.get("artists") or [])
    return f"{item.get('name', '?')} — {artistas}"


def _items(res: Any, clave: str = "tracks") -> list[dict[str, Any]]:
    return (res or {}).get(clave, {}).get("items") or []


# --------------------------------------------------------- que permite la API


def _sondear_api(raw: Any) -> dict[str, Any]:
    """Que puede hacer de verdad esta aplicacion. Medido, no supuesto.

    Existe porque dos versiones de esta sonda murieron dando por buena la
    documentacion de Spotify. Cuesta menos de diez peticiones, una vez.
    """
    print("-" * 78)
    print("QUE PERMITE ESTA APLICACION (medido ahora mismo)")
    print("-" * 78)

    capacidades: dict[str, Any] = {"search_limit": 0}

    for limite in _ESCALERA:
        try:
            raw.search(q="nirvana", type="track", limit=limite)
        except Exception as exc:
            print(f"  search limit={limite:<3} MAL   {_http(exc)}")
            break
        capacidades["search_limit"] = limite
        print(f"  search limit={limite:<3} OK")

    if not capacidades["search_limit"]:
        return capacidades

    # Un artista cualquiera con el que probar los endpoints de artista.
    artistas = _items(raw.search(q="nirvana", type="artist", limit=1), "artists")
    if not artistas:
        return capacidades
    artist_id = str(artistas[0]["id"])

    for nombre, llamada in (
        ("artist_top_tracks", lambda: raw.artist_top_tracks(artist_id)),
        ("artist_albums", lambda: raw.artist_albums(artist_id, limit=_maximo(capacidades))),
    ):
        try:
            llamada()
        except Exception as exc:
            capacidades[nombre] = False
            print(f"  {nombre:<20} MAL   {_http(exc)}")
        else:
            capacidades[nombre] = True
            print(f"  {nombre:<20} OK")

    print()
    return capacidades


def _maximo(capacidades: dict[str, Any]) -> int:
    return int(capacidades.get("search_limit") or 1)


# ------------------------------------------------------------ las cuatro vias


def _resolver_artista(raw: Any, nombre: str) -> tuple[str, str] | None:
    """`(id, nombre real)` del artista que Spotify cree que es ese, o None."""
    try:
        items = _items(raw.search(q=nombre, type="artist", limit=1), "artists")
    except Exception as exc:
        print(f"{MAL} error buscando el artista: {_http(exc)}")
        return None
    if not items:
        return None
    return str(items[0]["id"]), str(items[0].get("name", "?"))


def _cabecera_artista(pedido: str, real: str) -> bool:
    """Enseña a que artista se ha ido Spotify. Devuelve si es creible.

    Va ANTES que la cancion a proposito: si el artista esta mal, lo de abajo ya
    no significa nada. Es donde se cae la via C y con el nombre al final costaba
    verlo.
    """
    parecido = similitud(normalize(pedido), normalize(real))
    vale = parecido >= 72
    print(f"                      pidio {pedido!r} {'=' if vale else '!'}> "
          f"Spotify da {real!r}  (parecido {parecido:.0f})")
    if not vale:
        print("                      ^^ OJO: no es el mismo artista.")
    return vale


def _ranking(oido: str, pistas: list[dict[str, Any]], cuantas: int = 3) -> None:
    """Las mejores coincidencias con el titulo oido, para poder juzgarlas."""
    puntuadas = sorted(
        ((similitud(normalize(oido), normalize(str(t.get("name", "")))), t) for t in pistas),
        key=lambda par: par[0],
        reverse=True,
    )
    for i, (score, t) in enumerate(puntuadas[:cuantas]):
        flecha = "->" if i == 0 else f"{i + 1}a:"
        print(f"                      {flecha} {_pista(t)}   ({score:.1f})")
    if len(puntuadas) > 1:
        print(f"                      margen {puntuadas[0][0] - puntuadas[1][0]:+.1f}")


def _via_catalogo(raw: Any, titulo: str, artista: str, cap: dict[str, Any]) -> None:
    """B) Traerse las canciones del artista y emparejar en local."""
    print("     B) CATALOGO")
    inicio = time.perf_counter()
    resuelto = _resolver_artista(raw, artista)
    if resuelto is None:
        print(f"                      Spotify no reconoce a {artista!r}")
        return
    artist_id, real = resuelto
    _cabecera_artista(artista, real)

    encontradas: dict[str, dict[str, Any]] = {}
    avisos: list[str] = []
    peticiones = 1
    limite = _maximo(cap)

    if cap.get("artist_top_tracks"):
        peticiones += 1
        try:
            for t in raw.artist_top_tracks(artist_id).get("tracks", []):
                encontradas.setdefault(normalize(str(t.get("name", ""))), t)
        except Exception as exc:
            avisos.append(f"sin exitos ({_http(exc)})")

    for offset in range(0, _MAX_CANCIONES, limite):
        if peticiones >= _MAX_PETICIONES:
            avisos.append(f"cortado en {_MAX_PETICIONES} peticiones: ya es demasiado para voz")
            break
        try:
            peticiones += 1
            lote = _items(
                raw.search(q=f'artist:"{real}"', type="track", limit=limite, offset=offset)
            )
        except Exception as exc:
            avisos.append(f"busqueda cortada en offset {offset} ({_http(exc)})")
            break
        for t in lote:
            encontradas.setdefault(normalize(str(t.get("name", ""))), t)
        if len(lote) < limite:
            break

    dt = time.perf_counter() - inicio
    for aviso in avisos:
        print(f"                      ({aviso})")
    if not encontradas:
        print(f"                      sin canciones ({peticiones} peticiones, {dt:.2f} s)")
        return
    print(f"                      {len(encontradas)} canciones en "
          f"{peticiones} peticiones, {dt:.2f} s")
    _ranking(titulo, list(encontradas.values()))


def _via_una_busqueda(raw: Any, titulo: str, artista: str, cap: dict[str, Any]) -> None:
    """D) El artista como filtro y el titulo destrozado como texto libre.

    LA CANDIDATA BARATA. El filtro `artist:"..."` reduce el espacio a las
    canciones de ese artista, y la busqueda difusa de Spotify solo tiene que
    acertar dentro de ahi. Una peticion, sin traerse nada a memoria.
    """
    print("     D) UNA SOLA BUSQUEDA")
    inicio = time.perf_counter()
    consulta = f'artist:"{artista}" {titulo}'
    try:
        lote = _items(raw.search(q=consulta, type="track", limit=_maximo(cap)))
    except Exception as exc:
        print(f"                      {MAL} {_http(exc)}")
        return
    dt = time.perf_counter() - inicio
    print(f"                      q={consulta!r}  ({dt:.2f} s, 1 peticion)")
    if not lote:
        print("                      (nada)")
        return
    _ranking(titulo, lote)


def main() -> int:
    parser = argparse.ArgumentParser(description="¿Como se resuelve mejor un titulo mal oido?")
    parser.add_argument("frases", nargs="*", help="transcripciones tal como las oye Whisper")
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.ERROR)
    # spotipy registra cada 4xx con la URL entera y ocupa media pantalla.
    logging.getLogger("spotipy.client").setLevel(logging.CRITICAL)

    config = Config.load(args.config)
    frases = list(args.frases) or list(EJEMPLOS)

    print("=" * 78)
    print("¿COMO SE RESUELVE MEJOR UN TITULO MAL OIDO?")
    print("=" * 78)

    spotify = SpotifyClient(config, Secrets())
    if not spotify.available or not spotify.connect():
        print(f"\n{MAL} sin conexion con Spotify. Ejecuta antes diagnose_spotify.py")
        return 1
    raw = spotify._client  # sonda: se mira por dentro a proposito

    resolver = build_resolver(config.llm)
    if resolver is None:
        print(f"\n{MAL} sin Ollama. Las vias A, B y D necesitan el modelo.")
        return 1

    print(f"\nmodelo: {config.llm.model}    frases: {len(frases)}\n")

    cap = _sondear_api(raw)
    if not cap["search_limit"]:
        print(f"{MAL} la busqueda no funciona con ningun limite. Nada que medir.")
        return 1

    for frase in frases:
        titulo, artista_oido = _partir(frase)
        print("-" * 78)
        print(f"OIDO:  {frase!r}")

        if artista_oido is None:
            print("       (no se dijo artista: estas vias no aplican)\n")
            continue

        # --- A) lo que hace el asistente hoy
        inicio = time.perf_counter()
        corregido = resolver.resolve(titulo, artista_oido)
        dt = time.perf_counter() - inicio
        print("     A) HOY")
        if corregido is None:
            print(f"                      descartada por el guardia ({dt:.2f} s) -> no suena nada")
        else:
            consulta = (
                f'track:"{corregido.titulo}" artist:"{corregido.artista}"'
                if corregido.artista else corregido.titulo
            )
            try:
                lote = _items(raw.search(q=consulta, type="track", limit=1))
            except Exception as exc:
                lote = []
                print(f"                      ({_http(exc)})")
            print(f"                      el LLM dice {corregido.titulo!r} de "
                  f"{corregido.artista!r} ({dt:.2f} s)")
            print(f"                      -> {_pista(lote[0]) if lote else '(nada)'}")

        # El artista segun el LLM, que es la parte que acierta. Si el guardia
        # tiro la correccion entera, se le vuelve a preguntar sin filtrar.
        artista_llm = corregido.artista if corregido else ""
        if not artista_llm:
            crudo = resolver._ask(f"{titulo} de {artista_oido}")  # sonda
            try:
                artista_llm = MusicQuery.model_validate(crudo or {}).artista
            except Exception:
                artista_llm = ""

        if artista_llm:
            _via_catalogo(raw, titulo, artista_llm, cap)
            _via_una_busqueda(raw, titulo, artista_llm, cap)
        else:
            print("     B) CATALOGO          el LLM no dio artista")
            print("     D) UNA SOLA BUSQUEDA el LLM no dio artista")

        # --- C) sin LLM, con el artista tal como se oyo
        print("     C) SIN LLM")
        resuelto = _resolver_artista(raw, artista_oido)
        if resuelto is None:
            print(f"                      Spotify no reconoce a {artista_oido!r}")
        else:
            _cabecera_artista(artista_oido, resuelto[1])
        print()

    print("=" * 78)
    print("QUE MIRAR — los TITULOS, no las notas:")
    print()
    print("  1. EL ARTISTA PRIMERO. Si sale '!>', Spotify se fue a otro y lo de")
    print("     debajo no significa nada. Ahi se cae C, ya medido.")
    print()
    print("  2. ¿B o D aciertan donde A falla? Entonces acotar por artista es")
    print("     mejor que fiarse del titulo que da el LLM.")
    print()
    print("  3. ¿D acierta tanto como B? GANA D: una peticion en vez de varias")
    print("     paginas, y nada que traerse a memoria. Mira las dos cifras de")
    print("     peticiones antes de decidir.")
    print()
    print("  4. Mira el MARGEN con la segunda. Si la buena gana por poco, un")
    print("     umbral no las separara de forma fiable y estariamos repitiendo")
    print("     el error del guardia de verosimilitud.")
    print()
    print("  5. ¿Aparece el titulo bueno? Si no esta en la lista, no hay")
    print("     emparejamiento que valga.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
