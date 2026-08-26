"""¿Y si en vez de corregir el titulo, buscamos en el catalogo del artista?

POR QUE ESTA SONDA
------------------
Lo que hay hoy (`music_ai.py`) le pide al LLM que corrija {titulo, artista} y
acepta la correccion si se parece a lo que se oyo. **Medido en el PC, no
funciona**: una respuesta equivocada ("Nothing Matters" de Metallica, cuando es
"Nothing ELSE Matters") puntuo 87, y una correcta ("Blinding Lights") puntuo
72.7. Una mala saco mas nota que una buena, asi que ningun umbral las separa.

Pero esa misma tanda enseño donde SI hay señal: el modelo acerto **el artista 4
de 5 veces** y **el titulo solo 2 de 5**. El artista es la parte fiable.

La idea a medir: usar el artista para traer sus canciones DE VERDAD desde
Spotify, y emparejar contra esa lista el titulo tal como se oyo. Es la misma
situacion en la que `similitud` si esta validada —texto destrozado contra un
conjunto real y pequeño de titulos bien escritos, igual que tus me gusta, donde
el peor acierto puntua 91.9 y el mejor fallo 65.0— en vez de contra lo que el
modelo se imagine.

LAS TRES VIAS QUE SE COMPARAN, y la tercera es la importante
------------------------------------------------------------
    A) HOY          el LLM corrige titulo+artista, guardia, y se busca
    B) LLM+CATALOGO el LLM da SOLO el artista; el titulo sale de sus canciones
    C) SIN LLM      Spotify resuelve el artista el solo; el titulo, igual

**Si C acierta tanto como B, el LLM sobra en esta capa.** La vez pasada no se
hizo esta pregunta y se implemento de mas.

LO QUE YA SE MIDIO EN EL PC (2026-08-26, primera tanda)
--------------------------------------------------------
**La via C se cae, y se cae en el artista.** La busqueda de Spotify no aguanta
la fonetica espanola:

    "tibi guerl" -> TINI            (era TV Girl)
    "uiquen"     -> ChocQuibTown    (era The Weeknd)
    "metalica"   -> Metallica       correcto
    "nirvana"    -> Nirvana         correcto

Solo acierta cuando Whisper ya habia escrito bien el nombre. Asi que el LLM NO
sobra: hace falta para el artista. La via B lo acerto 4 de 5, y el unico fallo
fue donde el modelo hizo eco ("Tibi Guerl").

Y una sorpresa a favor de lo que ya hay: en la via A, el LLM dijo "Nothing
Matters" de Metallica —mal, es "Nothing ELSE Matters"— y la busqueda con
filtros de campo mas la verificacion de cobertura **devolvieron la cancion
correcta igualmente**. La red de abajo atrapa parte de lo que el guardia deja
pasar.

Queda por medir lo unico que no se pudo por dos fallos de esta sonda (403 en
`artist_top_tracks` y HTTP 400 por pedir `limit=50`): **si el titulo bueno
aparece entre las canciones que se traen del artista**, y a cuantas peticiones.

Uso (en el PC, con Ollama y Spotify en marcha):

    python scripts/probe_artist_catalog.py
    python scripts/probe_artist_catalog.py "ponlobes rock de tv girl"

**Pon canciones TUYAS, y sobre todo alguna que NO tengas en me gusta**, que es
el caso que todo esto existe para arreglar. Saca las frases del log del
asistente con `-v`: interesan como las oye Whisper, no como se escriben.

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

#: Tamano de pagina de la busqueda por artista.
#:
#: 20 y no 50: MEDIDO en el PC el 2026-08-26, `limit=50` devuelve HTTP 400
#: "Invalid limit" pese a que la documentacion de Spotify da 50 como maximo del
#: endpoint de busqueda. 20 es el valor por defecto y funciona.
_PAGINA = 20

#: Techo de canciones que se traen por artista. Tres peticiones en el peor caso.
#: Es una sonda: aqui interesa saber si el titulo bueno APARECE, y con cuantas
#: peticiones hay que pagarlo. Ese coste es parte de la decision.
_MAX_CANCIONES = 60

#: Los exitos del artista serian una sola peticion y cubririan lo que la gente
#: pide normalmente, pero MEDIDO en el PC el 2026-08-26 el endpoint
#: `/v1/artists/{id}/top-tracks` devuelve **403 Forbidden** para todos los
#: artistas con esta aplicacion. Se sigue intentando —no cuesta nada y en otras
#: cuentas funciona— pero ya no se depende de el.
_TOP = 10

EJEMPLOS = (
    "ponlobes rock de tv girl",
    "lovin machin de tibi guerl",
    "no tinelsmatters de metalica",
    "blain ding lights de uiquen",
    "esmels laik tin espirit de nirvana",
)


def _partir(frase: str) -> tuple[str, str | None]:
    """Parte "X de Y" por el ULTIMO "de", igual que hace el router."""
    if " de " in frase:
        titulo, _, artista = frase.rpartition(" de ")
        return titulo.strip(), artista.strip()
    return frase.strip(), None


def _pista(item: dict[str, Any]) -> str:
    artistas = ", ".join(a.get("name", "") for a in item.get("artists") or [])
    return f"{item.get('name', '?')} — {artistas}"


def _resolver_artista(raw: Any, nombre: str) -> tuple[str, str] | None:
    """`(id, nombre real)` del artista que Spotify cree que es ese, o None.

    Aqui es donde se ve si la busqueda difusa de Spotify aguanta un nombre
    destrozado: "tibi guerl" -> TV Girl, o nada.
    """
    try:
        res = raw.search(q=nombre, type="artist", limit=1)
    except Exception as exc:
        print(f"{MAL} error buscando el artista: {exc}")
        return None
    items = (res or {}).get("artists", {}).get("items") or []
    if not items:
        return None
    return str(items[0]["id"]), str(items[0].get("name", "?"))


def _http(exc: Exception) -> str:
    """El codigo HTTP de un fallo de spotipy, sin las cuarenta lineas de URL.

    El mensaje de spotipy mete la URL entera con sus parametros y ocupa media
    pantalla; lo unico accionable es el codigo y la razon del final.
    """
    codigo = getattr(exc, "http_status", None)
    crudo = str(getattr(exc, "msg", "") or exc).replace("\n", " ")
    # Fuera la URL (es la mitad del mensaje) y la coletilla vacia del final.
    razon = " ".join(p for p in crudo.split() if "://" not in p)
    razon = razon.removesuffix(", reason: None").strip(" -:")
    return f"HTTP {codigo}: {razon[:50]}" if codigo else razon[:70]


def _canciones_del_artista(
    raw: Any, artist_id: str, nombre: str
) -> tuple[list[dict[str, Any]], list[str], int]:
    """`(canciones, avisos, peticiones)` de ese artista, deduplicadas por nombre.

    Si el titulo bueno no esta en esta lista, el emparejamiento no puede
    acertar. Por eso interesa tanto CUANTAS se traen como a que precio: cada
    pagina es una peticion en mitad de un comando de voz.
    """
    encontradas: dict[str, dict[str, Any]] = {}
    avisos: list[str] = []
    peticiones = 0

    # Los exitos serian la fuente barata (una peticion), pero devuelven 403 con
    # esta aplicacion. Se intenta igual: si algun dia vuelve, sale gratis.
    try:
        peticiones += 1
        for t in raw.artist_top_tracks(artist_id).get("tracks", [])[:_TOP]:
            encontradas.setdefault(normalize(str(t.get("name", ""))), t)
    except Exception as exc:
        avisos.append(f"sin exitos del artista ({_http(exc)})")

    for offset in range(0, _MAX_CANCIONES, _PAGINA):
        try:
            peticiones += 1
            res = raw.search(
                q=f'artist:"{nombre}"', type="track", limit=_PAGINA, offset=offset
            )
        except Exception as exc:
            avisos.append(f"busqueda cortada en offset {offset} ({_http(exc)})")
            break
        items = (res or {}).get("tracks", {}).get("items") or []
        for t in items:
            encontradas.setdefault(normalize(str(t.get("name", ""))), t)
        if len(items) < _PAGINA:
            break

    return list(encontradas.values()), avisos, peticiones


def _emparejar(oido: str, pistas: list[dict[str, Any]]) -> list[tuple[float, dict[str, Any]]]:
    """Las canciones del artista ordenadas por parecido con el titulo oido."""
    puntuadas = [
        (similitud(normalize(oido), normalize(str(t.get("name", "")))), t) for t in pistas
    ]
    puntuadas.sort(key=lambda par: par[0], reverse=True)
    return puntuadas


def _via_catalogo(raw: Any, titulo_oido: str, artista: str, etiqueta: str) -> None:
    """Imprime que sale de emparejar el titulo oido contra el catalogo real.

    Se enseña el artista que sale ANTES que la cancion, y a proposito: si el
    artista esta mal, lo de abajo ya no significa nada. MEDIDO en el PC, ahi es
    donde se cae la via sin LLM ("tibi guerl" -> TINI, "uiquen" ->
    ChocQuibTown), y con el nombre escondido al final costaba verlo.
    """
    started = time.perf_counter()
    resuelto = _resolver_artista(raw, artista)
    if resuelto is None:
        print(f"     {etiqueta:16} Spotify no reconoce a {artista!r}")
        return
    artist_id, nombre_real = resuelto

    # ¿Es el mismo nombre que se le pidio, o Spotify se ha ido a otro artista?
    parecido = similitud(normalize(artista), normalize(nombre_real))
    sello = "=" if parecido >= 72 else "!"
    print(f"     {etiqueta:16} pidio {artista!r} {sello}> Spotify da {nombre_real!r}"
          f"  (parecido {parecido:.0f})")
    if sello == "!":
        print("                      ^^ OJO: no es el mismo artista. Lo de abajo sobra.")

    pistas, avisos, peticiones = _canciones_del_artista(raw, artist_id, nombre_real)
    # +1 por resolver el artista: el coste que cuenta es el del comando entero,
    # no el de una de sus mitades.
    peticiones += 1
    dt = time.perf_counter() - started
    for aviso in avisos:
        print(f"                      ({aviso})")

    if not pistas:
        print(f"                      sin canciones ({peticiones} peticiones, {dt:.2f} s)")
        return

    ranking = _emparejar(titulo_oido, pistas)
    mejor, segunda = ranking[0], (ranking[1] if len(ranking) > 1 else None)

    print(f"                      {len(pistas)} canciones en "
          f"{peticiones} peticiones, {dt:.2f} s")
    print(f"                      -> {_pista(mejor[1])}   ({mejor[0]:.1f})")
    if segunda:
        print(f"                      2a: {_pista(segunda[1])}   ({segunda[0]:.1f})"
              f"   margen {mejor[0] - segunda[0]:+.1f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="¿Sirve el catalogo del artista?")
    parser.add_argument("frases", nargs="*", help="transcripciones tal como las oye Whisper")
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.ERROR)
    # spotipy registra cada 4xx con la URL entera y ocupa media pantalla. Los
    # fallos que importan aqui ya se imprimen resumidos con `_http`.
    logging.getLogger("spotipy.client").setLevel(logging.CRITICAL)

    config = Config.load(args.config)
    frases = list(args.frases) or list(EJEMPLOS)

    print("=" * 78)
    print("¿SIRVE BUSCAR EN EL CATALOGO DEL ARTISTA?")
    print("=" * 78)

    spotify = SpotifyClient(config, Secrets())
    if not spotify.available or not spotify.connect():
        print(f"\n{MAL} sin conexion con Spotify. Ejecuta antes diagnose_spotify.py")
        return 1
    raw = spotify._client  # sonda: se mira por dentro a proposito

    resolver = build_resolver(config.llm)
    if resolver is None:
        print(f"\n{MAL} sin Ollama. Las vias A y B necesitan el modelo.")
        return 1

    print(f"\nmodelo: {config.llm.model}    frases: {len(frases)}\n")

    for frase in frases:
        titulo_oido, artista_oido = _partir(frase)
        print("-" * 78)
        print(f"OIDO:  {frase!r}")

        if artista_oido is None:
            print("       (no se dijo artista: este enfoque no aplica aqui)\n")
            continue

        # --- A) lo que hace el asistente hoy
        started = time.perf_counter()
        corregido = resolver.resolve(titulo_oido, artista_oido)
        dt = time.perf_counter() - started
        if corregido is None:
            print(f"     A) HOY           descartada por el guardia ({dt:.2f} s) -> no suena nada")
        else:
            consulta = (
                f'track:"{corregido.titulo}" artist:"{corregido.artista}"'
                if corregido.artista else corregido.titulo
            )
            try:
                res = raw.search(q=consulta, type="track", limit=1)
                items = (res or {}).get("tracks", {}).get("items") or []
            except Exception as exc:
                items = []
                print(f"       (error: {exc})")
            hallado = _pista(items[0]) if items else "(nada)"
            print(f"     A) HOY           el LLM dice {corregido.titulo!r} de "
                  f"{corregido.artista!r} ({dt:.2f} s)")
            print(f"                      -> {hallado}")

        # --- B) el LLM solo para el artista, el titulo del catalogo real
        artista_llm = corregido.artista if corregido else ""
        if not artista_llm:
            # El guardia tiro la correccion entera, pero el artista puede seguir
            # siendo bueno: se vuelve a pedir sin filtrar nada.
            crudo = resolver._ask(f"{titulo_oido} de {artista_oido}")  # sonda
            try:
                artista_llm = MusicQuery.model_validate(crudo or {}).artista
            except Exception:
                artista_llm = ""
        if artista_llm:
            _via_catalogo(raw, titulo_oido, artista_llm, "B) LLM+CATALOGO")
        else:
            print("     B) LLM+CATALOGO  el LLM no dio artista")

        # --- C) sin LLM: que Spotify resuelva el artista tal como se oyo
        _via_catalogo(raw, titulo_oido, artista_oido, "C) SIN LLM")
        print()

    print("=" * 78)
    print("QUE MIRAR — y hay que mirar los TITULOS, no las notas:")
    print()
    print("  1. EMPIEZA POR EL ARTISTA. Si la linea dice '!>', Spotify se fue a")
    print("     otro artista y la cancion de debajo no significa nada. Es donde")
    print("     se cayo la via C en la primera tanda: 'tibi guerl' -> TINI y")
    print("     'uiquen' -> ChocQuibTown.")
    print()
    print("  2. ¿B acierta donde A falla? Entonces vale la pena sacar el titulo")
    print("     del catalogo del artista en vez de fiarse del que da el LLM.")
    print()
    print("  3. ¿C acierta tanto como B? Seria la mejor noticia posible —el LLM")
    print("     sobraria en esta capa y se podria volver al 3B—, pero la primera")
    print("     tanda dice que no: la busqueda de Spotify no aguanta la fonetica")
    print("     espanola y solo acierta cuando Whisper ya habia escrito bien el")
    print("     nombre. Confirmalo con TUS artistas antes de darlo por cerrado.")
    print()
    print("  4. Mira el MARGEN con la segunda. Si la buena gana por poco, un")
    print("     umbral no va a separarlas de forma fiable y estariamos repitiendo")
    print("     el error del guardia de verosimilitud.")
    print()
    print("  5. ¿Sale el titulo bueno en la lista del artista? Si no aparece, no")
    print("     hay emparejamiento que valga: habria que traer mas canciones.")
    print()
    print("  6. Ojo a las PETICIONES y a los segundos: se suman al comando. Si")
    print("     hacen falta tres paginas para encontrar el titulo, eso pesa tanto")
    print("     como el LLM en la decision.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
