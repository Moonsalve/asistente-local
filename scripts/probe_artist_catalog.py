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

**Si C acierta tanto como B, el LLM sobra en esta capa.** La busqueda de
Spotify ya es difusa, y puede que "tibi guerl" le lleve a TV Girl sin ayuda. Ese
seria el mejor resultado posible: cero latencia, cero VRAM, menos codigo. La vez
pasada no se hizo esta pregunta y se implemento de mas.

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

#: Cuantas canciones del artista se traen para emparejar. Dos llamadas: sus
#: exitos (10, una peticion) mas una busqueda filtrada por artista. Se mide
#: cuanto cuesta, que es parte de la decision.
_TOP = 10
_POR_ARTISTA = 50

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


def _canciones_del_artista(raw: Any, artist_id: str, nombre: str) -> list[dict[str, Any]]:
    """Las canciones de ese artista, deduplicadas por nombre.

    Dos fuentes porque una sola no basta: sus exitos cubren lo que la gente pide
    normalmente, y la busqueda filtrada llega a lo que no es un exito. Si el
    titulo bueno no esta en esta lista, el emparejamiento no puede acertar — y
    eso es justo lo que hay que ver aqui.
    """
    encontradas: dict[str, dict[str, Any]] = {}
    try:
        top = raw.artist_top_tracks(artist_id).get("tracks", [])[:_TOP]
        for t in top:
            encontradas.setdefault(normalize(str(t.get("name", ""))), t)
    except Exception as exc:
        print(f"       (sin exitos del artista: {exc})")
    try:
        res = raw.search(q=f'artist:"{nombre}"', type="track", limit=_POR_ARTISTA)
        for t in (res or {}).get("tracks", {}).get("items") or []:
            encontradas.setdefault(normalize(str(t.get("name", ""))), t)
    except Exception as exc:
        print(f"       (sin busqueda por artista: {exc})")
    return list(encontradas.values())


def _emparejar(oido: str, pistas: list[dict[str, Any]]) -> list[tuple[float, dict[str, Any]]]:
    """Las canciones del artista ordenadas por parecido con el titulo oido."""
    puntuadas = [
        (similitud(normalize(oido), normalize(str(t.get("name", "")))), t) for t in pistas
    ]
    puntuadas.sort(key=lambda par: par[0], reverse=True)
    return puntuadas


def _via_catalogo(raw: Any, titulo_oido: str, artista: str, etiqueta: str) -> None:
    """Imprime que sale de emparejar el titulo oido contra el catalogo real."""
    started = time.perf_counter()
    resuelto = _resolver_artista(raw, artista)
    if resuelto is None:
        print(f"     {etiqueta:16} Spotify no reconoce al artista {artista!r}")
        return
    artist_id, nombre_real = resuelto
    pistas = _canciones_del_artista(raw, artist_id, nombre_real)
    dt = time.perf_counter() - started

    if not pistas:
        print(f"     {etiqueta:16} {nombre_real!r}, pero sin canciones ({dt:.2f} s)")
        return

    ranking = _emparejar(titulo_oido, pistas)
    mejor, segunda = ranking[0], (ranking[1] if len(ranking) > 1 else None)
    margen = mejor[0] - segunda[0] if segunda else float("inf")

    print(f"     {etiqueta:16} artista={nombre_real!r}  {len(pistas)} canciones  ({dt:.2f} s)")
    print(f"                      -> {_pista(mejor[1])}   ({mejor[0]:.1f})")
    if segunda:
        print(f"                      2a: {_pista(segunda[1])}   ({segunda[0]:.1f})"
              f"   margen {margen:+.1f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="¿Sirve el catalogo del artista?")
    parser.add_argument("frases", nargs="*", help="transcripciones tal como las oye Whisper")
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.ERROR)
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
    print("  1. ¿B acierta donde A falla? Entonces el catalogo del artista es")
    print("     mejor que corregir el titulo a ciegas, y merece la pena el cambio.")
    print()
    print("  2. ¿C acierta tanto como B? ENTONCES EL LLM SOBRA EN ESTA CAPA:")
    print("     Spotify ya resuelve el artista solo. Es el mejor resultado")
    print("     posible —cero latencia, cero VRAM, menos codigo— y significaria")
    print("     poder volver al modelo de 3B, o quitar el 7B del arranque.")
    print()
    print("  3. Mira el MARGEN con la segunda. Si la buena gana por poco, un")
    print("     umbral no va a separarlas de forma fiable y estariamos repitiendo")
    print("     el error del guardia de verosimilitud.")
    print()
    print("  4. ¿Sale el titulo bueno en la lista del artista? Si no aparece, no")
    print("     hay emparejamiento que valga: habria que traer mas canciones, y")
    print("     eso son mas peticiones en mitad de un comando de voz.")
    print()
    print("  5. Ojo al coste en segundos de B y C: se suma al comando, igual que")
    print("     el LLM. Si son dos peticiones a Spotify, se notan.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
