"""Diagnostico de la correccion de titulos con LLM.

`probe_music_ai.py` decidio SI merecia la pena hacer esto. Este script comprueba
que lo que se hizo funciona, y sobre todo enseña **la frontera**: que
correcciones se aceptan, cuales se descartan por inverosimiles, y con cuanto
margen. Esa frontera es el unico parametro ajustable de todo el modulo.

Se enseña por cada frase:

    lo que dice el LLM      titulo y artista corregidos, y cuanto tardo
    la puntuacion           cuanto se parece a lo que se oyo (0-100)
    el veredicto            ACEPTA / DESCARTA contra PLAUSIBLE_THRESHOLD
    lo que sonaria          el resultado de buscar eso en Spotify (con --spotify)

Uso (en el PC, con Ollama arrancado y el venv activado):

    python scripts/diagnose_music_ai.py
    python scripts/diagnose_music_ai.py "lovin machin de tibi guerl"
    python scripts/diagnose_music_ai.py --spotify        # busca de verdad

Pon frases TUYAS, dichas como Whisper las transcribe. Para sacarlas: arranca el
asistente con `-v`, pide canciones en voz alta y copia lo que aparezca
transcrito en el log.

No reproduce nada: solo busca.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asistente.config import Config, Secrets  # noqa: E402
from asistente.router.text import normalize  # noqa: E402
from asistente.skills.music_ai import (  # noqa: E402
    PLAUSIBLE_THRESHOLD,
    MusicQuery,
    build_resolver,
)
from asistente.skills.spotify import SpotifyClient, similitud  # noqa: E402

OK = "  OK "
MAL = " MAL "

#: Destrozos reales de `say` + Whisper, y alguno inventado del mismo estilo.
#: Los tres ultimos NO son canciones que exista forma de reconocer: estan para
#: ver que hace el modelo cuando no sabe, que es la pregunta importante.
EJEMPLOS: tuple[tuple[str, str | None], ...] = (
    ("lovin machin", "tibi guerl"),
    ("no tinelsmatters", "metalica"),
    ("esmels laik tin espirit", "nirvana"),
    ("blain ding lights", "uiquen"),
    ("ponlobes rock", "tv girl"),
    ("creep", None),
    ("musica tranquila", None),
    ("una cancion bonita", None),
)


def _partir(frase: str) -> tuple[str, str | None]:
    """Parte "X de Y" por el ULTIMO "de", igual que hace el router."""
    marca = " de "
    if marca in frase:
        titulo, _, artista = frase.rpartition(marca)
        return titulo.strip(), artista.strip()
    return frase.strip(), None


def _puntuacion(query: str, artist: str | None, corregido: MusicQuery) -> float:
    """La misma comparacion que hace `MusicResolver._plausible`."""
    if artist:
        oido, propuesto = f"{query} {artist}", corregido.pedido
    else:
        oido, propuesto = query, corregido.titulo
    return similitud(normalize(oido), normalize(propuesto))


def _suena(client: SpotifyClient, corregido: MusicQuery) -> str:
    """Que reproduciria `play_query` con el nombre corregido, sin reproducirlo."""
    raw = client._client  # diagnostico: se mira por dentro a proposito
    if raw is None:
        return "(sin Spotify)"
    consulta = (
        f'track:"{corregido.titulo}" artist:"{corregido.artista}"'
        if corregido.artista
        else corregido.titulo
    )
    try:
        res = raw.search(q=consulta, type="track", limit=1)
    except Exception as exc:  # el diagnostico no debe morirse por una peticion
        return f"(error: {exc})"
    items = (res or {}).get("tracks", {}).get("items") or []
    if not items:
        return "(nada)"
    artistas = ", ".join(a.get("name", "") for a in items[0].get("artists") or [])
    return f"{items[0].get('name', '?')} — {artistas}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnostico del corrector de titulos")
    parser.add_argument(
        "frases", nargs="*", help='transcripciones, p.ej. "lovin machin de tibi guerl"'
    )
    parser.add_argument("--spotify", action="store_true", help="buscar ademas en Spotify")
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    args = parser.parse_args()

    # Silencio: el resolvedor ya registra su veredicto y aqui se imprime mejor.
    logging.basicConfig(level=logging.ERROR)

    config = Config.load(args.config)
    peticiones = [_partir(f) for f in args.frases] if args.frases else list(EJEMPLOS)

    print("=" * 78)
    print("CORRECCION DE TITULOS CON LLM")
    print("=" * 78)
    print(f"\nmodelo: {config.llm.model}    umbral de verosimilitud: {PLAUSIBLE_THRESHOLD:.0f}")

    if not config.spotify.resolve_with_llm:
        print(f"\n{MAL} spotify.resolve_with_llm esta en false: el asistente NO lo usara.")
        print("      (este diagnostico sigue, para poder probarlo antes de activarlo)")

    resolver = build_resolver(config.llm)
    if resolver is None:
        print(f"\n{MAL} no se pudo construir el resolvedor. ¿Esta Ollama arrancado?")
        print("      ollama serve   y   ollama pull " + config.llm.model)
        return 1

    spotify = SpotifyClient(config, Secrets())
    if args.spotify and not (spotify.available and spotify.connect()):
        print(f"\n{MAL} sin conexion con Spotify. Ejecuta antes diagnose_spotify.py")
        return 1

    aceptadas = descartadas = fallos = 0
    peor_aceptada, mejor_descartada = 100.0, 0.0
    total_s = 0.0

    for query, artist in peticiones:
        oido = f"{query} de {artist}" if artist else query
        print("-" * 78)
        print(f"OIDO:  {oido!r}")

        # Se llama al metodo interno para poder ver la respuesta CRUDA del
        # modelo aunque el veredicto sea descartarla: `resolve()` devuelve None
        # en ese caso, y entonces no habria nada que enseñar.
        started = time.perf_counter()
        payload = resolver._ask(oido)  # diagnostico: interesa la respuesta cruda
        dt = time.perf_counter() - started
        total_s += dt

        if payload is None:
            print(f"{MAL} el LLM no respondio ({dt:.2f} s)")
            fallos += 1
            continue

        try:
            corregido = MusicQuery.model_validate(payload)
        except Exception:
            print(f"{MAL} respuesta invalida ({dt:.2f} s): {payload!r}")
            fallos += 1
            continue

        score = _puntuacion(query, artist, corregido)
        vale = score >= PLAUSIBLE_THRESHOLD
        marca = OK if vale else MAL
        veredicto = "ACEPTA" if vale else "DESCARTA"

        print(f"       el LLM dice   titulo={corregido.titulo!r} "
              f"artista={corregido.artista!r}  ({dt:.2f} s)")
        print(f"{marca} {veredicto:9} puntuacion {score:.1f} "
              f"contra un umbral de {PLAUSIBLE_THRESHOLD:.0f}")
        if artist is None:
            print("       (sin artista dicho: se comparan solo los titulos)")

        if vale:
            aceptadas += 1
            peor_aceptada = min(peor_aceptada, score)
        else:
            descartadas += 1
            mejor_descartada = max(mejor_descartada, score)

        if args.spotify:
            print(f"       sonaria       {_suena(spotify, corregido)}")
        print()

    print("=" * 78)
    n = len(peticiones)
    print(f"aceptadas {aceptadas}/{n}   descartadas {descartadas}/{n}   sin respuesta {fallos}/{n}")
    if aceptadas:
        print(f"peor aceptada:    {peor_aceptada:.1f}")
    if descartadas:
        print(f"mejor descartada: {mejor_descartada:.1f}")
    if aceptadas and descartadas:
        hueco = peor_aceptada - mejor_descartada
        print(f"hueco: {hueco:+.1f} puntos", end="")
        print("  (si es negativo, ningun umbral separa estos casos)" if hueco <= 0 else "")
    if n:
        print(f"coste medio del modelo: {total_s / n:.2f} s por peticion")

    print()
    print("QUE MIRAR:")
    print("  - Las que DESCARTA, ¿eran invenciones? Entonces el umbral hace su trabajo.")
    print("  - ¿Descarta correcciones BUENAS? Baja PLAUSIBLE_THRESHOLD en music_ai.py,")
    print("    sin pasar de la mejor descartada de arriba.")
    print("  - ¿ACEPTA canciones que no son? El modelo se las inventa y se parecen")
    print("    demasiado a lo que oiste: subir el umbral no basta, hace falta otro modelo.")
    print("  - Con 'musica tranquila' o 'una cancion bonita' el modelo NO deberia sacarse")
    print("    un titulo concreto: no son canciones, son generos. Si lo hace, se descartan")
    print("    por puntuacion, que es exactamente para lo que esta el umbral.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
