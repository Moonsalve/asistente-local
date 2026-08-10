"""Compara TRES formas de encontrar una cancion mal transcrita.

Para decidir si merece la pena rehacer la busqueda de musica alrededor del LLM
hay una pregunta que no se puede responder razonando, solo midiendo:

    ¿sabe el modelo local que existe "TV Girl", o se lo va a inventar?

Este script lo responde con TUS canciones y TU modelo. Para cada frase prueba:

    A) Spotify a pelo      la transcripcion tal cual, a la Web API
    B) LLM y luego Spotify el modelo corrige la ortografia y se busca eso
    C) Tu biblioteca       lo que hace hoy el asistente

y enseña que devuelve cada una. Si B acierta donde A falla, el cambio vale la
pena. Si B se inventa titulos, no.

Uso (en el PC, con Ollama y Spotify en marcha):

    python scripts/probe_music_ai.py
    python scripts/probe_music_ai.py "reproduce lovin machin de tibi guerl"

Sin argumentos usa una lista de ejemplos con destrozos tipicos. Cambialos por
canciones TUYAS: es lo unico que hace la medida representativa.

No reproduce nada: solo busca.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

#: Destrozos reales medidos con `say` + Whisper, y alguno inventado del mismo
#: estilo. Sustituyelos por los tuyos.
EJEMPLOS = [
    "loving machine de tv gery",
    "lovin machin de tibi guerl",
    "ponlo best rock de tv girl",
    "no tinelsmatters de metalica",
    "blain ding lights de de uiquen",
    "esmels laik tin espirit de nirvana",
]

_PROMPT = """Eres un corrector de nombres de canciones. Recibes una peticion de \
musica transcrita por un sistema de voz en espanol, que escribe los nombres en \
ingles como suenan y comete errores.

Devuelve SOLO un objeto JSON con dos claves:
  "titulo":  el nombre real de la cancion, escrito correctamente
  "artista": el nombre real del grupo o cantante, o "" si no se dice

Si no reconoces la cancion, escribe tu mejor reconstruccion literal en vez de \
inventarte otra distinta. No expliques nada.

Peticion: {frase}"""


def _normaliza_con_llm(client: object, model: str, frase: str) -> tuple[str, str, float]:
    """(titulo, artista, segundos) segun el LLM."""
    started = time.perf_counter()
    respuesta = client.chat(  # type: ignore[attr-defined]
        model=model,
        messages=[{"role": "user", "content": _PROMPT.format(frase=frase)}],
        format="json",
        options={"temperature": 0.0},
    )
    dt = time.perf_counter() - started
    try:
        datos = json.loads(respuesta["message"]["content"])
        return str(datos.get("titulo", "")), str(datos.get("artista", "")), dt
    except (KeyError, ValueError, TypeError):
        return "", "", dt


def _primera(client: object, consulta: str) -> str:
    """El primer resultado de Spotify para esa consulta, como texto."""
    try:
        res = client.search(q=consulta, type="track", limit=1)  # type: ignore[attr-defined]
    except Exception as exc:  # noqa: BLE001
        return f"(error: {exc})"
    items = (res or {}).get("tracks", {}).get("items") or []
    if not items:
        return "(nada)"
    item = items[0]
    artistas = ", ".join(a.get("name", "") for a in item.get("artists") or [])
    return f"{item.get('name', '?')} — {artistas}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Compara formas de resolver un titulo mal oido")
    parser.add_argument("frases", nargs="*", help="transcripciones a probar")
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    args = parser.parse_args()

    from asistente.config import Config, Secrets
    from asistente.skills.spotify import SpotifyClient

    config = Config.load(args.config)
    frases = args.frases or EJEMPLOS

    print("=" * 78)
    print("¿QUIEN ENCUENTRA MEJOR UNA CANCION MAL TRANSCRITA?")
    print("=" * 78)

    spotify = SpotifyClient(config, Secrets())
    if not spotify.available or not spotify.connect():
        print(" MAL  sin conexion con Spotify. Ejecuta antes diagnose_spotify.py")
        return 1
    raw = spotify._client  # noqa: SLF001 - sonda: se mira por dentro a proposito
    spotify.warmup()

    try:
        from ollama import Client

        llm = Client(host=config.llm.host)
        llm.chat(model=config.llm.model, messages=[{"role": "user", "content": "hola"}],
                 options={"num_predict": 1})
    except Exception as exc:  # noqa: BLE001
        print(f" MAL  sin Ollama ({exc}). Arrancalo y vuelve a probar.")
        return 1

    print(f"\nmodelo: {config.llm.model}   frases: {len(frases)}\n")

    for frase in frases:
        print("-" * 78)
        print(f"OIDO:  {frase!r}\n")

        print(f"  A) Spotify a pelo      {_primera(raw, frase)}")

        titulo, artista, dt = _normaliza_con_llm(llm, config.llm.model, frase)
        consulta = f'track:"{titulo}" artist:"{artista}"' if artista else titulo
        hallado = _primera(raw, consulta) if titulo else "(el LLM no devolvio nada)"
        print(f"  B) LLM ({dt:.2f}s) dice  titulo={titulo!r} artista={artista!r}")
        print(f"     y Spotify da        {hallado}")

        cancion = spotify.find_liked(frase)
        print(f"  C) Tu biblioteca       {cancion.label if cancion else '(nada)'}")
        print()

    print("=" * 78)
    print("QUE MIRAR:")
    print("  - Si B acierta donde A falla, el LLM aporta y merece la pena el cambio.")
    print("  - Si B devuelve titulos que NO son los que dijiste, se los esta")
    print("    inventando, y eso es peor que no encontrar nada.")
    print("  - Si A ya acierta casi siempre, no hace falta LLM: sobra con limpiar")
    print("    la consulta antes de mandarla.")
    print("  - El coste de B se suma a cada comando de musica (arriba, por frase).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
