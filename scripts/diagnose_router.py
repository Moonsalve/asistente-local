"""Diagnostico del router: scores reales por frase.

Sirve para calibrar el umbral del coseno. Imprime, para cada frase, el intent
ganador, su score y el segundo clasificado. La separacion entre ambos es lo que
indica si al catalogo le faltan ejemplos que distingan dos intents parecidos.

Uso:  PYTHONPATH=src python scripts/diagnose_router.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asistente.router.catalog import load_catalog  # noqa: E402
from asistente.router.embedder import OnnxEmbedder  # noqa: E402
from asistente.router.semantic import SemanticMatcher  # noqa: E402
from asistente.router.text import normalize  # noqa: E402

# Frases que SI deben resolver a un intent.
POSITIVES = [
    ("pásala", "media.next"),
    ("esta no me gusta", "media.next"),
    ("no quiero escuchar esta", "media.next"),
    ("brinca esta canción", "media.next"),
    ("regrésala", "media.previous"),
    ("échale para atrás", "media.previous"),
    ("párale a la música", "media.play_pause"),
    ("detén eso", "media.play_pause"),
    ("súbele", "volume.up"),
    ("no se escucha nada", "volume.up"),
    ("bájale tantito", "volume.down"),
    ("quita el audio", "volume.mute"),
    ("sin sonido", "volume.mute"),
    ("volumen al 40", "volume.set"),
    # Volumen con destino: mismo intent, slot distinto. La pareja
    # "silencia spotify" / "termina el spotify" es la que mas cerca ha estado de
    # cruzarse (0.532 contra 0.357 antes de anclar app.close).
    ("súbele a spotify", "volume.up"),
    ("bájale a la música", "volume.down"),
    ("cállale el sonido a spotify", "volume.mute"),
    ("pon spotify al 30", "volume.set"),
    ("a qué volumen está la música", "volume.query"),
    ("pon música de los 80", "spotify.play"),
    ("quiero escuchar a shakira", "spotify.play"),
    ("ábreme el chrome", "open.target"),
    ("corre la calculadora", "open.target"),
    ("mata el discord", "app.close"),
    ("termina el spotify", "app.close"),
    ("entra a youtube", "open.target"),
    ("llévame a netflix", "open.target"),
    ("búscame el precio del dólar", "web.search"),
    ("googlea restaurantes cerca", "web.search"),
    ("qué hora tienes", "system.time"),
]

# Frases que NO deben resolver a ningun intent: tienen que caer al LLM.
# Son las que fijan el suelo del umbral.
NEGATIVES = [
    "explícame la teoría de la relatividad en dos frases",
    "cuéntame un chiste",
    "qué opinas de la inteligencia artificial",
    "cuánto es doscientos por trescientos",
    "recuérdame comprar leche mañana",
    "cómo se dice buenos días en japonés",
    "escribe un correo para mi jefe",
    "qué tiempo hace en tokio y cuánto cuesta un vuelo",
]


def main() -> int:
    embedder = OnnxEmbedder()
    catalog = load_catalog(ROOT / "commands.yaml", embedder)
    matcher = SemanticMatcher(catalog, embedder, threshold=0.25)

    print(f"{'frase':<48} {'ganador':<18} {'score':>6}  {'2o':<18} {'score':>6}")
    print("-" * 104)

    positive_scores: list[float] = []
    wrong_positives = 0
    print("\n### POSITIVOS\n")
    for phrase, expected in POSITIVES:
        m = matcher.match(normalize(phrase))
        if m is None:
            wrong_positives += 1
            print(f"ERR {phrase:<44} {'_fallback (escalo al LLM)':<18}")
            continue
        ok = "OK " if m.intent == expected else "ERR"
        wrong_positives += 0 if m.intent == expected else 1
        positive_scores.append(m.score)
        print(
            f"{ok} {phrase:<44} {m.intent:<18} {m.score:>6.3f}  "
            f"{m.runner_up or '-':<18} {m.runner_up_score:>6.3f}"
        )

    wrong_negatives = 0
    print("\n### NEGATIVOS (deben ganar la clase _fallback)\n")
    for phrase in NEGATIVES:
        m = matcher.match(normalize(phrase))
        if m is None:
            print(f"OK  {phrase:<44} {'_fallback':<18}")
        else:
            wrong_negatives += 1
            print(f"ERR {phrase:<44} {m.intent:<18} {m.score:>6.3f}")

    print("\n" + "=" * 104)
    print(f"positivo mas bajo        : {min(positive_scores):.3f}")
    print(f"negativos mal clasificados: {wrong_negatives}/{len(NEGATIVES)}")
    print(f"positivos mal clasificados: {wrong_positives}/{len(POSITIVES)}")
    return 0 if wrong_negatives == 0 and wrong_positives == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
