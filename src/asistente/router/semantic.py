"""Etapa 2 del router: coincidencia semantica y extraccion de slots.

Es el caballo de batalla del sistema. Compara el *significado* de tu frase con
el de las frases de ejemplo del catalogo, asi que reconoce formas que nunca se
escribieron: "esta cancion no me gusta" cae en `media.next` sin compartir una
sola palabra con "siguiente cancion".

Coste: un forward pass del encoder (~10-20 ms) mas un producto matricial contra
una matriz de unos pocos cientos de filas (microsegundos).

POR QUE NO BASTA UN UMBRAL ABSOLUTO
-----------------------------------
El diseno original usaba "coseno >= 0.80 => es un comando". Medido contra este
catalogo, no funciona: e5 comprime todas las similitudes en la banda 0.85-0.96,
y ahi los positivos bajan a 0.856 mientras los negativos suben a 0.903. Se
solapan, asi que NINGUN umbral los separa.

La solucion son dos cambios que se refuerzan:

1. **Centrado** (en `Catalog.project`): restar el centroide del catalogo elimina
   la direccion comun que infla todos los cosenos. El rango util pasa a
   0.19-0.73.
2. **Clase negativa** (`fallback: true` en el catalogo): en vez de preguntar
   "supera el umbral?", se pregunta "se parece mas a un comando o a charla?".
   La decision pasa a ser relativa, que es mucho mas estable que una absoluta, y
   se ajusta de forma declarativa: si una pregunta abierta se ejecuta como
   comando, se anade a los ejemplos del fallback.

El umbral sigue existiendo, pero ya solo como red de seguridad para frases que
no se parecen a nada, no como el mecanismo principal de decision.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from asistente.router.catalog import Catalog
from asistente.router.embedder import Embedder


@dataclass(frozen=True, slots=True)
class SemanticMatch:
    intent: str
    score: float
    #: Segundo mejor intent y su score. Sirve para detectar intents que se
    #: solapan: si la diferencia es minima, al catalogo le faltan ejemplos que
    #: los separen.
    runner_up: str | None
    runner_up_score: float


class SemanticMatcher:
    def __init__(self, catalog: Catalog, embedder: Embedder, threshold: float = 0.25) -> None:
        self._catalog = catalog
        self._embedder = embedder
        self._threshold = threshold

    @property
    def threshold(self) -> float:
        return self._threshold

    def match(self, normalized: str) -> SemanticMatch | None:
        """Intent ganador por coseno centrado.

        Devuelve None si gana la clase negativa o si nadie llega al umbral
        minimo; en ambos casos el motor escala al LLM.
        """
        raw = self._embedder.encode([normalized])
        query = self._catalog.project(raw)[0]
        # Catalogo y consulta estan centrados y normalizados: el producto punto
        # ES el coseno, sin dividir por normas.
        scores = self._catalog.embeddings @ query

        best_idx = int(np.argmax(scores))
        # Clamp: en float32 el producto punto de un vector consigo mismo sale
        # 1.0000001, y RouteResult.score exige el rango [-1, 1].
        best_score = float(np.clip(scores[best_idx], -1.0, 1.0))
        best_intent = self._catalog.example_owners[best_idx]

        if best_score < self._threshold or self._catalog.is_fallback(best_intent):
            return None

        runner_up, runner_up_score = self._best_other_intent(scores, best_intent)
        return SemanticMatch(
            intent=best_intent,
            score=best_score,
            runner_up=runner_up,
            runner_up_score=runner_up_score,
        )

    def _best_other_intent(self, scores: np.ndarray, exclude: str) -> tuple[str | None, float]:
        best_intent: str | None = None
        best_score = -1.0
        for idx, owner in enumerate(self._catalog.example_owners):
            if owner == exclude:
                continue
            if (score := float(scores[idx])) > best_score:
                best_score, best_intent = score, owner
        return best_intent, best_score


def extract_slots(catalog: Catalog, intent: str, normalized: str) -> dict[str, str] | None:
    """Aplica los regex del intent ya decidido y devuelve los grupos nombrados.

    Devuelve `{}` si el intent no declara patrones (no necesita argumentos), y
    `None` si declara patrones pero ninguno casa, lo que el motor interpreta
    como "escala al LLM" si `slots_required` esta activo.

    Clave del diseno: estos regex NO desambiguan intents, solo extraen. Por eso
    "pon" puede aparecer en `spotify.play` y en `app.open` sin conflicto: cuando
    llegan aqui, la decision ya esta tomada.

    Se recorren TODOS los patrones acumulando, y para cada clave manda el primero
    que la rellena. Asi un intent puede repartir sus argumentos en patrones
    independientes -"volumen al 40 de spotify" necesita un regex para el nivel y
    otro para el destino- sin que eso cambie nada donde varios patrones son
    alternativas del mismo argumento: ahi el primero sigue ganando, igual que
    cuando esto devolvia al primer acierto.
    """
    patterns = catalog.slot_regexes[intent]
    if not patterns:
        return {}

    slots: dict[str, str] = {}
    for pattern in patterns:
        if (m := pattern.search(normalized)) is None:
            continue
        for key, value in m.groupdict().items():
            if key not in slots and value and value.strip():
                slots[key] = value.strip()
    return slots or None
