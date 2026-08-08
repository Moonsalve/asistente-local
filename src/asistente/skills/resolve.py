"""Resolucion de nombres hablados a entradas de la allowlist.

Whisper transcribe marcas como suenan ("spotifai", "discor", "vs code"), asi que
la comparacion no puede ser exacta. El orden es deliberado:

1. Coincidencia exacta del nombre canonico o de un alias.
2. Coincidencia difusa (RapidFuzz) por encima de un umbral alto.

La difusa va DESPUES y con umbral alto a proposito: es la que puede confundir
"discord" con "record", y preferimos escalar al LLM antes que abrir la
aplicacion equivocada.
"""

from __future__ import annotations

from rapidfuzz import fuzz, process

from asistente.config import AppSpec
from asistente.router.text import normalize

#: Umbral de similitud (0-100). Alto a proposito: mas vale no reconocer que
#: reconocer mal.
FUZZY_THRESHOLD = 82.0


def build_alias_index(apps: dict[str, AppSpec]) -> dict[str, str]:
    """alias normalizado -> nombre canonico de la app."""
    index: dict[str, str] = {}
    for name, spec in apps.items():
        index[normalize(name)] = name
        for alias in spec.aliases:
            index[normalize(alias)] = name
    return index


def resolve(spoken: str, index: dict[str, str]) -> str | None:
    """Devuelve el nombre canonico, o None si no hay una coincidencia fiable."""
    key = normalize(spoken)
    if not key:
        return None
    if (exact := index.get(key)) is not None:
        return exact

    match = process.extractOne(key, index.keys(), scorer=fuzz.WRatio, score_cutoff=FUZZY_THRESHOLD)
    return index[match[0]] if match is not None else None
