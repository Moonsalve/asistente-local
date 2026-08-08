"""Carga de `commands.yaml` y precomputo de indices.

Todo el coste de preparar el catalogo (parsear YAML, compilar regex, codificar
las frases de ejemplo a vectores) se paga una sola vez al arrancar. En runtime
el router solo hace un lookup de diccionario y un producto matricial.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import yaml

from asistente.router.schema import IntentSpec
from asistente.router.text import normalize

if TYPE_CHECKING:
    from asistente.router.embedder import Embedder


class CatalogError(ValueError):
    """El catalogo esta mal formado. Siempre es un fallo de arranque, nunca de runtime."""


@dataclass(frozen=True, slots=True)
class Catalog:
    """Catalogo compilado y listo para consultar."""

    intents: dict[str, IntentSpec]
    #: frase normalizada -> nombre de intent (etapa literal, O(1))
    literal_index: dict[str, str]
    #: regex compilados por intent, en el orden declarado
    slot_regexes: dict[str, tuple[re.Pattern[str], ...]]
    #: matriz (n_ejemplos, dim) ya centrada y renormalizada
    embeddings: np.ndarray
    #: nombre de intent para cada fila de `embeddings`
    example_owners: tuple[str, ...]
    #: centroide del catalogo, necesario para proyectar las consultas igual
    mean: np.ndarray

    def project(self, vectors: np.ndarray) -> np.ndarray:
        """Centra y renormaliza para que el producto punto siga siendo un coseno.

        Restar el centroide elimina la direccion comun que domina los embeddings
        de e5 y que comprime todas las similitudes en la banda 0.85-0.96. Medido
        sobre este catalogo, el centrado abre el rango util a 0.19-0.73, que es
        lo que hace utilizable la comparacion entre intents.
        """
        centered = vectors - self.mean
        norms = np.clip(np.linalg.norm(centered, axis=1, keepdims=True), a_min=1e-9, a_max=None)
        return (centered / norms).astype(np.float32)

    def is_fallback(self, intent: str) -> bool:
        return self.intents[intent].fallback


def load_catalog(path: Path, embedder: Embedder) -> Catalog:
    """Lee el YAML, valida y precomputa los indices.

    Falla ruidosamente ante cualquier inconsistencia: un catalogo roto detectado
    al arrancar es un error de 2 minutos, uno detectado en runtime es un comando
    que se traga el usuario a mitad de una frase.
    """
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "intents" not in raw:
        raise CatalogError(f"{path}: se esperaba una clave raiz 'intents'")

    intents: dict[str, IntentSpec] = {}
    for name, spec in raw["intents"].items():
        try:
            intents[str(name)] = IntentSpec.model_validate(spec)
        except Exception as exc:  # noqa: BLE001 - se re-lanza con contexto util
            raise CatalogError(f"intent '{name}' invalido: {exc}") from exc

    if not intents:
        raise CatalogError(f"{path}: el catalogo esta vacio")

    if not any(spec.fallback for spec in intents.values()):
        raise CatalogError(
            f"{path}: falta un intent con fallback=true. Sin clase negativa el router "
            "clasifica cualquier frase como un comando (ver semantic.py)."
        )

    literal_index: dict[str, str] = {}
    for name, spec in intents.items():
        if spec.fallback and spec.literal:
            raise CatalogError(f"intent '{name}': un fallback no puede tener frases literales")
        for phrase in spec.literal:
            key = normalize(phrase)
            if (owner := literal_index.get(key)) is not None:
                raise CatalogError(
                    f"frase literal duplicada '{phrase}': la reclaman '{owner}' y '{name}'"
                )
            literal_index[key] = name

    slot_regexes: dict[str, tuple[re.Pattern[str], ...]] = {}
    for name, spec in intents.items():
        compiled = []
        for pattern in spec.slot_patterns:
            try:
                compiled.append(re.compile(pattern, flags=re.IGNORECASE))
            except re.error as exc:
                raise CatalogError(f"intent '{name}': regex invalido {pattern!r}: {exc}") from exc
        slot_regexes[name] = tuple(compiled)
        if spec.slots_required and not compiled:
            raise CatalogError(f"intent '{name}': slots_required=true pero no hay slot_patterns")

    # Un solo batch de encoding para todo el catalogo: mucho mas rapido que
    # frase a frase, y solo ocurre al arrancar.
    examples: list[str] = []
    owners: list[str] = []
    for name, spec in intents.items():
        for phrase in spec.examples:
            examples.append(normalize(phrase))
            owners.append(name)

    raw = embedder.encode(examples)
    mean = raw.mean(axis=0, keepdims=True).astype(np.float32)
    centered = raw - mean
    norms = np.clip(np.linalg.norm(centered, axis=1, keepdims=True), a_min=1e-9, a_max=None)

    return Catalog(
        intents=intents,
        literal_index=literal_index,
        slot_regexes=slot_regexes,
        embeddings=(centered / norms).astype(np.float32),
        example_owners=tuple(owners),
        mean=mean,
    )
