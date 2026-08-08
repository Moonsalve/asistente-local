"""Etapa 1 del router: coincidencia literal.

Un lookup de diccionario sobre el texto ya normalizado. Cuesta microsegundos y
cubre las formas que repites a diario, que en la practica son la mayoria de las
interacciones. Existe para que el comando mas frecuente sea tambien el mas
barato.
"""

from __future__ import annotations

from asistente.router.catalog import Catalog


def match_literal(catalog: Catalog, normalized: str) -> str | None:
    """Devuelve el nombre del intent, o None si no hay coincidencia exacta."""
    return catalog.literal_index.get(normalized)
