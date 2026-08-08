"""Normalizacion de texto para el matching.

Whisper devuelve acentos, mayusculas y puntuacion; el catalogo se escribe sin
acentos por comodidad. Normalizar ambos lados hace que la etapa literal sea un
lookup de diccionario en O(1) en vez de una comparacion difusa.

Ojo: esto es solo para *comparar*. La extraccion de slots trabaja sobre el texto
normalizado tambien (los regex del catalogo estan escritos sin acentos), pero el
valor del slot se devuelve tal cual aparece ahi, asi que "cancion" llega a la
skill sin tilde. Las skills que lo necesiten (busqueda web) no se ven afectadas
porque los buscadores toleran la falta de acentos.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

_PUNCT = re.compile(r"[^\w\s]", flags=re.UNICODE)
_SPACES = re.compile(r"\s+")


@lru_cache(maxsize=2048)
def normalize(text: str) -> str:
    """Minusculas, sin acentos, sin puntuacion, espacios colapsados.

    Se cachea porque el catalogo normaliza las mismas frases en cada arranque y
    la etapa literal normaliza cada transcripcion una sola vez.
    """
    lowered = text.strip().lower()
    # NFD separa la letra de su diacritico; Mn = "Mark, nonspacing" (los acentos).
    # Se preserva la enye: en espanol distingue palabras, quitarla rompe sentido.
    decomposed = unicodedata.normalize("NFD", lowered)
    stripped = "".join(
        ch
        for ch, cat in ((c, unicodedata.category(c)) for c in decomposed)
        if cat != "Mn" or ch == "̃"
    )
    recomposed = unicodedata.normalize("NFC", stripped)
    without_punct = _PUNCT.sub(" ", recomposed)
    return _SPACES.sub(" ", without_punct).strip()
