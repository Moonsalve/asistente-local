"""Numeros en espanol escritos con letra.

Whisper transcribe "volumen al cincuenta" tal cual, no como "50". Sin esto,
`volume.set` escalaria al LLM casi siempre: un fallback de 300 ms para algo que
se resuelve con un diccionario.

Cubre 0-100, que es todo lo que necesita un porcentaje de volumen. Fuera de ese
rango devuelve None y el comando escala al LLM, que es el comportamiento
correcto: mejor delegar que adivinar.
"""

from __future__ import annotations

from asistente.router.text import normalize

_UNITS: dict[str, int] = {
    "cero": 0, "un": 1, "uno": 1, "una": 1, "dos": 2, "tres": 3, "cuatro": 4,
    "cinco": 5, "seis": 6, "siete": 7, "ocho": 8, "nueve": 9, "diez": 10,
    "once": 11, "doce": 12, "trece": 13, "catorce": 14, "quince": 15,
    "dieciseis": 16, "diecisiete": 17, "dieciocho": 18, "diecinueve": 19,
    "veinte": 20, "veintiuno": 21, "veintiun": 21, "veintidos": 22,
    "veintitres": 23, "veinticuatro": 24, "veinticinco": 25, "veintiseis": 26,
    "veintisiete": 27, "veintiocho": 28, "veintinueve": 29,
}
_TENS: dict[str, int] = {
    "treinta": 30, "cuarenta": 40, "cincuenta": 50, "sesenta": 60,
    "setenta": 70, "ochenta": 80, "noventa": 90,
}
_HUNDRED = {"cien", "ciento"}


def parse_spanish_number(text: str) -> int | None:
    """Convierte digitos o palabras a entero. None si no reconoce nada."""
    cleaned = normalize(text)
    if not cleaned:
        return None

    # Camino rapido: Whisper suele devolver digitos cuando puede.
    digits = "".join(ch for ch in cleaned if ch.isdigit())
    if digits and digits == cleaned.replace(" ", ""):
        value = int(digits)
        return value if 0 <= value <= 100 else None

    words = [w for w in cleaned.split() if w != "y"]
    if not words:
        return None

    if words[0] in _HUNDRED:
        # "cien" solo, o "ciento" seguido de resto (que ya se sale del rango util).
        return 100 if len(words) == 1 else None

    # Un numero valido en este rango es, como mucho, una decena seguida de una
    # unidad: "cuarenta y dos". Cualquier otro orden ("cinco treinta") no es un
    # numero, y aceptarlo produciria un volumen inventado.
    total = 0
    seen_tens = False
    seen_unit = False
    for word in words:
        if (tens := _TENS.get(word)) is not None:
            if seen_tens or seen_unit:
                return None
            seen_tens = True
            total += tens
        elif (unit := _UNITS.get(word)) is not None:
            # Tras una decena solo puede ir una unidad simple: "treinta veinte"
            # no es un numero aunque ambas partes lo sean.
            if seen_unit or (seen_tens and unit >= 10):
                return None
            seen_unit = True
            total += unit
        else:
            return None

    return total if 0 <= total <= 100 else None
