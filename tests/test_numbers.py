"""Numeros hablados en espanol.

Existe para que `volume.set` no escale al LLM: Whisper transcribe "al cincuenta"
con letra tan a menudo como con digitos, y pagar 300 ms de LLM por convertir una
palabra a un entero es exactamente lo que la arquitectura intenta evitar.
"""

from __future__ import annotations

import pytest

from asistente.numbers import parse_spanish_number


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("40", 40),
        ("0", 0),
        ("100", 100),
        ("cero", 0),
        ("cinco", 5),
        ("quince", 15),
        ("veinte", 20),
        ("veinticinco", 25),
        ("treinta", 30),
        ("treinta y cinco", 35),
        ("cuarenta y dos", 42),
        ("cincuenta", 50),
        ("setenta y ocho", 78),
        ("noventa y nueve", 99),
        ("cien", 100),
        # Whisper devuelve acentos y mayusculas; se normaliza antes de parsear.
        ("Cincuenta", 50),
        ("dieciséis", 16),
    ],
)
def test_parses_known_numbers(text: str, expected: int) -> None:
    assert parse_spanish_number(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "",
        "muchisimo",
        "un poco mas",
        "cinco treinta",  # orden invalido, no es un numero
        "doscientos",  # fuera del rango util de un porcentaje
        "150",
    ],
)
def test_rejects_what_it_cannot_parse(text: str) -> None:
    """Devolver None hace que el comando escale al LLM, que es lo correcto:
    mejor delegar que fijar un volumen inventado."""
    assert parse_spanish_number(text) is None
