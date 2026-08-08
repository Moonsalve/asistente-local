"""Tests de la deteccion de palabra clave sobre la transcripcion.

Es la puerta de entrada del asistente en modo transcripcion: un falso negativo
lo deja sordo y un falso positivo ejecuta acciones que nadie pidio. Ambos
extremos importan, asi que se prueban los dos.
"""

from __future__ import annotations

import pytest

from asistente.audio.keyphrase import KeyphraseGate

PHRASES = ("apolo", "apollo", "a polo")


@pytest.fixture
def gate() -> KeyphraseGate:
    return KeyphraseGate(PHRASES)


@pytest.mark.parametrize(
    ("heard", "expected"),
    [
        # La clave y la orden del tiron: una sola transcripcion.
        ("Apolo, pon música", "pon musica"),
        ("apolo pasa la canción", "pasa la cancion"),
        ("Apolo abre YouTube", "abre youtube"),
        # Whisper parte o deforma los nombres propios.
        ("A polo, sube el volumen", "sube el volumen"),
        ("Apollo, qué hora es", "que hora es"),
        # Cortesia entre la clave y la orden.
        ("Apolo, por favor, pon música", "pon musica"),
        ("Apolo oye abre spotify", "abre spotify"),
        # Solo la clave: cadena vacia = "escucha la orden ahora".
        ("Apolo", ""),
        ("apolo.", ""),
    ],
)
def test_recognises_the_keyphrase(gate: KeyphraseGate, heard: str, expected: str) -> None:
    assert gate.match(heard) == expected


@pytest.mark.parametrize(
    "heard",
    [
        "",
        "pon música",  # orden sin clave: no va dirigida al asistente
        "qué hora es",
        "oye mira lo que te digo",
        "el apolo 11 llegó a la luna",  # la clave NO va al principio
        "hola qué tal",
    ],
)
def test_ignores_what_is_not_addressed_to_it(gate: KeyphraseGate, heard: str) -> None:
    """En modo transcripcion se oye toda conversacion cercana: lo que no
    empiece por la clave tiene que ignorarse en silencio."""
    assert gate.match(heard) is None


def test_empty_phrase_list_is_rejected() -> None:
    with pytest.raises(ValueError):
        KeyphraseGate(())


def test_custom_multiword_phrase() -> None:
    """Debe admitir claves de varias palabras, no solo nombres sueltos."""
    gate = KeyphraseGate(("oye apolo",))
    assert gate.match("oye apolo pon música") == "pon musica"
    assert gate.match("pon música") is None
