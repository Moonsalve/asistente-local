"""Tests del struct de SendInput.

Este archivo existe por un bug concreto: la union `INPUT` estaba declarada solo
con `KEYBDINPUT`, que es lo unico que el asistente usa. El resultado eran 32
bytes en vez de 40, y `SendInput` rechaza la llamada entera cuando `cbSize` no
coincide con `sizeof(INPUT)`. Sintoma: `volume.mute` y todas las teclas
multimedia fallaban en silencio, sin excepcion y sin log, porque el unico indicio
era un valor de retorno de 0 que nadie miraba.

No se puede testear el efecto (hace falta Windows), pero si el tamano, que es
donde estaba el error. `ctypes.sizeof` no necesita Windows.
"""

from __future__ import annotations

import ctypes

from asistente.skills import winkeys


def test_input_struct_matches_what_sendinput_expects() -> None:
    """40 bytes en x64. Los 8 de mas respecto a KEYBDINPUT los aporta MOUSEINPUT,
    que es el miembro grande de la union aunque este modulo no lo use nunca."""
    assert winkeys.input_struct_size() == winkeys.EXPECTED_INPUT_SIZE


def test_union_is_sized_by_mouseinput_not_keybdinput() -> None:
    """Fija el motivo, no solo el numero: si alguien quita `mi` de la union
    porque 'no se usa', este test explica por que no se puede."""
    assert ctypes.sizeof(winkeys._MouseInput) > ctypes.sizeof(winkeys._KeyBdInput)
    assert ctypes.sizeof(winkeys._InputUnion) == ctypes.sizeof(winkeys._MouseInput)


def test_press_is_a_noop_outside_windows() -> None:
    """Importar y llamar en la Mac tiene que ser seguro: los tests del router
    arrastran este modulo por la cadena de imports de las skills."""
    if not winkeys.IS_WINDOWS:
        assert winkeys.press(winkeys.VirtualKey.VOLUME_MUTE) is False
