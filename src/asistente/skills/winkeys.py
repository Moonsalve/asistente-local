"""Teclas multimedia de Windows via SendInput.

Son la red de seguridad del control de audio: no necesitan autenticacion,
funcionan con Spotify, YouTube Music o cualquier reproductor, y cuestan menos de
5 ms. La Web API de Spotify es mas capaz (puede buscar por nombre), pero falla
cuando no hay dispositivo activo; en ese caso se cae aqui.

El volumen NO se toca desde aqui: vive en `winaudio.py`, que usa la API de audio
de Windows y puede fijar un porcentaje exacto en vez de moverse a saltos de 2%.
Este modulo solo manda pulsaciones.

Se usa `SendInput` en lugar de `keybd_event` porque este ultimo esta deprecado
desde Windows Vista y no funciona con aplicaciones que leen la cola de entrada
moderna.

EL TAMANO DE `INPUT` NO ES NEGOCIABLE
-------------------------------------
`SendInput` valida que `cbSize` sea exactamente `sizeof(INPUT)` y devuelve 0 con
`ERROR_INVALID_PARAMETER` si no cuadra. En x64 son **40 bytes**, y esa cifra sale
de la union completa: el miembro mas grande es `MOUSEINPUT` (32 bytes), no
`KEYBDINPUT` (24). Declarar la union solo con `ki` -que es lo unico que este
modulo usa- da 32 bytes y hace que *todas* las pulsaciones fallen en silencio.
Por eso estan los tres miembros aunque solo se rellene uno. `tests/test_winkeys.py`
fija el tamano para que no vuelva a pasar.

Todo este modulo es no-operativo fuera de Windows: importar en macOS/Linux es
seguro (para poder correr los tests del router), pero cada llamada devuelve
False en vez de reventar.
"""

from __future__ import annotations

import ctypes
import logging
import sys
from enum import IntEnum

log = logging.getLogger(__name__)

IS_WINDOWS = sys.platform == "win32"

#: Lo que `SendInput` espera en `cbSize`. Se calcula igual en todas las
#: plataformas para poder comprobarlo desde los tests en la Mac.
_POINTER_SIZE = ctypes.sizeof(ctypes.c_void_p)
EXPECTED_INPUT_SIZE = 40 if _POINTER_SIZE == 8 else 28


class VirtualKey(IntEnum):
    """Codigos de tecla virtual de Windows para control multimedia."""

    VOLUME_MUTE = 0xAD
    VOLUME_DOWN = 0xAE
    VOLUME_UP = 0xAF
    MEDIA_NEXT_TRACK = 0xB0
    MEDIA_PREV_TRACK = 0xB1
    MEDIA_STOP = 0xB2
    MEDIA_PLAY_PAUSE = 0xB3


_INPUT_KEYBOARD = 1
_KEYEVENTF_KEYUP = 0x0002

# Tipos equivalentes a los de `wintypes`, definidos a mano porque
# `ctypes.wintypes` solo se puede importar en Windows y estas estructuras tienen
# que ser inspeccionables desde los tests de la Mac.
_WORD = ctypes.c_uint16
_DWORD = ctypes.c_uint32
_LONG = ctypes.c_int32
_ULONG_PTR = ctypes.c_uint64 if _POINTER_SIZE == 8 else ctypes.c_uint32


class _MouseInput(ctypes.Structure):
    """No se usa, pero es el miembro mas grande de la union y por tanto el que
    fija el tamano de `INPUT`. Quitarlo rompe SendInput entero."""

    _fields_ = (
        ("dx", _LONG),
        ("dy", _LONG),
        ("mouseData", _DWORD),
        ("dwFlags", _DWORD),
        ("time", _DWORD),
        ("dwExtraInfo", _ULONG_PTR),
    )


class _KeyBdInput(ctypes.Structure):
    _fields_ = (
        ("wVk", _WORD),
        ("wScan", _WORD),
        ("dwFlags", _DWORD),
        ("time", _DWORD),
        ("dwExtraInfo", _ULONG_PTR),
    )


class _HardwareInput(ctypes.Structure):
    _fields_ = (("uMsg", _DWORD), ("wParamL", _WORD), ("wParamH", _WORD))


class _InputUnion(ctypes.Union):
    _fields_ = (("mi", _MouseInput), ("ki", _KeyBdInput), ("hi", _HardwareInput))


class _Input(ctypes.Structure):
    _fields_ = (("type", _DWORD), ("union", _InputUnion))


def input_struct_size() -> int:
    """Tamano real de `INPUT`. Lo consulta `scripts/diagnose_volume.py` para
    poder decir 'las teclas no funcionan y este es el motivo' sin adivinar."""
    return ctypes.sizeof(_Input)


if IS_WINDOWS:  # pragma: no cover - solo se ejecuta en el PC destino
    _user32 = ctypes.WinDLL("user32", use_last_error=True)
    _user32.SendInput.argtypes = (ctypes.c_uint, ctypes.POINTER(_Input), ctypes.c_int)
    _user32.SendInput.restype = ctypes.c_uint

    def _send_key(vk: VirtualKey) -> bool:
        events = (_Input * 2)(
            _Input(type=_INPUT_KEYBOARD, union=_InputUnion(ki=_KeyBdInput(wVk=vk, dwFlags=0))),
            _Input(
                type=_INPUT_KEYBOARD,
                union=_InputUnion(ki=_KeyBdInput(wVk=vk, dwFlags=_KEYEVENTF_KEYUP)),
            ),
        )
        sent = _user32.SendInput(2, events, ctypes.sizeof(_Input))
        if sent != 2:
            # Los dos motivos reales: cbSize incorrecto (error 87), o UIPI
            # bloqueando la inyeccion porque la ventana en primer plano corre
            # elevada y el asistente no (error 5).
            log.warning(
                "SendInput(%s) inyecto %d de 2 eventos (error %d)",
                vk.name,
                sent,
                ctypes.get_last_error(),
            )
            return False
        return True

else:

    def _send_key(vk: VirtualKey) -> bool:
        log.debug("SendInput(%s) ignorado: no estamos en Windows", vk.name)
        return False


def press(vk: VirtualKey) -> bool:
    """Envia una pulsacion completa (down + up). False si no hizo efecto."""
    return _send_key(vk)
