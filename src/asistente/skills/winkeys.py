"""Control de Windows: teclas multimedia y volumen del sistema.

Las teclas multimedia son la red de seguridad de todo el control de audio: no
necesitan autenticacion, funcionan con Spotify, YouTube Music o cualquier
reproductor, y cuestan menos de 5 ms. La Web API de Spotify es mas capaz (puede
buscar por nombre), pero falla cuando no hay dispositivo activo; en ese caso se
cae aqui.

Se usa `ctypes.SendInput` en lugar de `keybd_event` porque este ultimo esta
deprecado desde Windows Vista y no funciona con aplicaciones que leen la cola de
entrada moderna.

Todo este modulo es no-operativo fuera de Windows: importar en macOS/Linux es
seguro (para poder correr los tests del router), pero cada llamada devuelve
False en vez de reventar.
"""

from __future__ import annotations

import ctypes
import logging
import sys
from ctypes import wintypes
from enum import IntEnum

log = logging.getLogger(__name__)

IS_WINDOWS = sys.platform == "win32"


class VirtualKey(IntEnum):
    """Codigos de tecla virtual de Windows para control multimedia."""

    VOLUME_MUTE = 0xAD
    VOLUME_DOWN = 0xAE
    VOLUME_UP = 0xAF
    MEDIA_NEXT_TRACK = 0xB0
    MEDIA_PREV_TRACK = 0xB1
    MEDIA_STOP = 0xB2
    MEDIA_PLAY_PAUSE = 0xB3


if IS_WINDOWS:  # pragma: no cover - solo se ejecuta en el PC destino
    _INPUT_KEYBOARD = 1
    _KEYEVENTF_KEYUP = 0x0002

    class _KeyBdInput(ctypes.Structure):
        _fields_ = (
            ("wVk", wintypes.WORD),
            ("wScan", wintypes.WORD),
            ("dwFlags", wintypes.DWORD),
            ("time", wintypes.DWORD),
            ("dwExtraInfo", ctypes.POINTER(wintypes.ULONG)),
        )

    class _InputUnion(ctypes.Union):
        _fields_ = (("ki", _KeyBdInput),)

    class _Input(ctypes.Structure):
        _fields_ = (("type", wintypes.DWORD), ("union", _InputUnion))

    def _send_key(vk: VirtualKey) -> bool:
        events = (_Input * 2)(
            _Input(_INPUT_KEYBOARD, _InputUnion(ki=_KeyBdInput(wVk=vk, dwFlags=0))),
            _Input(
                _INPUT_KEYBOARD,
                _InputUnion(ki=_KeyBdInput(wVk=vk, dwFlags=_KEYEVENTF_KEYUP)),
            ),
        )
        sent = ctypes.windll.user32.SendInput(2, ctypes.byref(events), ctypes.sizeof(_Input))
        return sent == 2

else:

    def _send_key(vk: VirtualKey) -> bool:
        log.debug("SendInput(%s) ignorado: no estamos en Windows", vk.name)
        return False


def press(vk: VirtualKey) -> bool:
    """Envia una pulsacion completa (down + up). False si no hizo efecto."""
    return _send_key(vk)


def get_volume_percent() -> int | None:
    """Volumen maestro actual en 0-100, o None si no se puede leer.

    Usa la escala *escalar* de la API de Windows, no la de decibelios: la
    escalar es la que corresponde a lo que muestra el control de volumen.
    """
    endpoint = _audio_endpoint()
    if endpoint is None:
        return None
    return round(endpoint.GetMasterVolumeLevelScalar() * 100)


def set_volume_percent(level: int) -> bool:
    """Fija el volumen maestro. `level` se recorta a 0-100."""
    endpoint = _audio_endpoint()
    if endpoint is None:
        return False
    endpoint.SetMasterVolumeLevelScalar(max(0, min(100, level)) / 100.0, None)
    return True


def _audio_endpoint():  # noqa: ANN202 - tipo COM opaco
    """Interfaz IAudioEndpointVolume, o None fuera de Windows.

    El import es perezoso porque `pycaw` arrastra `comtypes`, que solo existe en
    Windows, y este modulo tiene que poder importarse en la Mac de desarrollo.
    """
    if not IS_WINDOWS:
        return None
    try:  # pragma: no cover - solo en el PC destino
        from comtypes import CLSCTX_ALL
        from ctypes import POINTER, cast

        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

        speakers = AudioUtilities.GetSpeakers()
        interface = speakers.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        return cast(interface, POINTER(IAudioEndpointVolume))
    except Exception:
        log.exception("no se pudo abrir el endpoint de audio")
        return None
