"""Volumen maestro de Windows, via la API de audio (pycaw).

`IAudioEndpointVolume` es el mando del dispositivo de salida: el que mueve la
ruedecita del teclado y afecta a todo lo que suena en el PC. Es el que se usa
cuando dices "sube el volumen" sin especificar nada mas.

POR QUE AQUI NO HAY VOLUMEN POR APLICACION
------------------------------------------
Windows tiene un segundo mando, `ISimpleAudioVolume`, que es la barra del
mezclador de volumen para cada programa. Se llego a usar para "baja el volumen
de Spotify" porque es instantaneo y no necesita autenticacion, y **se quito a
proposito**: no es el mando que la gente quiere decir. El del mezclador solo
afecta a lo que este PC saca por los altavoces, no se ve desde ninguna parte de
Spotify y no se sincroniza con el movil. "El volumen de Spotify" es el de dentro
de Spotify, y ese solo se toca por la Web API (ver `skills/volume.py`).

Queda `list_sessions()`, que solo lee, porque saber que aplicaciones tienen
audio abierto es util en el diagnostico.

POR QUE SE CACHEA EL ENDPOINT
-----------------------------
Activarlo cuesta una activacion COM (decenas de ms) y el dispositivo de salida
rara vez cambia, asi que se cachea y se invalida si una llamada falla — que es
justo lo que pasa al cambiar de altavoces a auriculares: la interfaz vieja sigue
viva pero empieza a devolver errores.

Fuera de Windows todo devuelve None/False sin lanzar, para poder importar y
testear en la Mac.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)

IS_WINDOWS = sys.platform == "win32"

_endpoint: Any | None = None
_com_ready = False
_last_error: str | None = None


@dataclass(frozen=True, slots=True)
class SessionInfo:
    """Una entrada del mezclador de volumen. Solo la usa el diagnostico."""

    process: str
    percent: int
    muted: bool


def last_error() -> str | None:
    """Ultimo fallo legible, para que el diagnostico pueda explicarse."""
    return _last_error


def pycaw_version() -> str:
    """Version instalada de pycaw, o por que no se pudo saber.

    Se imprime en el diagnostico porque pycaw ha cambiado la forma de su API
    entre versiones sin avisar (ver `_master`), y saber cual hay puesta ahorra
    la mitad del trabajo cuando algo deja de funcionar de un dia para otro.
    """
    try:
        from importlib.metadata import version

        return version("pycaw")
    except Exception as exc:
        return f"desconocida ({exc})"


def _fail(message: str, exc: Exception | None = None) -> None:
    global _last_error
    _last_error = f"{message}: {exc}" if exc else message
    if exc is not None:
        log.debug("%s", _last_error, exc_info=exc)


def _clamp(level: int) -> int:
    return max(0, min(100, level))


def _ensure_com() -> bool:
    """Inicializa COM una vez en el hilo actual.

    Las skills se ejecutan siempre en el hilo principal del pipeline, asi que
    con una vez basta. Se hace explicito en lugar de confiar en que comtypes lo
    haya hecho por su cuenta al importarse.
    """
    global _com_ready
    if _com_ready:
        return True
    if not IS_WINDOWS:
        # Se registra como error para que el diagnostico tenga algo que decir en
        # vez de imprimir "no se pudo leer: None".
        _fail("la API de audio de Windows no existe en esta plataforma")
        return False
    try:  # pragma: no cover - solo en el PC destino
        import comtypes

        comtypes.CoInitialize()
    except OSError as exc:  # ya inicializado con otro modelo de apartamento
        log.debug("CoInitialize devolvio %s; se continua", exc)
    except Exception as exc:
        _fail("no se pudo inicializar COM", exc)
        return False
    _com_ready = True
    return True


def endpoint_from_speakers(
    speakers: Any,
    endpoint_iface: Any,
    clsctx_all: Any,
    cast: Any,
    pointer: Any,
) -> Any:
    """Saca `IAudioEndpointVolume` de lo que sea que devuelva `GetSpeakers()`.

    pycaw cambio la forma de esa llamada sobre la marcha: hasta 2024 devolvia el
    `IMMDevice` crudo, que hay que activar a mano, y desde 20251023 devuelve un
    envoltorio `AudioDevice` que ya expone la interfaz hecha. Se aceptan las dos
    porque el pyproject solo pide `pycaw>=20240210` y una resolucion de version
    distinta en el PC no puede dejar el volumen sin funcionar.

    Sintoma de no contemplarlo, y motivo de que esta funcion exista:
    `'AudioDevice' object has no attribute 'Activate'`, con el volumen del PC
    degradado a las teclas multimedia sin mas aviso.

    Las dependencias de COM llegan por parametro para poder testear la eleccion
    de rama fuera de Windows, que es donde estuvo el fallo.
    """
    if hasattr(speakers, "EndpointVolume"):
        return speakers.EndpointVolume
    interface = speakers.Activate(endpoint_iface._iid_, clsctx_all, None)
    return cast(interface, pointer(endpoint_iface))


def _master() -> Any | None:
    """Interfaz `IAudioEndpointVolume` del dispositivo de salida por defecto."""
    global _endpoint
    if _endpoint is not None:
        return _endpoint
    if not _ensure_com():
        return None
    try:  # pragma: no cover - solo en el PC destino
        from ctypes import POINTER, cast

        from comtypes import CLSCTX_ALL
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

        _endpoint = endpoint_from_speakers(
            AudioUtilities.GetSpeakers(),
            IAudioEndpointVolume,
            CLSCTX_ALL,
            cast,
            POINTER,
        )
    except ImportError as exc:
        _fail("falta pycaw/comtypes (pip install -e .)", exc)
        return None
    except Exception as exc:
        _fail("no se pudo abrir el dispositivo de salida", exc)
        return None
    return _endpoint


def _with_master(action: str, call: Any) -> Any:
    """Ejecuta `call(endpoint)` reintentando una vez con el endpoint renovado.

    El reintento existe por el caso real de cambiar de altavoces a auriculares:
    la interfaz cacheada sigue viva pero empieza a devolver errores, y no hay
    forma barata de detectarlo salvo intentarlo.
    """
    global _endpoint
    for attempt in (1, 2):
        endpoint = _master()
        if endpoint is None:
            return None
        try:  # pragma: no cover - solo en el PC destino
            return call(endpoint)
        except Exception as exc:
            _endpoint = None
            if attempt == 2:
                _fail(f"fallo {action} sobre el volumen maestro", exc)
                return None
    return None


def master_percent() -> int | None:
    """Volumen maestro en 0-100, o None si no se puede leer.

    Usa la escala *escalar* de Windows, no la de decibelios: la escalar es la
    que corresponde a lo que muestra el control de volumen.
    """
    value = _with_master("leer", lambda e: e.GetMasterVolumeLevelScalar())
    return None if value is None else round(float(value) * 100)


def set_master_percent(level: int) -> bool:
    """Fija el volumen maestro. `level` se recorta a 0-100."""
    scalar = _clamp(level) / 100.0
    done = _with_master("escribir", lambda e: e.SetMasterVolumeLevelScalar(scalar, None) or True)
    return bool(done)


def master_muted() -> bool | None:
    value = _with_master("leer el silencio", lambda e: e.GetMute())
    return None if value is None else bool(value)


def set_master_muted(muted: bool) -> bool:
    done = _with_master("silenciar", lambda e: e.SetMute(bool(muted), None) or True)
    return bool(done)


def list_sessions() -> list[SessionInfo]:
    """Que aplicaciones tienen audio abierto. Solo lectura, y solo la usa
    `scripts/diagnose_volume.py`: sirve para confirmar que Spotify esta de
    verdad sonando cuando un comando de volumen no responde."""
    if not _ensure_com():
        return []
    out: list[SessionInfo] = []
    try:  # pragma: no cover - solo en el PC destino
        from pycaw.pycaw import AudioUtilities

        for session in AudioUtilities.GetAllSessions():
            control = session.SimpleAudioVolume
            if control is None:
                continue
            try:
                name = session.Process.name() if session.Process else "(sonidos del sistema)"
                out.append(
                    SessionInfo(
                        process=name,
                        percent=round(float(control.GetMasterVolume()) * 100),
                        muted=bool(control.GetMute()),
                    )
                )
            except Exception:
                continue
    except Exception as exc:
        _fail("no se pudo enumerar el mezclador de volumen", exc)
    return out
