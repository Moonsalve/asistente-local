"""Volumen de Windows: maestro y por aplicacion, via la API de audio (pycaw).

DOS NIVELES DISTINTOS
---------------------
Windows tiene dos mandos independientes y el usuario los percibe como cosas
distintas, asi que el asistente tambien los trata como tales:

- **Maestro** (`IAudioEndpointVolume`): el mando del dispositivo de salida. Es
  el que mueve la ruedecita del teclado y afecta a todo.
- **Por aplicacion** (`ISimpleAudioVolume`): la barra que sale en el mezclador
  de volumen para cada programa. Es *multiplicativa* sobre el maestro: con el
  maestro al 50% y Spotify al 100%, Spotify suena al 50% real.

"Sube el volumen de Spotify" es lo segundo. Hacerlo por aqui y no por la Web API
tiene dos ventajas: funciona aunque Spotify no tenga sesion de Connect activa, y
cuesta milisegundos en vez de un viaje de red.

Limite conocido: solo existe sesion mientras la aplicacion tiene audio abierto.
Spotify cerrado -o abierto pero sin haber sonado nunca- no aparece en el
mezclador. Para ese caso esta el fallback a la Web API en `skills/volume.py`.

POR QUE SE CACHEA EL ENDPOINT PERO NO LAS SESIONES
--------------------------------------------------
Activar el endpoint maestro cuesta una activacion COM (decenas de ms) y el
dispositivo de salida rara vez cambia, asi que se cachea y se invalida si una
llamada falla (que es justo lo que pasa al cambiar de dispositivo: la interfaz
vieja empieza a devolver errores). Las sesiones, en cambio, aparecen y
desaparecen con las aplicaciones, asi que se enumeran en cada llamada.

Fuera de Windows todo devuelve None/False sin lanzar, para poder importar y
testear en la Mac.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Sequence
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

        speakers = AudioUtilities.GetSpeakers()
        interface = speakers.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        _endpoint = cast(interface, POINTER(IAudioEndpointVolume))
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


def _matching_sessions(process_names: Sequence[str]) -> list[Any]:
    """Controles `ISimpleAudioVolume` de los procesos indicados.

    Devuelve *todos* los que casan, no el primero: Spotify abre varios procesos
    y no siempre es el mismo el que tiene el audio, asi que se les aplica el
    cambio a todos y asi el resultado no depende del orden de enumeracion.
    """
    if not _ensure_com():
        return []
    wanted = {name.lower().removesuffix(".exe") for name in process_names}
    found: list[Any] = []
    try:  # pragma: no cover - solo en el PC destino
        from pycaw.pycaw import AudioUtilities

        for session in AudioUtilities.GetAllSessions():
            if session.Process is None or session.SimpleAudioVolume is None:
                continue
            try:
                name = session.Process.name()
            except Exception:
                # El proceso murio entre enumerar y consultar. Normal.
                continue
            if name.lower().removesuffix(".exe") in wanted:
                found.append(session.SimpleAudioVolume)
    except ImportError as exc:
        _fail("falta pycaw/comtypes (pip install -e .)", exc)
    except Exception as exc:
        _fail("no se pudo enumerar el mezclador de volumen", exc)
    return found


def app_percent(process_names: Sequence[str]) -> int | None:
    """Volumen de la aplicacion en el mezclador (0-100), o None si no suena."""
    for control in _matching_sessions(process_names):
        try:  # pragma: no cover - solo en el PC destino
            return round(float(control.GetMasterVolume()) * 100)
        except Exception as exc:
            _fail("no se pudo leer el volumen de la aplicacion", exc)
    return None


def set_app_percent(process_names: Sequence[str], level: int) -> bool:
    """Fija el volumen de la aplicacion. False si no tiene sesion de audio."""
    scalar = _clamp(level) / 100.0
    changed = False
    for control in _matching_sessions(process_names):
        try:  # pragma: no cover - solo en el PC destino
            control.SetMasterVolume(scalar, None)
            changed = True
        except Exception as exc:
            _fail("no se pudo fijar el volumen de la aplicacion", exc)
    return changed


def app_muted(process_names: Sequence[str]) -> bool | None:
    for control in _matching_sessions(process_names):
        try:  # pragma: no cover - solo en el PC destino
            return bool(control.GetMute())
        except Exception as exc:
            _fail("no se pudo leer el silencio de la aplicacion", exc)
    return None


def set_app_muted(process_names: Sequence[str], muted: bool) -> bool:
    changed = False
    for control in _matching_sessions(process_names):
        try:  # pragma: no cover - solo en el PC destino
            control.SetMute(bool(muted), None)
            changed = True
        except Exception as exc:
            _fail("no se pudo silenciar la aplicacion", exc)
    return changed


def list_sessions() -> list[SessionInfo]:
    """Todo el mezclador, para `scripts/diagnose_volume.py`."""
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
