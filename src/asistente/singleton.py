"""Una sola instancia del asistente a la vez.

POR QUE HACE FALTA
------------------
Mientras se arrancaba desde la terminal, lanzar dos era dificil de hacer sin
querer: veias la ventana. Con un acceso directo en el escritorio Y otro en el
Inicio de Windows, lanzar el segundo por costumbre es cuestion de dias. Y dos
instancias no se estorban de forma visible, se estorban de forma confusa:

  - Las dos abren el microfono. Windows lo permite, asi que las dos oyen la
    misma orden y las dos la ejecutan. Pides la siguiente cancion y saltan dos.
  - Las dos cargan Whisper en VRAM: 1.6 GB de mas sobre un presupuesto de 8 GB
    que ya iba al 85%. Ollama se queda sin sitio y descarga capas a RAM, con lo
    que el LLM pasa de 300 ms a varios segundos.
  - Las dos hablan por Piper a la vez, encima.

El sintoma que llega al usuario -"a veces se vuelve lentisimo y repite todo"-
no apunta ni de lejos a la causa. Mejor negarse a arrancar y decirlo.

COMO
----
En Windows, un mutex con nombre del kernel. Es el mecanismo correcto y no el
tipico fichero de bloqueo con el PID dentro: si el proceso muere de forma
brusca -Administrador de tareas, corte de luz- el kernel libera el mutex solo,
mientras que el fichero se queda ahi y la siguiente ejecucion se cree que ya
hay una instancia viva. Un bloqueo obsoleto que impide arrancar es peor que no
tener bloqueo.

Fuera de Windows, `flock`, que tiene la misma propiedad: el bloqueo cuelga del
descriptor abierto, no del contenido del fichero, y el sistema lo suelta al
morir el proceso. Existe sobre todo para que esto se pueda probar en el Mac.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from types import TracebackType

from asistente.runtime import APP_NAME, data_dir

log = logging.getLogger(__name__)

#: Sin prefijo `Global\`, o sea espacio de nombres de la sesion. Es lo que
#: queremos: dos usuarios distintos con sesion iniciada a la vez pueden tener
#: cada uno su asistente, y ademas `Global\` requiere privilegios que no
#: necesitamos para nada.
_MUTEX_NAME = f"{APP_NAME}-single-instance"

_ERROR_ALREADY_EXISTS = 183


class SingleInstance:
    """Bloqueo de instancia unica. Usar como gestor de contexto.

        with SingleInstance() as lock:
            if not lock.acquired:
                return 1
    """

    def __init__(self, name: str = _MUTEX_NAME) -> None:
        self._name = name
        self._handle: object | None = None
        self._file: object | None = None
        self.acquired = False

    def acquire(self) -> bool:
        if self.acquired:
            return True
        windows = sys.platform == "win32"
        self.acquired = self._acquire_windows() if windows else self._acquire_posix()
        return self.acquired

    def _acquire_windows(self) -> bool:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.CreateMutexW(None, False, self._name)
        if not handle:
            # Sin mutex no podemos garantizar nada. Se deja arrancar: negar el
            # arranque por un fallo del propio guardia seria peor que el riesgo
            # que el guardia evita.
            log.warning("no se pudo crear el mutex de instancia unica; se sigue sin comprobar")
            return True
        if kernel32.GetLastError() == _ERROR_ALREADY_EXISTS:
            kernel32.CloseHandle(handle)
            return False
        self._handle = handle
        return True

    def _acquire_posix(self) -> bool:
        import fcntl

        directory = data_dir()
        directory.mkdir(parents=True, exist_ok=True)
        handle = (directory / f"{self._name}.lock").open("w")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            handle.close()
            return False
        self._file = handle
        return True

    def release(self) -> None:
        if self._handle is not None:
            import ctypes

            ctypes.windll.kernel32.CloseHandle(self._handle)
            self._handle = None
        if self._file is not None:
            self._file.close()  # type: ignore[attr-defined]
            self._file = None
        self.acquired = False

    def __enter__(self) -> SingleInstance:
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.release()


def lock_path() -> Path:
    """Ruta del fichero de bloqueo POSIX. Solo para diagnostico y pruebas."""
    return data_dir() / f"{_MUTEX_NAME}.lock"
