"""Lo que cambia cuando el asistente NO corre desde una terminal.

Arrancar con `pythonw.exe` en vez de `python.exe` quita la ventana de consola,
que es justo lo que se quiere para un proceso de fondo. Pero se lleva por
delante tres cosas que el codigo daba por sentadas, y las tres fallan en
silencio, que es el peor modo de fallo posible:

1. NO HAY stdout NI stderr. Bajo `pythonw` valen literalmente `None`. El
   handler de logging escribe en el vacio, `logsetup` se traga la excepcion y
   el asistente arranca perfectamente sin que quede rastro de nada. Cuando algo
   va mal no hay ni por donde empezar.

2. EL DIRECTORIO DE TRABAJO ES ARBITRARIO. Un acceso directo de Windows lanza
   el proceso con el `WorkingDirectory` que le pongas, y si se lanza desde el
   Inicio puede ser `C:\\Windows\\System32`. Todo el proyecto usa rutas
   relativas -`config.yaml`, `commands.yaml`, `models/es_MX-...onnx`,
   `voz.json`- asi que desde otro directorio no encuentra NADA.

3. NO HAY Ctrl-C. El bucle es `while True` y la unica forma de pararlo seria el
   Administrador de tareas. Un proceso con el microfono abierto que no se puede
   cerrar no es aceptable: por eso existe `tray.py`.

Este modulo resuelve 1 y 2. El 3 lo resuelven `tray.py` y el evento de parada
de `pipeline.py`.

POR QUE NO SE EMPAQUETA CON PyInstaller
---------------------------------------
Es la pregunta obvia, y la respuesta esta en `cuda_setup.py`: las DLL de CUDA
se localizan EN TIEMPO DE EJECUCION recorriendo `site.getsitepackages()` en
busca de `nvidia/cublas/bin`. Dentro de un binario congelado no hay
site-packages, `getsitepackages()` no devuelve nada util y el STT cae con
"cublas64_12.dll is not found" sin que el aviso llegue a ninguna parte.

Arreglarlo obliga a reescribir `cuda_setup` para el modo congelado Y a empotrar
~2 GB de DLL de NVIDIA en el bundle. A cambio de que? De no tener que escribir
la ruta del interprete en un acceso directo. No sale a cuenta: un `.lnk` a
`pythonw.exe` da el mismo doble clic sin ventana, y ademas sigue usando el venv
de verdad, con lo que un `pip install` sigue surtiendo efecto sin reempaquetar.
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

#: Nombre corto de la aplicacion. Se usa para el directorio de datos, el mutex
#: de instancia unica y el titulo del icono de la bandeja.
APP_NAME = "Apolo"

#: Ficheros que identifican la raiz del proyecto. Se buscan los dos: `models/`
#: podria no existir todavia y `src/` tambien esta en cualquier checkout.
_ROOT_MARKERS = ("config.yaml", "commands.yaml")


def app_root() -> Path:
    """Directorio raiz del proyecto: el que contiene `config.yaml`.

    Se busca hacia arriba desde este fichero en vez de contar directorios
    (`parents[2]`) porque contar se rompe en silencio en cuanto alguien mueve un
    modulo de sitio. El marcador es un dato, no una posicion.
    """
    if getattr(sys, "frozen", False):  # por si algun dia se empaqueta
        return Path(sys.executable).resolve().parent

    here = Path(__file__).resolve()
    for parent in here.parents:
        if all((parent / marker).is_file() for marker in _ROOT_MARKERS):
            return parent
    # Instalado como wheel fuera del checkout: el usuario tendra que arrancar
    # desde el directorio que contenga su config.
    return Path.cwd()


def data_dir() -> Path:
    """Directorio para lo que el asistente ESCRIBE: registros, icono, bloqueo.

    Fuera del repositorio a proposito. El repo puede estar en una carpeta
    sincronizada o de solo lectura, y unos registros que rotan dentro de un
    checkout de git son basura que tarde o temprano acaba en un commit.
    """
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    else:
        base = Path(os.environ.get("XDG_STATE_HOME") or Path.home() / ".local" / "state")
    return base / APP_NAME


def log_dir() -> Path:
    directory = data_dir() / "logs"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def has_console() -> bool:
    """`False` cuando el proceso corre bajo `pythonw` o congelado sin consola.

    La comprobacion es literalmente la condicion que importa: si `sys.stderr`
    es `None`, un `StreamHandler` no tiene donde escribir. No se usa `isatty()`
    porque redirigir la salida a un fichero desde una terminal sigue siendo una
    salida perfectamente valida.
    """
    return sys.stderr is not None


def anchor_working_directory() -> Path:
    """Situa el proceso en la raiz del proyecto. Devuelve el directorio.

    UNA linea que arregla TODAS las rutas relativas del proyecto a la vez, en
    vez de ir parcheando `config.yaml`, `commands.yaml`, la voz de Piper, el
    perfil de voz y la cache de descubrimiento uno por uno. Es process-wide, y
    por eso solo se llama desde `__main__`: una biblioteca no tiene derecho a
    hacer esto, un punto de entrada si.
    """
    root = app_root()
    os.chdir(root)
    return root


def notify(title: str, message: str, *, blocking: bool = True) -> None:
    """Avisa al usuario cuando no hay consola donde imprimir el problema.

    Sin esto, un arranque que falla en el preflight -falta la voz de Piper, por
    ejemplo- se traduce en que haces doble clic y no pasa absolutamente nada.
    El registro lo explica, pero hay que saber que existe y donde esta.

    `blocking=False` para lo que NO es fatal. `MessageBoxW` no vuelve hasta que
    alguien pulsa Aceptar, asi que un aviso de arranque llamado desde el hilo
    principal dejaria el asistente sin cargar hasta que alguien mire la
    pantalla. Lo fatal si bloquea: ahi no queda nada por hacer, y si el proceso
    se muriera antes de que el cuadro aparezca no se veria el error.

    En cualquier otro sitio, o si el cuadro de dialogo falla, se cae a stderr:
    esta funcion no puede ser nunca la causa de que un error no se vea.
    """
    if not blocking:
        threading.Thread(
            target=notify, args=(title, message), name="notify", daemon=True
        ).start()
        return

    if sys.platform == "win32":
        try:
            import ctypes

            # MB_OK | MB_ICONERROR | MB_SETFOREGROUND
            ctypes.windll.user32.MessageBoxW(None, message, title, 0x10 | 0x10000)
            return
        except Exception:
            pass

    if sys.stderr is not None:
        print(f"{title}: {message}", file=sys.stderr)
