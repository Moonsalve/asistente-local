"""Localiza las DLL de CUDA en Windows antes de cargar CTranslate2.

EL PROBLEMA
-----------
faster-whisper corre sobre CTranslate2, que enlaza contra cuBLAS y cuDNN de
CUDA 12. Esas librerias no vienen con el driver de NVIDIA: hay que instalarlas
aparte. La via comoda son los paquetes pip `nvidia-cublas-cu12` y
`nvidia-cudnn-cu12`, pero dejan las DLL en

    <venv>/Lib/site-packages/nvidia/cublas/bin/
    <venv>/Lib/site-packages/nvidia/cudnn/bin/

y Windows no las busca ahi por su cuenta. El sintoma es exactamente:

    RuntimeError: Library cublas64_12.dll is not found or cannot be loaded

HAY QUE HACER DOS COSAS, NO UNA
-------------------------------
Windows tiene dos mecanismos de busqueda de DLL y cada uno mira sitios
distintos:

  1. `os.add_dll_directory()` -> lo usan las extensiones de Python (.pyd) al
     resolver su tabla de importacion.
  2. `PATH` -> lo usa `LoadLibrary` cuando lo llama codigo nativo por su cuenta.

CTranslate2 carga cuBLAS desde su propio codigo C++, o sea por la via 2. Por eso
registrar solo los directorios con `add_dll_directory` NO basta: los directorios
aparecen como registrados en el log y la carga sigue fallando, que es
exactamente lo que se observo. Aqui se hacen las dos.

Este modulo tiene que ejecutarse ANTES del primer `import faster_whisper`, y por
eso `transcriber.py` lo llama en su propio import.

En Linux y macOS no hace nada: alli el enlazador dinamico ya resuelve por
RPATH y no existe `os.add_dll_directory`.
"""

from __future__ import annotations

import logging
import os
import site
import sys
from pathlib import Path

log = logging.getLogger(__name__)

#: Paquetes pip de NVIDIA que traen DLL que CTranslate2 necesita cargar.
_NVIDIA_DLL_PACKAGES = ("cublas", "cudnn", "cuda_runtime")

_already_configured = False


def _candidate_roots() -> list[Path]:
    roots = [Path(p) for p in site.getsitepackages()]
    if (user_site := site.getusersitepackages()) and isinstance(user_site, str):
        roots.append(Path(user_site))
    # En un venv, site.getsitepackages() a veces devuelve el prefijo base en vez
    # del del entorno; anadir la ruta derivada de sys.prefix lo cubre.
    roots.append(Path(sys.prefix) / "Lib" / "site-packages")
    return roots


def ensure_cuda_dlls() -> list[Path]:
    """Registra los directorios de DLL de NVIDIA. Devuelve los que encontro.

    Idempotente: llamarla varias veces no duplica registros.
    """
    global _already_configured
    if sys.platform != "win32" or _already_configured:
        return []

    seen: set[Path] = set()
    for root in _candidate_roots():
        for package in _NVIDIA_DLL_PACKAGES:
            dll_dir = root / "nvidia" / package / "bin"
            if not dll_dir.is_dir() or dll_dir in seen:
                continue
            seen.add(dll_dir)
            os.add_dll_directory(str(dll_dir))
            log.debug("CUDA: registrado %s", dll_dir)

    if seen:
        # La mitad que faltaba. CTranslate2 llama a LoadLibrary desde su codigo
        # C++, y esa via ignora add_dll_directory pero SI mira el PATH.
        _prepend_to_path(seen)

    _already_configured = True
    if not seen:
        log.warning(
            "no se encontraron las DLL de CUDA de los paquetes pip de NVIDIA. "
            'Si el STT falla con "cublas64_12.dll is not found": '
            'pip install -e ".[gpu]"'
        )
    else:
        _verify_loadable(seen)
    return sorted(seen)


def _prepend_to_path(directories: set[Path]) -> None:
    """Antepone los directorios al PATH del proceso.

    Antepone y no anade: si hay otra version de CUDA instalada en el sistema,
    queremos que gane la de los paquetes pip, que es la que corresponde a la
    version de CTranslate2 instalada.
    """
    current = os.environ.get("PATH", "")
    existing = set(current.split(os.pathsep))
    missing = [str(d) for d in sorted(directories) if str(d) not in existing]
    if missing:
        os.environ["PATH"] = os.pathsep.join([*missing, current])
        log.info("CUDA: %d directorio(s) anadidos al PATH del proceso", len(missing))


def _verify_loadable(directories: set[Path]) -> None:
    """Comprueba de verdad que Windows puede cargar la DLL clave.

    Registrar los directorios no garantiza nada: hasta ahora el log decia
    "registrado" y la carga fallaba igual. Intentarlo aqui convierte el
    diagnostico en una respuesta en vez de una suposicion, y ademas deja la
    DLL ya cargada en el proceso, con lo que CTranslate2 la encuentra sin
    volver a buscarla.
    """
    import ctypes

    key_dll = "cublas64_12.dll"
    if not any((d / key_dll).is_file() for d in directories):
        log.warning(
            "CUDA: los directorios estan pero falta %s. "
            "Diagnostico: python scripts/diagnose_cuda.py",
            key_dll,
        )
        return

    try:
        ctypes.WinDLL(key_dll)
    except OSError as exc:
        log.warning("CUDA: %s existe pero Windows no puede cargarla: %s", key_dll, exc)
        log.warning("CUDA: diagnostico completo con  python scripts/diagnose_cuda.py")
        return
    log.info("CUDA: %s cargada correctamente", key_dll)
