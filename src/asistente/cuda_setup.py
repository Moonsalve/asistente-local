"""Localiza las DLL de CUDA en Windows antes de cargar CTranslate2.

EL PROBLEMA
-----------
faster-whisper corre sobre CTranslate2, que enlaza contra cuBLAS y cuDNN de
CUDA 12. Esas librerias no vienen con el driver de NVIDIA: hay que instalarlas
aparte. La via comoda son los paquetes pip `nvidia-cublas-cu12` y
`nvidia-cudnn-cu12`, pero dejan las DLL en

    <venv>/Lib/site-packages/nvidia/cublas/bin/
    <venv>/Lib/site-packages/nvidia/cudnn/bin/

y desde Python 3.8 Windows YA NO busca DLL en el PATH del proceso: hay que
declarar los directorios explicitamente con `os.add_dll_directory`. Sin esto el
sintoma es exactamente:

    RuntimeError: Library cublas64_12.dll is not found or cannot be loaded

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
            # A INFO y no a DEBUG: cuando el STT falla, lo primero que se
            # quiere saber es si estos directorios se registraron o no.
            log.info("CUDA: registrado %s (%d dll)", dll_dir, len(list(dll_dir.glob("*.dll"))))

    _already_configured = True
    if not seen:
        log.warning(
            "no se encontraron las DLL de CUDA de los paquetes pip de NVIDIA. "
            'Si el STT falla con "cublas64_12.dll is not found": '
            'pip install -e ".[gpu]"'
        )
    elif not any((d / "cublas64_12.dll").is_file() for d in seen):
        # Los directorios existen pero no contienen la DLL concreta que
        # CTranslate2 busca: instalacion incompleta o version de CUDA distinta.
        log.warning(
            "CUDA: los directorios de NVIDIA estan pero falta cublas64_12.dll. "
            "Diagnostico completo: python scripts/diagnose_cuda.py"
        )
    return sorted(seen)
