"""Diagnostico de la instalacion de CUDA en Windows.

Responde de forma definitiva por que CTranslate2 no encuentra sus DLL, en vez de
adivinar. Comprueba, en orden:

  1. Que interprete es y si es un venv.
  2. Que paquetes pip de NVIDIA estan instalados y donde.
  3. Que DLL existen realmente en disco (no solo el directorio).
  4. Si Windows es capaz de cargarlas tras registrar los directorios.
  5. Si CTranslate2 ve la GPU.

Uso:  python scripts/diagnose_cuda.py
"""

from __future__ import annotations

import ctypes
import site
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

#: Las que CTranslate2 necesita para CUDA 12. cudnn_ops es la que falta mas a
#: menudo: cuDNN 8 la llamaba distinto y muchas guias antiguas instalan esa.
REQUIRED_DLLS = {
    "cublas": ["cublas64_12.dll", "cublasLt64_12.dll"],
    "cudnn": ["cudnn64_9.dll", "cudnn_ops64_9.dll"],
    "cuda_runtime": ["cudart64_12.dll"],
}


def _roots() -> list[Path]:
    roots = [Path(p) for p in site.getsitepackages()]
    roots.append(Path(sys.prefix) / "Lib" / "site-packages")
    return list(dict.fromkeys(roots))


def main() -> int:
    print("=" * 78)
    print("1. INTERPRETE")
    print("=" * 78)
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    print(f"  ejecutable : {sys.executable}")
    print(f"  version    : {sys.version.split()[0]}")
    print(f"  venv       : {'SI' if in_venv else 'NO  <-- las dependencias pueden estar en otro sitio'}")
    if sys.platform != "win32":
        print("\n  No estamos en Windows: el resto del diagnostico no aplica.")
        return 0

    print("\n" + "=" * 78)
    print("2. PAQUETES pip DE NVIDIA")
    print("=" * 78)
    try:
        from importlib.metadata import version

        for package in ("nvidia-cublas-cu12", "nvidia-cudnn-cu12", "nvidia-cuda-runtime-cu12"):
            try:
                print(f"  {package:<28} {version(package)}")
            except Exception:
                print(f"  {package:<28} NO INSTALADO")
    except Exception as exc:  # noqa: BLE001
        print(f"  no se pudo consultar: {exc}")

    print("\n" + "=" * 78)
    print("3. DLL EN DISCO")
    print("=" * 78)
    found_dirs: list[Path] = []
    missing: list[str] = []
    for root in _roots():
        for package, dlls in REQUIRED_DLLS.items():
            dll_dir = root / "nvidia" / package / "bin"
            if not dll_dir.is_dir():
                continue
            found_dirs.append(dll_dir)
            print(f"  {dll_dir}")
            for dll in dlls:
                exists = (dll_dir / dll).is_file()
                print(f"      {'OK ' if exists else 'FALTA'} {dll}")
                if not exists:
                    missing.append(dll)

    if not found_dirs:
        print("  NINGUN directorio nvidia/*/bin encontrado.")
        print('  -> pip install -e ".[gpu]"')

    print("\n" + "=" * 78)
    print("4. CARGA REAL DE LAS DLL")
    print("=" * 78)
    from asistente.cuda_setup import ensure_cuda_dlls

    registered = ensure_cuda_dlls()
    print(f"  directorios registrados: {len(registered)}")
    for path in registered:
        print(f"      {path}")

    for package, dlls in REQUIRED_DLLS.items():
        for dll in dlls:
            try:
                ctypes.WinDLL(dll)
                print(f"  OK    {dll} se carga correctamente")
            except OSError as exc:
                print(f"  ERROR {dll}: {exc}")

    print("\n" + "=" * 78)
    print("5. CTRANSLATE2")
    print("=" * 78)
    try:
        import ctranslate2

        print(f"  version          : {ctranslate2.__version__}")
        print(f"  GPUs que detecta : {ctranslate2.get_cuda_device_count()}")
        types = ctranslate2.get_supported_compute_types("cuda")
        print(f"  compute types    : {sorted(types)}")
        if "int8_float16" not in types:
            print("  AVISO: int8_float16 no soportado; usa int8_float32 en config.yaml")
    except Exception as exc:  # noqa: BLE001
        print(f"  ERROR: {type(exc).__name__}: {exc}")

    print("\n" + "=" * 78)
    if missing:
        print("VEREDICTO: faltan DLL en disco ->", ", ".join(sorted(set(missing))))
        print('Reinstala con:  pip install --force-reinstall -e ".[gpu]"')
    elif not found_dirs:
        print('VEREDICTO: los paquetes CUDA no estan instalados en este interprete.')
        print('Instala con:  pip install -e ".[gpu]"')
    else:
        print("VEREDICTO: las DLL estan. Mira el apartado 4 para ver cual falla al cargar.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
