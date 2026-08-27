"""Crea los accesos directos de Windows para arrancar Apolo sin terminal.

    python scripts/install_windows.py                 escritorio + inicio de sesion
    python scripts/install_windows.py --no-autostart  solo el del escritorio
    python scripts/install_windows.py --uninstall     quita los dos

QUE CREA, EXACTAMENTE
---------------------
Dos accesos directos identicos a `pythonw.exe -m asistente`, con el directorio
de trabajo puesto en la raiz del proyecto y un icono generado a partir del mismo
dibujo que usa la bandeja del sistema:

    Escritorio\\Apolo.lnk    doble clic para arrancarlo cuando quieras
    Inicio\\Apolo.lnk        arranca solo al iniciar sesion

POR QUE `pythonw.exe` Y NO UN .exe EMPAQUETADO
----------------------------------------------
`pythonw.exe` es el interprete de Python sin consola. Un acceso directo a el se
comporta como cualquier programa -doble clic, icono propio, sin ventana negra- y
ademas sigue usando el venv del proyecto, con lo que un `pip install` surte
efecto sin reempaquetar nada.

Un binario de PyInstaller no es viable aqui, y la razon es concreta:
`cuda_setup.py` localiza las DLL de cuBLAS y cuDNN recorriendo
`site.getsitepackages()`, que dentro de un binario congelado no existe. El STT
caeria con "cublas64_12.dll is not found". Ver el encabezado de `runtime.py`.

POR QUE LA CARPETA DE INICIO Y NO EL PROGRAMADOR DE TAREAS
----------------------------------------------------------
El Programador de tareas puede ejecutar en sesiones distintas o antes de que la
sesion este montada, y ahi el proceso NO tiene acceso al microfono ni al
dispositivo de audio predeterminado. La carpeta de Inicio arranca dentro de tu
sesion, que es exactamente donde vive el microfono. Ademas no pide permisos de
administrador.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asistente.runtime import APP_NAME, data_dir  # noqa: E402
from asistente.tray import write_ico  # noqa: E402

SHORTCUT_NAME = f"{APP_NAME}.lnk"

#: Se pasan por variables de entorno y no interpolados en el texto del script:
#: las rutas de Windows llevan espacios, comillas simples y `$`, y concatenarlas
#: dentro de PowerShell es una fuente inagotable de fallos raros.
_CREATE_SHORTCUT = """
$ErrorActionPreference = 'Stop'
$shell = New-Object -ComObject WScript.Shell
$lnk = $shell.CreateShortcut($env:APOLO_LNK)
$lnk.TargetPath = $env:APOLO_TARGET
$lnk.Arguments = $env:APOLO_ARGS
$lnk.WorkingDirectory = $env:APOLO_CWD
$lnk.Description = $env:APOLO_DESC
if ($env:APOLO_ICON) { $lnk.IconLocation = $env:APOLO_ICON }
$lnk.Save()
"""

_FOLDERS = (
    "[Environment]::GetFolderPath('Desktop'); [Environment]::GetFolderPath('Startup')"
)


def _powershell(script: str, env: dict[str, str] | None = None) -> str:
    """Ejecuta PowerShell y devuelve su salida. Lanza si falla."""
    import os

    completed = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True,
        text=True,
        env={**os.environ, **(env or {})},
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "PowerShell fallo sin decir por que")
    return completed.stdout.strip()


def special_folders() -> tuple[Path, Path]:
    """`(Escritorio, Inicio)` segun Windows, no segun una ruta adivinada.

    Se le preguntan a `GetFolderPath` en vez de componer `%APPDATA%\\...` porque
    las dos carpetas se pueden redirigir -OneDrive mueve el Escritorio, y las
    politicas de dominio mueven el Inicio- y una ruta inventada crearia el
    acceso directo en un sitio que no mira nadie.
    """
    lines = _powershell(_FOLDERS).splitlines()
    if len(lines) < 2:
        raise RuntimeError(f"no se pudieron leer las carpetas de Windows: {lines!r}")
    return Path(lines[0].strip()), Path(lines[1].strip())


def launcher() -> tuple[Path, str]:
    """`(interprete, argumentos)` del acceso directo.

    `pythonw.exe` vive junto a `python.exe` en el mismo venv, asi que se deriva
    del interprete que esta ejecutando esto: cualquier otra forma de buscarlo
    puede acabar apuntando al Python del sistema, que no tiene las dependencias.
    """
    pythonw = Path(sys.executable).with_name("pythonw.exe")
    if not pythonw.is_file():
        print(f"AVISO: no existe {pythonw}; se usara {sys.executable} (abrira una consola)")
        pythonw = Path(sys.executable)
    return pythonw, "-m asistente"


def check_importable(python: Path) -> str | None:
    """Comprueba que `-m asistente` va a funcionar. Devuelve el problema, o None.

    Este script mete `src/` en `sys.path` a mano, asi que aqui `import asistente`
    funciona SIEMPRE, tambien cuando el paquete no esta instalado. El acceso
    directo no tiene ese apano. Sin esta comprobacion, el instalador diria
    "listo" y el doble clic fallaria con un ImportError que nadie veria, porque
    no hay consola donde se imprima.
    """
    completed = subprocess.run(
        [str(python), "-c", "import asistente"],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )
    if completed.returncode == 0:
        return None
    return (
        f"{python} no puede importar el paquete `asistente`.\n"
        f"   Instalalo en el venv antes de crear los accesos directos:\n"
        f'       pip install -e ".[gpu,tray]"'
    )


def create_shortcut(path: Path, target: Path, args: str, icon: Path | None) -> None:
    _powershell(
        _CREATE_SHORTCUT,
        {
            "APOLO_LNK": str(path),
            "APOLO_TARGET": str(target),
            "APOLO_ARGS": args,
            "APOLO_CWD": str(ROOT),
            "APOLO_DESC": f"{APP_NAME} — asistente de voz local",
            "APOLO_ICON": str(icon) if icon else "",
        },
    )


def install(autostart: bool) -> int:
    desktop, startup = special_folders()
    python, args = launcher()

    # `python.exe` y no `pythonw.exe`: el segundo no tiene donde escribir el
    # error, y aqui queremos leerlo. Comparten entorno, asi que lo que importe
    # uno lo importa el otro.
    if problem := check_importable(Path(sys.executable)):
        print(f"ERROR: {problem}")
        return 1

    icon = write_ico(data_dir() / "apolo.ico")
    if icon is None:
        print("AVISO: sin Pillow no se puede generar el icono; se usara el de Python")

    targets = [desktop / SHORTCUT_NAME]
    if autostart:
        targets.append(startup / SHORTCUT_NAME)

    for path in targets:
        create_shortcut(path, python, args, icon)
        print(f"creado  {path}")

    print()
    print(f"Arranca con doble clic en el icono de {APP_NAME} del escritorio.")
    print("Tarda ~30 s en cargar los modelos; el icono de la bandeja aparece enseguida")
    print("y su globo pasa de 'arrancando' a 'escuchando' cuando esta listo.")
    if autostart:
        print("\nSe arrancara solo al iniciar sesion. Para quitarlo: --uninstall")
    return 0


def uninstall() -> int:
    desktop, startup = special_folders()
    removed = 0
    for path in (desktop / SHORTCUT_NAME, startup / SHORTCUT_NAME):
        if path.is_file():
            path.unlink()
            print(f"borrado {path}")
            removed += 1
    if not removed:
        print("no habia accesos directos que borrar")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="install_windows")
    parser.add_argument(
        "--no-autostart",
        action="store_true",
        help="no crear el acceso directo en la carpeta de Inicio",
    )
    parser.add_argument("--uninstall", action="store_true", help="quitar los accesos directos")
    args = parser.parse_args()

    if sys.platform != "win32":
        print(f"Esto solo tiene sentido en Windows (estas en {sys.platform}).")
        return 1

    try:
        return uninstall() if args.uninstall else install(not args.no_autostart)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
