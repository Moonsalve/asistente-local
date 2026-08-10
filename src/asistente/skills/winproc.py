"""Procesos vivos de Windows: quien esta corriendo y como matarlo.

POR QUE HACE FALTA MIRAR LOS PROCESOS DE VERDAD
-----------------------------------------------
`app.close` mataba `<spec.process>.exe`, un nombre que sale de la config. Eso
falla de tres formas distintas y las tres en silencio:

  - Las apps de la Microsoft Store llegan del descubrimiento con `process=None`
    (no hay forma de deducirlo del AppID), asi que la skill se rendia antes de
    intentarlo. Y sin embargo la mayoria SI corren como un .exe normal y
    visible: Spotify de la Store es `Spotify.exe`, igual que el de escritorio.
  - Los .lnk del menu Inicio traen un nombre ADIVINADO
    (`name.replace(" ", "")`), asi que "Visual Studio Code" pedia matar
    `VisualStudioCode.exe`, que no existe: el proceso es `Code.exe`.
  - Cuando taskkill fallaba, su mensaje se tiraba y el usuario oia siempre lo
    mismo ("no estaba abierto"), tanto si la app no estaba abierta como si el
    nombre era incorrecto o el sistema denego el permiso.

Preguntarle al sistema que hay corriendo convierte una suposicion en un dato.
La allowlist sigue siendo la frontera de seguridad: aqui solo se enumera; quien
decide que se puede matar es `apps.py`, y solo con lo que ya estaba autorizado.

Se usa `tasklist` y no `psutil` para no anadir una dependencia nativa mas al
entorno de Windows, que ya arrastra CUDA, pycaw y onnxruntime.
"""

from __future__ import annotations

import csv
import io
import logging
import subprocess
import sys

log = logging.getLogger(__name__)

#: Margen amplio: `tasklist` en un equipo cargado tarda cientos de ms, pero un
#: cuelgue no puede dejar el asistente esperando en mitad de un comando.
_TIMEOUT_S = 8.0


def _decode(raw: bytes) -> str:
    """La consola de Windows no escribe en UTF-8 ni en una codificacion fija.

    Los nombres de imagen son ASCII, asi que para lo que aqui importa da igual;
    lo que no puede pasar es que un byte raro en un mensaje de error levante un
    `UnicodeDecodeError` y tumbe el comando. latin-1 nunca falla y cierra la
    cadena de intentos.
    """
    for encoding in ("utf-8", "cp1252"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="replace")


def _run(command: list[str]) -> tuple[int, str, str]:
    """Ejecuta y devuelve (codigo, stdout, stderr) ya decodificados.

    Sin `text=True` a proposito: ese modo decodifica con la codificacion local
    en modo estricto, y una 'ó' de un mensaje de error en cp850 podia reventar.
    """
    try:
        result = subprocess.run(  # noqa: S603
            command,
            capture_output=True,
            check=False,
            timeout=_TIMEOUT_S,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        log.debug("fallo %s: %s", command[0], exc)
        return -1, "", str(exc)
    return result.returncode, _decode(result.stdout), _decode(result.stderr)


def running_images() -> tuple[str, ...]:
    """Nombres de imagen de los procesos vivos (`('Spotify.exe', ...)`).

    Tupla vacia si no se puede saber —fuera de Windows, o si `tasklist` falla—.
    Quien llama tiene que distinguir "no hay nada corriendo" de "no se pudo
    preguntar", y por eso existe tambien `can_list_processes()`.
    """
    if sys.platform != "win32":
        return ()

    # /NH quita la cabecera y /FO CSV entrecomilla los campos, que es lo unico
    # que hace parseable una salida con nombres que llevan espacios.
    code, out, err = _run(["tasklist", "/FO", "CSV", "/NH"])
    if code != 0:
        log.debug("tasklist devolvio %s: %s", code, err.strip())
        return ()

    imagenes: list[str] = []
    for row in csv.reader(io.StringIO(out)):
        if row and (nombre := row[0].strip()):
            imagenes.append(nombre)
    return tuple(imagenes)


def can_list_processes() -> bool:
    return sys.platform == "win32"


def normalize_image(name: str) -> str:
    """Clave de comparacion: sin extension, sin mayusculas, sin espacios sobrantes."""
    stem = name.strip()
    if stem.lower().endswith(".exe"):
        stem = stem[: -len(".exe")]
    return stem.lower()


def kill_image(image: str) -> tuple[bool, str]:
    """Mata todos los procesos con ese nombre de imagen. `(ok, motivo)`.

    `/F` fuerza —sin el, una app con un dialogo abierto ignora la peticion y el
    comando parece no hacer nada— y `/T` se lleva los hijos, que es lo que hace
    falta con los navegadores y con Spotify, que abren un proceso por pestana o
    por render.
    """
    if sys.platform != "win32":
        return False, "cerrar aplicaciones solo funciona en Windows"

    nombre = image if image.lower().endswith(".exe") else f"{image}.exe"
    code, out, err = _run(["taskkill", "/F", "/T", "/IM", nombre])
    if code == 0:
        return True, ""

    motivo = (err.strip() or out.strip() or f"taskkill devolvio {code}").splitlines()[0]
    log.debug("taskkill %s: %s", nombre, motivo)
    return False, motivo
