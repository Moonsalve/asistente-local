"""Icono en la bandeja del sistema: la unica forma de ver y parar el asistente.

POR QUE ES OBLIGATORIO Y NO UN ADORNO
-------------------------------------
Corriendo de fondo no hay ventana, no hay Ctrl-C y no hay salida por pantalla.
Un proceso invisible que mantiene el microfono abierto y no se puede cerrar mas
que por el Administrador de tareas no es un asistente, es un problema. El icono
resuelve las tres preguntas que uno se hace de un proceso de fondo:

    esta vivo?      -> el icono esta o no esta
    que ha hecho?   -> "Ver registro"
    como lo paro?   -> "Salir"

DEGRADA, NO ROMPE
-----------------
`pystray` y `Pillow` son las unicas dependencias del proyecto que existen solo
para la interfaz. Si faltan, `start()` devuelve `False` y el asistente arranca
igual, sin icono: se para desde el Administrador de tareas y se avisa de ello
en el registro. Preferible a que no arranque por una dependencia de adorno.

SI NO APARECE EL ICONO
----------------------
Windows 11 esconde por defecto los iconos nuevos en el desplegable de la flecha
(^) de la barra de tareas. Antes de dar por hecho que fallo, mirar ahi: se saca
arrastrandolo a la barra, o desde Configuracion > Personalizacion > Barra de
tareas > Otros iconos de la bandeja.

Si tampoco esta ahi, `start()` lo dice: espera a que `pystray` CONFIRME que el
icono esta puesto antes de devolver `True`. Antes devolvia `True` en cuanto
arrancaba el hilo, asi que un fallo dentro de `pystray` -que ocurre en ese hilo,
no en este- dejaba al asistente creyendo que tenia icono y al usuario sin el.
La diferencia importa porque el resultado decide si se avisa al usuario o no.

LA SALIDA TIENE UN PLAZO
------------------------
Pulsar "Salir" pone un evento que el bucle de `pipeline.py` comprueba entre
frases. Como `record()` devuelve como mucho a los 2 s cuando nadie habla, la
salida es casi inmediata en la practica. Pero "casi siempre" no basta para el
unico boton que apaga esto: si el microfono se desconecta, la cola de audio deja
de entregar bloques y el bucle se queda esperando para siempre. Por eso hay un
plazo maximo tras el cual el proceso se mata a si mismo. Es aceptable porque no
hay nada que guardar: las metricas viven en memoria y el token de Spotify lo
persiste spotipy en cuanto lo refresca.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from asistente.runtime import APP_NAME

log = logging.getLogger(__name__)

#: Segundos que se le dan al bucle para salir por las buenas antes de matar el
#: proceso. Cuatro son el doble del peor caso normal (los 2 s de `record()`).
FORCED_EXIT_S = 4.0

#: Segundos que se le dan a `pystray` para confirmar que el icono esta puesto.
#: Generoso a proposito: compite con la carga de Whisper por la CPU. Lo que se
#: gana esperando es poder distinguir "no hay icono" de "todavia no".
STARTUP_TIMEOUT_S = 10.0

#: Verde medio: legible sobre una barra de tareas clara y sobre una oscura. Un
#: icono blanco desaparece en el tema claro de Windows y uno negro en el oscuro.
_COLOR = (46, 204, 113, 255)


def _draw(size: int) -> Any:
    """Dibuja un microfono. Todo en fracciones del tamano, para que el mismo
    codigo sirva para los 16 px de la bandeja y los 256 del acceso directo."""
    from PIL import Image, ImageDraw

    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    unit = size / 64.0
    width = max(2, int(4 * unit))

    draw.rounded_rectangle(
        [24 * unit, 8 * unit, 40 * unit, 38 * unit], radius=8 * unit, fill=_COLOR
    )
    draw.arc(
        [16 * unit, 24 * unit, 48 * unit, 50 * unit],
        start=0,
        end=180,
        fill=_COLOR,
        width=width,
    )
    draw.line([32 * unit, 50 * unit, 32 * unit, 56 * unit], fill=_COLOR, width=width)
    draw.line([23 * unit, 56 * unit, 41 * unit, 56 * unit], fill=_COLOR, width=width)
    return image


def write_ico(path: Path) -> Path | None:
    """Genera el .ico del acceso directo. `None` si Pillow no esta.

    Se genera en vez de versionar un binario para que el icono de la bandeja y
    el del acceso directo no puedan divergir: los dos salen de `_draw`.
    """
    try:
        image = _draw(256)
    except ImportError:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="ICO", sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (256, 256)])
    return path


def reveal(path: Path) -> None:
    """Abre un fichero o carpeta con la aplicacion que le toque."""
    try:
        if sys.platform == "win32":
            os.startfile(path)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.run(["/usr/bin/open", str(path)], check=False)
        else:
            subprocess.run(["xdg-open", str(path)], check=False)
    except Exception:
        log.exception("no se pudo abrir %s", path)


class Tray:
    """Icono de bandeja. `start()` no bloquea; el icono vive en su propio hilo."""

    def __init__(
        self,
        *,
        on_quit: Callable[[], None],
        log_file: Path | None = None,
        project_dir: Path | None = None,
    ) -> None:
        self._on_quit = on_quit
        self._log_file = log_file
        self._project_dir = project_dir
        self._icon: Any | None = None
        self._thread: threading.Thread | None = None
        #: Por que no hay icono, en una linea que se le pueda ensenar al
        #: usuario. `None` mientras no haya fallado nada.
        self.failure: str | None = None

    def start(self) -> bool:
        """Arranca el icono y ESPERA a que aparezca. `False` si no lo consigue.

        El motivo del fallo queda en `self.failure` para que quien llama pueda
        contarselo al usuario: sin consola, un aviso en el registro no lo lee
        nadie, y quedarse sin icono significa quedarse sin forma de parar el
        asistente que no sea el Administrador de tareas.
        """
        try:
            import pystray
        except ImportError:
            self.failure = (
                'falta pystray. Instalalo con:  pip install -e ".[tray]"'
            )
            log.warning(
                "sin icono de bandeja (falta pystray). El asistente funciona, pero para "
                'pararlo tendras que usar el Administrador de tareas. Instalalo con: '
                'pip install -e ".[tray]"'
            )
            return False

        try:
            image = _draw(64)
        except ImportError:
            self.failure = 'falta Pillow. Instalalo con:  pip install -e ".[tray]"'
            log.warning("sin icono de bandeja: falta Pillow")
            return False

        items = [pystray.MenuItem(f"{APP_NAME} en marcha", None, enabled=False)]
        if self._log_file is not None:
            items.append(
                pystray.MenuItem("Ver registro", self._open_log, default=True)
            )
        if self._project_dir is not None:
            items.append(pystray.MenuItem("Carpeta del proyecto", self._open_project))
        items.append(pystray.Menu.SEPARATOR)
        items.append(pystray.MenuItem("Salir", self._quit))

        icon = pystray.Icon(
            APP_NAME.lower(), image, f"{APP_NAME} — escuchando", menu=pystray.Menu(*items)
        )
        self._icon = icon

        # Demonio: si el bucle principal muere de forma inesperada, este hilo no
        # puede ser el que mantenga vivo un proceso sin asistente dentro.
        ready = threading.Event()
        self._thread = threading.Thread(
            target=self._serve, args=(icon, ready), name="tray", daemon=True
        )
        self._thread.start()

        if not ready.wait(STARTUP_TIMEOUT_S) and self.failure is None:
            self.failure = f"pystray no respondio en {STARTUP_TIMEOUT_S:.0f} s"
        if self.failure is not None:
            log.error("sin icono de bandeja: %s", self.failure)
            self._icon = None
            return False

        log.info("icono de bandeja activo (clic derecho para salir)")
        return True

    def _serve(self, icon: Any, ready: threading.Event) -> None:
        """Cuerpo del hilo del icono. Nunca deja morir el hilo en silencio.

        `icon.run()` no vuelve hasta que se para el icono, asi que todo lo que
        falle dentro -un backend que no encuentra su libreria, una bandeja que
        rechaza el icono- ocurre AQUI, en otro hilo. Sin este `except`, la
        excepcion se la comeria el `threading` por defecto y `start()` habria
        devuelto `True` igualmente.
        """

        def _ready(_icon: Any) -> None:
            # `visible = True` es el idioma de pystray para "ya lo puedes
            # mostrar"; el propio setter no hace nada si ya estaba visible.
            _icon.visible = True
            ready.set()

        try:
            icon.run(setup=_ready)
        except Exception as exc:
            self.failure = f"{type(exc).__name__}: {exc}"
            log.exception("el icono de bandeja fallo")
        finally:
            # Tambien en el fallo: si no, `start()` se comeria el plazo entero
            # para acabar diciendo lo mismo diez segundos mas tarde.
            ready.set()

    def set_title(self, text: str) -> None:
        """Texto del globo al pasar el raton. Es el unico indicador de estado
        que hay corriendo de fondo: distingue "cargando" de "listo", que son 30
        segundos de diferencia en los que si no, no pasa nada visible."""
        if self._icon is not None:
            try:
                self._icon.title = f"{APP_NAME} — {text}"
            except Exception:
                log.debug("no se pudo actualizar el titulo del icono", exc_info=True)

    def _open_log(self) -> None:
        if self._log_file is not None:
            reveal(self._log_file)

    def _open_project(self) -> None:
        if self._project_dir is not None:
            reveal(self._project_dir)

    def _quit(self) -> None:
        log.info("salida pedida desde la bandeja")
        self._on_quit()
        self.stop()
        _force_exit_after(FORCED_EXIT_S)

    def stop(self) -> None:
        if self._icon is not None:
            try:
                self._icon.stop()
            except Exception:
                log.debug("el icono ya estaba parado", exc_info=True)
            self._icon = None


def _force_exit_after(seconds: float) -> None:
    """Mata el proceso si en `seconds` no ha salido por las buenas.

    `os._exit` y no `sys.exit`: esto corre en el hilo del icono, y un `SystemExit`
    ahi solo terminaria ese hilo, dejando exactamente el proceso zombi que se
    quiere evitar.
    """

    def _kill() -> None:
        log.warning("el bucle no salio en %.0f s; se cierra el proceso a la fuerza", seconds)
        os._exit(0)

    timer = threading.Timer(seconds, _kill)
    timer.daemon = True
    timer.start()
