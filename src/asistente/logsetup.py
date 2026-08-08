"""Configuracion de logging resistente a la consola de Windows.

EL SINTOMA
----------
    23:16:42 INFO  asistente.tts.speaker: TTS calentado en 96 ms
    --- Logging error ---
    23:16:42 INFO  asistente: activacion por transcripcion: ...

Ese "--- Logging error ---" lo imprime el propio `logging` cuando un handler no
consigue emitir un registro. El mensaje se pierde y en su lugar queda un aviso
que no dice ni de donde venia ni por que.

La causa habitual en Windows es la codificacion: la consola usa cp1252/cp850 y
librerias como Piper registran simbolos IPA al fonemizar, que ahi no se pueden
codificar. Reconfigurar stdout/stderr a UTF-8 deberia bastar, pero no siempre:
un handler creado antes de la reconfiguracion, o una consola que no acepta el
cambio, siguen fallando.

LA SOLUCION
-----------
Un handler que, si no puede emitir, reintenta con una version del texto sin
caracteres problematicos en vez de rendirse. Y si aun asi falla, imprime UNA
linea diciendo que logger lo provoco, para poder arreglarlo de raiz en vez de
seguir adivinando.
"""

from __future__ import annotations

import logging
import sys

#: Loggers de terceros que registran cada peticion HTTP en INFO. En el arranque
#: son decenas de lineas que tapan lo que si importa.
NOISY_LOGGERS = ("httpx", "httpcore", "huggingface_hub", "urllib3", "filelock")


class RobustStreamHandler(logging.StreamHandler):  # type: ignore[type-arg]
    """StreamHandler que adapta el texto a lo que la consola sabe escribir.

    El saneado va ANTES de escribir, no como reintento tras fallar. Reintentar
    no funciona: cuando un `write` revienta con UnicodeEncodeError, el
    TextIOWrapper se queda con el codificador en mal estado y la siguiente
    escritura vuelve a fallar aunque el texto ya sea ASCII puro. Medido.
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
            encoding = getattr(self.stream, "encoding", None) or "utf-8"
            # Ida y vuelta por la codificacion real de la consola: lo que no
            # quepa se sustituye por '?' en vez de tumbar el registro entero.
            safe = message.encode(encoding, errors="replace").decode(encoding, errors="replace")
            self.stream.write(safe + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)

    def handleError(self, record: logging.LogRecord) -> None:
        """Una linea util en vez del '--- Logging error ---' generico."""
        if not logging.raiseExceptions:
            return
        exc = sys.exc_info()[1]
        try:
            sys.stderr.write(
                f"[logging] no se pudo emitir un registro de '{record.name}' "
                f"({type(exc).__name__}: {exc})\n"
            )
        except Exception:
            pass


def configure(verbose: bool = False) -> None:
    """Deja el logging listo. Llamar lo primero en `main()`."""
    # Reconfigurar antes de crear el handler: asi hereda ya el stream en UTF-8.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass

    handler = RobustStreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S"
        )
    )

    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    # force: reemplaza cualquier handler que una libreria haya instalado al
    # importarse, que es otra via por la que acababa emitiendo un handler sin
    # reconfigurar.
    for existing in root.handlers[:]:
        root.removeHandler(existing)
    root.addHandler(handler)

    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
