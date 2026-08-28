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

import contextlib
import io
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from asistente.runtime import has_console, log_dir

#: Loggers de terceros que registran cada peticion HTTP en INFO. En el arranque
#: son decenas de lineas que tapan lo que si importa.
NOISY_LOGGERS = ("httpx", "httpcore", "huggingface_hub", "urllib3", "filelock")

#: Rotacion del registro en disco. Un turno son un par de lineas, asi que 2 MB
#: son semanas de uso; lo que se quiere evitar es el fichero que crece sin
#: limite en una maquina que nadie mira.
_MAX_BYTES = 2 * 1024 * 1024
_BACKUPS = 3


class RobustStreamHandler(logging.StreamHandler):  # type: ignore[type-arg]
    """StreamHandler que adapta el texto a lo que la consola sabe escribir.

    El saneado va ANTES de escribir, no como reintento tras fallar. Reintentar
    no funciona: cuando un `write` revienta con UnicodeEncodeError, el
    TextIOWrapper se queda con el codificador en mal estado y la siguiente
    escritura vuelve a fallar aunque el texto ya sea ASCII puro. Medido.

    EL `flush` VA APARTE DEL `write`, y no es un detalle. Medido en Windows: la
    linea sale en pantalla y es el flush posterior el que revienta con
    `OSError: [WinError 1] Incorrect function`, segun que consola. Metiendo los
    dos en el mismo `try`, cada linea del arranque salia seguida de un
    "[logging] no se pudo emitir un registro de ...". O sea que el mensaje de
    error decia justo lo contrario de lo que habia pasado -el registro SI se
    emitio- y duplicaba la salida entera. Un diagnostico falso es peor que
    ninguno: manda a buscar el fallo donde no esta.
    """

    def __init__(self, stream: object | None = None) -> None:
        super().__init__(stream)  # type: ignore[arg-type]
        self._flush_failed = False

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
            encoding = getattr(self.stream, "encoding", None) or "utf-8"
            # Ida y vuelta por la codificacion real de la consola: lo que no
            # quepa se sustituye por '?' en vez de tumbar el registro entero.
            safe = message.encode(encoding, errors="replace").decode(encoding, errors="replace")
            self.stream.write(safe + self.terminator)
        except Exception:
            self.handleError(record)
            return
        self._flush_quietly()

    def _flush_quietly(self) -> None:
        """Vacia el buffer sin montar un escandalo si no se puede.

        Se avisa UNA vez y no en cada linea: es una propiedad de la consola,
        no de cada registro, asi que repetirlo doscientas veces solo tapa lo
        que se quiere leer. El texto ya esta escrito; como mucho se quedara en
        el buffer hasta la siguiente linea.
        """
        try:
            self.flush()
        except Exception as exc:
            if self._flush_failed:
                return
            self._flush_failed = True
            with contextlib.suppress(Exception):
                self.stream.write(
                    f"[logging] esta consola no admite flush ({type(exc).__name__}: {exc}); "
                    f"la salida puede ir a tirones. No se pierde nada: el registro "
                    f"completo esta en el fichero.{self.terminator}"
                )

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


class _StreamToLog(io.TextIOBase):
    """Objeto tipo fichero que reenvia lo que le escriban al registro.

    Sin consola, `sys.stdout` y `sys.stderr` valen `None`. `print()` lo tolera
    -CPython comprueba y no hace nada- pero cualquier libreria que llame a
    `sys.stderr.write(...)` por su cuenta revienta con AttributeError, y lo hace
    dentro de codigo de terceros donde no hay un try que lo recoja.

    Sustituirlos por esto convierte esas escrituras en lineas de registro en vez
    de en una excepcion. Se acumula hasta el salto de linea porque muchas
    librerias escriben la linea en varios trozos y un registro por trozo seria
    ilegible.
    """

    def __init__(self, logger: logging.Logger, level: int) -> None:
        super().__init__()
        self._logger = logger
        self._level = level
        self._buffer = ""

    def write(self, text: str) -> int:
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line.strip():
                self._logger.log(self._level, "%s", line.rstrip())
        return len(text)

    def flush(self) -> None:
        if self._buffer.strip():
            self._logger.log(self._level, "%s", self._buffer.rstrip())
        self._buffer = ""

    def isatty(self) -> bool:
        return False

    def writable(self) -> bool:
        return True


class _Discard(io.TextIOBase):
    """Sustituto de `sys.stderr` cuando no hay ni consola ni registro.

    Acepta la misma firma que `_StreamToLog` para poder intercambiarlos sin un
    `if` en cada uso. Tira lo que le escriban, que es lo unico honesto que se
    puede hacer cuando no queda ningun sitio donde escribir: lo que evita es que
    una libreria de terceros tumbe el asistente con un AttributeError.
    """

    def __init__(self, logger: logging.Logger, level: int) -> None:
        super().__init__()

    def write(self, text: str) -> int:
        return len(text)

    def isatty(self) -> bool:
        return False

    def writable(self) -> bool:
        return True


def configure(
    verbose: bool = False,
    *,
    to_file: bool = True,
    log_file: Path | None = None,
) -> Path | None:
    """Deja el logging listo. Llamar lo primero en `main()`.

    Devuelve la ruta del registro en disco, o `None` si no se pudo abrir.

    EL REGISTRO EN DISCO NO ES OPCIONAL CUANDO NO HAY CONSOLA, y por eso se
    escribe SIEMPRE, tambien con terminal: asi la opcion "ver registro" del
    icono de la bandeja encuentra siempre algo, y el fichero de un arranque de
    fondo que fallo se puede comparar con el de uno que funciono.
    """
    console = has_console()

    # Reconfigurar antes de crear el handler: asi hereda ya el stream en UTF-8.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
            except (OSError, ValueError):
                pass

    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    # Reemplaza cualquier handler que una libreria haya instalado al importarse,
    # que es otra via por la que acababa emitiendo un handler sin reconfigurar.
    for existing in root.handlers[:]:
        root.removeHandler(existing)

    if console:
        handler = RobustStreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S"
            )
        )
        root.addHandler(handler)

    path = _add_file_handler(root, log_file) if to_file else None

    if not console:
        # Ahora que hay donde escribir, se tapan los `None`. Va DESPUES de
        # instalar el handler de fichero: al reves, las primeras escrituras
        # caerian en un logger sin handlers.
        #
        # Y SOLO SI HAY HANDLER, que es la parte que no es obvia. Sin handlers,
        # `logging` recurre a `lastResort`, que escribe en `sys.stderr`... que
        # seria este mismo objeto, que vuelve a registrar, que vuelve a
        # lastResort. Recursion infinita, y ademas en el unico caso en que se
        # da: el registro en disco no se pudo abrir, o sea cuando algo ya iba
        # mal. Un cuelgue girando la CPU es mucho peor que perder unas lineas.
        sink = _StreamToLog if root.handlers else _Discard
        sys.stdout = sink(logging.getLogger("stdout"), logging.INFO)  # type: ignore[assignment]
        sys.stderr = sink(logging.getLogger("stderr"), logging.ERROR)  # type: ignore[assignment]

    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    return path


def _add_file_handler(root: logging.Logger, log_file: Path | None) -> Path | None:
    """Instala el registro rotativo. Devuelve la ruta, o `None` si no pudo.

    Un fallo aqui NO puede impedir el arranque: si el disco esta lleno o el
    directorio no se puede crear, el asistente tiene que seguir funcionando
    aunque sea a ciegas. Pero se avisa por la consola si la hay.
    """
    try:
        path = log_file if log_file is not None else log_dir() / "apolo.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            path,
            maxBytes=_MAX_BYTES,
            backupCount=_BACKUPS,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as exc:
        if sys.stderr is not None:
            print(f"[logging] no se pudo abrir el registro en disco: {exc}", file=sys.stderr)
        return None

    # Con fecha, al contrario que en consola: este fichero sobrevive a la
    # sesion y "23:16:42" sin dia no sirve para nada al mirarlo una semana
    # despues.
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
    )
    root.addHandler(handler)
    return path
