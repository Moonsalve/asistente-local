"""Tests del arranque de fondo, sin terminal.

Lo que se prueba aqui son exactamente los tres supuestos que `pythonw.exe`
invalida (ver `runtime.py`), porque los tres fallan EN SILENCIO y por tanto
ninguno se nota probando a mano:

  - sin `stdout`, el registro se perdia entero y el asistente parecia ir bien
  - sin el cwd correcto, no encuentra `config.yaml` ni la voz de Piper
  - sin Ctrl-C, el bucle no tenia forma de terminar

Ninguno de estos tests necesita Windows: el guardia de instancia unica tiene
implementacion POSIX con `flock` precisamente para poder probarse aqui.
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

from asistente.audio.recorder import Utterance
from asistente.config import FollowUpConfig
from asistente.logsetup import configure
from asistente.pipeline import Assistant
from asistente.runtime import anchor_working_directory, app_root, data_dir, has_console
from asistente.singleton import SingleInstance
from asistente.skills.registry import SkillRegistry
from asistente.tray import Tray

# --------------------------------------------------------------------------
# runtime: donde estoy y donde escribo
# --------------------------------------------------------------------------


def test_the_project_root_is_the_one_that_holds_the_config() -> None:
    """`app_root` busca marcadores, no cuenta directorios.

    Si alguien mueve `runtime.py` de sitio, contar `parents[2]` daria una ruta
    equivocada sin fallar, y el asistente arrancaria con la config de otro
    sitio o sin ella.
    """
    root = app_root()
    assert (root / "config.yaml").is_file()
    assert (root / "commands.yaml").is_file()


def test_the_data_directory_is_outside_the_repository() -> None:
    """Los registros rotan; un checkout de git no es sitio para eso."""
    assert app_root() not in data_dir().parents


def test_anchoring_fixes_the_relative_paths_from_any_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """El caso real: un acceso directo lanzado con cwd en otra parte.

    Antes de anclar, `config.yaml` no existe desde ahi. Despues, si. Es la
    prueba de que UNA llamada arregla todas las rutas relativas del proyecto.
    """
    monkeypatch.chdir(tmp_path)
    assert not Path("config.yaml").is_file()

    root = anchor_working_directory()

    assert Path("config.yaml").is_file()
    assert Path.cwd() == root


def test_there_is_no_console_when_stderr_is_gone(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bajo `pythonw`, `sys.stderr` vale literalmente None."""
    monkeypatch.setattr(sys, "stderr", None)
    assert has_console() is False


# --------------------------------------------------------------------------
# singleton: dos asistentes a la vez son un desastre silencioso
# --------------------------------------------------------------------------


@pytest.fixture
def _lock_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Aisla el fichero de bloqueo, para no chocar con un Apolo de verdad."""
    monkeypatch.setattr("asistente.singleton.data_dir", lambda: tmp_path)
    return tmp_path


@pytest.mark.skipif(sys.platform == "win32", reason="en Windows el guardia es un mutex")
def test_a_second_instance_is_refused(_lock_dir: Path) -> None:
    """Lo que evita: dos Whisper en 8 GB de VRAM y cada orden ejecutada dos veces."""
    first = SingleInstance("prueba")
    second = SingleInstance("prueba")
    try:
        assert first.acquire() is True
        assert second.acquire() is False
    finally:
        first.release()
        second.release()


@pytest.mark.skipif(sys.platform == "win32", reason="en Windows el guardia es un mutex")
def test_releasing_lets_the_next_one_in(_lock_dir: Path) -> None:
    """Un bloqueo que no se suelta es peor que no tener bloqueo: deja el
    asistente sin poder arrancar nunca mas."""
    first = SingleInstance("prueba")
    assert first.acquire() is True
    first.release()

    second = SingleInstance("prueba")
    try:
        assert second.acquire() is True
    finally:
        second.release()


# --------------------------------------------------------------------------
# logsetup: sin consola, el fichero es el unico testigo
# --------------------------------------------------------------------------


@pytest.fixture
def _restore_logging() -> object:
    """`configure()` toca estado global (handlers, sys.stdout/stderr).

    Sin esto, el primer test que simula "sin consola" dejaria `sys.stderr`
    apuntando a un logger y pytest perderia su propia salida en los tests
    siguientes.
    """
    root = logging.getLogger()
    saved = (root.handlers[:], root.level, sys.stdout, sys.stderr)
    yield
    root.handlers[:] = saved[0]
    root.setLevel(saved[1])
    sys.stdout, sys.stderr = saved[2], saved[3]


def test_everything_reaches_the_file_when_there_is_no_console(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _restore_logging: object
) -> None:
    monkeypatch.setattr("asistente.logsetup.has_console", lambda: False)
    path = tmp_path / "apolo.log"

    assert configure(log_file=path) == path
    logging.getLogger("asistente").info("cargando whisper")

    assert "cargando whisper" in path.read_text(encoding="utf-8")


def test_a_library_writing_to_a_dead_stderr_does_not_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _restore_logging: object
) -> None:
    """La regresion concreta: bajo `pythonw`, `sys.stderr` es None y cualquier
    `sys.stderr.write(...)` de una libreria de terceros revienta con
    AttributeError dentro de codigo que no podemos envolver en un try."""
    monkeypatch.setattr("asistente.logsetup.has_console", lambda: False)
    monkeypatch.setattr(sys, "stderr", None)
    path = tmp_path / "apolo.log"
    configure(log_file=path)

    sys.stderr.write("piper dice algo raro\n")  # type: ignore[union-attr]

    assert "piper dice algo raro" in path.read_text(encoding="utf-8")


def test_a_log_file_that_cannot_be_opened_does_not_stop_the_assistant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _restore_logging: object
) -> None:
    """Disco lleno o carpeta sin permisos: se arranca a ciegas, pero se arranca.

    Aqui se fuerza el fallo poniendo un FICHERO donde tendria que ir la carpeta
    del registro, que es un error de E/S real y no un mock.

    Y ademas escribir en stderr tiene que seguir siendo inofensivo. Sin consola
    y sin fichero, el root se queda SIN handlers: si el redirector siguiera
    activo, `logging` caeria en `lastResort`, que escribe en `sys.stderr`, que
    es el propio redirector, que vuelve a registrar. Recursion infinita girando
    la CPU, en el unico caso en que se da.
    """
    monkeypatch.setattr("asistente.logsetup.has_console", lambda: False)
    blocker = tmp_path / "no-soy-carpeta"
    blocker.write_text("", encoding="utf-8")

    assert configure(log_file=blocker / "apolo.log") is None

    sys.stderr.write("y esto no puede colgarse\n")


# --------------------------------------------------------------------------
# consolas que escriben pero no dejan hacer flush
# --------------------------------------------------------------------------


class _ConsolaSinFlush(io.StringIO):
    """Consola de Windows medida: `write` va bien, `flush` da WinError 1."""

    def __init__(self) -> None:
        super().__init__()
        self.flushes = 0

    def flush(self) -> None:
        self.flushes += 1
        raise OSError(1, "Incorrect function")


def test_a_console_that_cannot_flush_still_logs() -> None:
    """El fallo que duplicaba la salida entera del arranque.

    `write` y `flush` estaban en el mismo `try`, asi que cada linea salia en
    pantalla y acto seguido salia "[logging] no se pudo emitir un registro de
    ...". El mensaje decia lo contrario de lo que habia pasado -el registro SI
    se emitio- y mandaba a buscar el fallo donde no estaba.
    """
    from asistente.logsetup import RobustStreamHandler

    consola = _ConsolaSinFlush()
    handler = RobustStreamHandler(consola)
    handler.setFormatter(logging.Formatter("%(message)s"))

    for numero in range(3):
        handler.emit(
            logging.LogRecord("prueba", logging.INFO, __file__, 1, f"linea {numero}", None, None)
        )

    salida = consola.getvalue()
    assert "linea 0" in salida
    assert "linea 2" in salida
    assert "no se pudo emitir" not in salida
    # Se avisa UNA vez de que la consola no admite flush, no en cada linea.
    assert salida.count("no admite flush") == 1
    assert consola.flushes == 3


def test_a_console_that_cannot_write_is_still_reported() -> None:
    """Lo contrario del test anterior: un `write` que falla SI es un registro
    perdido, y ahi el aviso tiene que seguir saliendo."""
    from asistente.logsetup import RobustStreamHandler

    class _Rota(io.StringIO):
        def write(self, text: str) -> int:
            raise OSError(1, "Incorrect function")

    handler = RobustStreamHandler(_Rota())
    handler.setFormatter(logging.Formatter("%(message)s"))
    fallos = []
    handler.handleError = lambda record: fallos.append(record)  # type: ignore[method-assign]

    handler.emit(logging.LogRecord("prueba", logging.INFO, __file__, 1, "hola", None, None))

    assert len(fallos) == 1


# --------------------------------------------------------------------------
# pipeline: la salida sin Ctrl-C
# --------------------------------------------------------------------------


class _Mic:
    sample_rate = 16_000

    def blocks(self) -> object:
        # Muy por encima de lo que los tests deberian consumir: si el bucle no
        # mira el evento de parada, el test falla por el contador, no por
        # quedarse colgado para siempre.
        return iter(range(1_000))

    def preroll_audio(self) -> np.ndarray:
        return np.zeros(0, dtype=np.float32)


class _CountingRecorder:
    """Devuelve siempre una frase vacia, que el bucle descarta y sigue."""

    def __init__(self, stop: object, after: int) -> None:
        self._stop = stop
        self._after = after
        self.calls = 0

    def record(self, blocks: object, preroll: np.ndarray) -> Utterance:
        self.calls += 1
        if self.calls >= self._after:
            self._stop.set()  # type: ignore[attr-defined]
        return Utterance(total_s=0.5)


class _CountingWakeWord:
    def __init__(self, stop: object, after: int) -> None:
        self._stop = stop
        self._after = after
        self.calls = 0

    def detected(self, block: object) -> bool:
        self.calls += 1
        if self.calls >= self._after:
            self._stop.set()  # type: ignore[attr-defined]
        return False

    def reset(self) -> None:
        pass


def _assistant(**kwargs: object) -> Assistant:
    """Ensambla un Assistant con lo minimo: aqui solo se prueba el bucle.

    Sin ventana de seguimiento: lo que se prueba aqui es que el bucle se para
    cuando se le pide. La ventana tiene sus propios tests en
    `test_follow_up.py`, y dejarla activa meteria su maquina de estados dentro
    de un test que no habla de ella.
    """
    kwargs.setdefault("follow_up", FollowUpConfig(enabled=False))
    return Assistant(
        _Mic(),  # type: ignore[arg-type]
        kwargs.pop("recorder", None),  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        SkillRegistry([]),
        None,  # type: ignore[arg-type]
        **kwargs,  # type: ignore[arg-type]
    )


def test_the_transcript_loop_stops_when_asked() -> None:
    """Modo por defecto del asistente. La parada se mira ENTRE frases, asi que
    el bucle da exactamente las vueltas pedidas y ni una mas."""
    import threading

    stop = threading.Event()
    recorder = _CountingRecorder(stop, after=3)
    assistant = _assistant(recorder=recorder, keyphrase=object(), stop=stop)

    assistant.run_forever()

    assert recorder.calls == 3


def test_the_wakeword_loop_stops_when_asked() -> None:
    import threading

    stop = threading.Event()
    wake_word = _CountingWakeWord(stop, after=5)
    assistant = _assistant(recorder=None, wake_word=wake_word, stop=stop)

    assistant.run_forever()

    assert wake_word.calls == 5


def test_without_a_stop_event_nothing_changes() -> None:
    """El modo terminal de siempre: sin evento, `_stopping` es False y el bucle
    es el `while True` de antes. Lo garantiza no haber tocado ese camino."""
    assistant = _assistant(recorder=None, keyphrase=object())
    assert assistant._stopping is False


# --------------------------------------------------------------------------
# arranque completo: nada puede desaparecer sin dejar rastro
# --------------------------------------------------------------------------


def test_an_unexpected_crash_leaves_a_trace_in_the_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _restore_logging: object
) -> None:
    """El modo de fallo que motiva todo el envoltorio de `main()`.

    De fondo, una excepcion sin capturar hace que el proceso se evapore: haces
    doble clic, el icono no llega a aparecer y no queda ni error ni pista. Este
    test recorre el arranque de verdad -anclar, configurar el registro, avisar-
    con el cwd en otro sitio, que es la situacion del acceso directo.
    """
    from asistente import __main__ as entry

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["asistente"])
    monkeypatch.setattr("asistente.logsetup.log_dir", lambda: tmp_path)
    # Los dos modulos preguntan por su cuenta si hay consola: `logsetup` para
    # decidir si escribe a fichero, `__main__` para decidir si avisa al usuario.
    monkeypatch.setattr("asistente.logsetup.has_console", lambda: False)
    monkeypatch.setattr(entry, "has_console", lambda: False)

    def _boom(*args: object, **kwargs: object) -> int:
        raise RuntimeError("se cayo Ollama")

    monkeypatch.setattr(entry, "_run", _boom)
    avisos: list[tuple[str, str]] = []
    monkeypatch.setattr(entry, "notify", lambda title, msg: avisos.append((title, msg)))

    assert entry.main() == 1

    registro = (tmp_path / "apolo.log").read_text(encoding="utf-8")
    assert "se cayo Ollama" in registro
    assert "RuntimeError" in registro
    # Y el usuario se entera por un canal que va a ver, no solo por el fichero.
    assert avisos and "apolo.log" in avisos[0][1]


# --------------------------------------------------------------------------
# tray: es opcional, y tiene que serlo de verdad
# --------------------------------------------------------------------------


def test_the_assistant_starts_without_pystray(monkeypatch: pytest.MonkeyPatch) -> None:
    """Poner None en sys.modules hace que `import pystray` lance ImportError,
    que es exactamente lo que pasa en una maquina sin el paquete."""
    monkeypatch.setitem(sys.modules, "pystray", None)

    tray = Tray(on_quit=lambda: None)

    assert tray.start() is False


def test_quitting_from_the_tray_sets_the_stop_event(monkeypatch: pytest.MonkeyPatch) -> None:
    """El cableado que apaga el asistente: el menu no para nada por su cuenta,
    solo pone el evento que mira el bucle.

    El plazo de salida forzosa se desactiva para el test. No es cosmetico: deja
    armado un temporizador que llama a `os._exit(0)` a los 4 segundos, y eso
    mata a pytest a media ejecucion sin dar error, porque el codigo de salida
    es 0.
    """
    import threading

    monkeypatch.setattr("asistente.tray._force_exit_after", lambda _: None)
    stop = threading.Event()
    tray = Tray(on_quit=stop.set)

    tray._quit()

    assert stop.is_set()


# --------------------------------------------------------------------------
# tray: "start() dijo que si" tiene que significar que hay icono
# --------------------------------------------------------------------------
#
# Es el fallo que motivo estos tests. `start()` devolvia True en cuanto
# arrancaba el hilo, asi que cualquier problema DENTRO de pystray -que ocurre
# en ese hilo, no en el que llama- dejaba al asistente creyendo que tenia icono
# y al usuario sin el, sin una linea en el log que lo dijera.
#
# Se falsea `pystray` entero y se sustituye el dibujo: lo que se prueba es el
# ciclo de vida del icono, no si Pillow sabe pintar un microfono.


class _FakePystray:
    """Modulo `pystray` de mentira, con lo justo que usa `tray.py`."""

    class Menu:
        SEPARATOR = object()

        def __init__(self, *items: object) -> None:
            self.items = items

    class MenuItem:
        def __init__(self, text: str, action: object = None, **kwargs: object) -> None:
            self.text = text

    def __init__(self, comportamiento: str) -> None:
        self._comportamiento = comportamiento
        self.Icon = self._make_icon()

    def _make_icon(self) -> type:
        comportamiento = self._comportamiento

        class Icon:
            def __init__(self, name: str, image: object, title: str, menu: object) -> None:
                self.visible = False
                self.title = title
                self.parado = False

            def run(self, setup: object = None) -> None:
                if comportamiento == "revienta":
                    raise RuntimeError("no hay bandeja en este escritorio")
                if comportamiento == "cuelga":
                    threading.Event().wait(5)  # nunca confirma
                    return
                setup(self)  # type: ignore[operator]
                # `run()` no vuelve hasta que se para el icono, igual que el de
                # verdad: si volviera, el test no distinguiria un icono puesto
                # de uno que aparecio y desaparecio.
                threading.Event().wait(5)

            def stop(self) -> None:
                self.parado = True

        return Icon


def _con_pystray(
    monkeypatch: pytest.MonkeyPatch, comportamiento: str
) -> None:
    monkeypatch.setitem(sys.modules, "pystray", _FakePystray(comportamiento))
    monkeypatch.setattr("asistente.tray._draw", lambda size: object())


def test_start_waits_until_pystray_confirms_the_icon(monkeypatch: pytest.MonkeyPatch) -> None:
    _con_pystray(monkeypatch, "bien")

    tray = Tray(on_quit=lambda: None)

    assert tray.start() is True
    assert tray.failure is None


def test_start_fails_when_the_backend_crashes(monkeypatch: pytest.MonkeyPatch) -> None:
    """El caso concreto que devolvia True. La excepcion ocurre en el hilo del
    icono, donde el `threading` por defecto se la come sin que nadie se
    entere."""
    _con_pystray(monkeypatch, "revienta")

    tray = Tray(on_quit=lambda: None)

    assert tray.start() is False
    assert tray.failure is not None
    assert "no hay bandeja" in tray.failure


def test_start_fails_when_pystray_never_confirms(monkeypatch: pytest.MonkeyPatch) -> None:
    """Un backend que ni falla ni arranca. Sin plazo, `start()` se quedaria
    esperando para siempre y el asistente no llegaria a cargar nada."""
    _con_pystray(monkeypatch, "cuelga")
    monkeypatch.setattr("asistente.tray.STARTUP_TIMEOUT_S", 0.05)

    tray = Tray(on_quit=lambda: None)

    assert tray.start() is False
    assert tray.failure is not None
    assert "no respondio" in tray.failure


def test_the_tray_is_on_by_default_even_with_a_console() -> None:
    """Lo que hacia que "no hay icono" fuera indistinguible de "aqui no toca".

    Antes solo se ponia bajo `pythonw`, asi que probar el arranque desde la
    terminal nunca mostraba icono, y eso no se podia distinguir de un icono
    roto.
    """
    from asistente import __main__ as entry

    assert entry._wants_tray(argparse.Namespace(tray=None, text=False)) is True
    # En modo texto no: no hay nada de fondo que parar.
    assert entry._wants_tray(argparse.Namespace(tray=None, text=True)) is False
    # Y lo que pidas explicitamente manda en los dos sentidos.
    assert entry._wants_tray(argparse.Namespace(tray=False, text=False)) is False
    assert entry._wants_tray(argparse.Namespace(tray=True, text=True)) is True


def test_a_missing_icon_is_reported_to_the_user(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sin consola, un aviso en el registro no lo lee nadie, y quedarse sin
    icono es quedarse sin forma de parar el asistente. Tiene que salir por un
    canal visible aunque el arranque continue."""
    from asistente import __main__ as entry

    monkeypatch.setattr(entry, "has_console", lambda: False)
    avisos: list[tuple[str, str]] = []
    monkeypatch.setattr(
        entry, "notify", lambda title, msg, **kwargs: avisos.append((title, msg))
    )

    entry._tray_unavailable("falta pystray", None)

    assert avisos
    assert "falta pystray" in avisos[0][1]
    assert "Administrador de tareas" in avisos[0][1]


def test_with_a_console_the_missing_icon_only_goes_to_the_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Un cuadro de dialogo delante de quien ya esta mirando una terminal
    sobra: ahi el aviso del log se ve."""
    from asistente import __main__ as entry

    monkeypatch.setattr(entry, "has_console", lambda: True)
    avisos: list[object] = []
    monkeypatch.setattr(entry, "notify", lambda *a, **k: avisos.append(a))

    entry._tray_unavailable("falta pystray", None)

    assert avisos == []
