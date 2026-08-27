"""Tests de la ventana de seguimiento.

LO QUE SE ESTA PROBANDO NO ES LA COMODIDAD, ES EL FILTRO. Encadenar ordenes es
la parte facil; lo dificil es que quitar la palabra clave no convierta al
asistente en algo que obedece a la television. Por eso la mayoria de estos
tests comprueban lo que la ventana IGNORA, no lo que ejecuta.

Todo con dobles: aqui se prueba la maquina de estados del bucle, no Whisper ni
el router. Un test que dependiera de los dos fallaria por motivos que no tienen
nada que ver con lo que se quiere garantizar.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

from asistente.audio.recorder import Utterance
from asistente.config import FollowUpConfig, VadConfig
from asistente.pipeline import Assistant
from asistente.router.schema import RouteResult, SpeechReply, Stage, ToolCall

SAMPLE_RATE = 16_000

#: Lo que devuelve el recorder cuando alguien habla: suficiente voz para pasar
#: las puertas de ruido, que aqui no son lo que se prueba.
VOZ = Utterance(audio=np.full(SAMPLE_RATE, 0.2, dtype=np.float32), speech_s=1.0, total_s=1.5)
SILENCIO = Utterance(total_s=0.5)


# --------------------------------------------------------------------------
# dobles
# --------------------------------------------------------------------------


class _Mic:
    sample_rate = SAMPLE_RATE

    def __init__(self) -> None:
        self.descartes = 0

    def blocks(self) -> object:
        return iter(range(1_000))

    def preroll_audio(self) -> np.ndarray:
        return np.zeros(0, dtype=np.float32)

    def discard_pending(self) -> float:
        self.descartes += 1
        return 0.0


class _ScriptedRecorder:
    """Entrega frases guionizadas y para el bucle cuando se le acaban.

    Parar con el evento de parada es lo mismo que hacen los tests del bucle
    principal: sin eso la ventana seguiria girando hasta agotar su plazo real,
    y un test que tarda cinco segundos no se ejecuta.
    """

    def __init__(self, guion: list[Utterance], stop: threading.Event) -> None:
        self._guion = list(guion)
        self._stop = stop
        self.esperas: list[float] = []

    def record(self, blocks: object, preroll: np.ndarray, *, wait_s: float = 2.0) -> Utterance:
        self.esperas.append(wait_s)
        if not self._guion:
            self._stop.set()
            return SILENCIO
        return self._guion.pop(0)


class _Transcriber:
    def __init__(self, textos: list[str]) -> None:
        self._textos = list(textos)

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        return self._textos.pop(0) if self._textos else ""


class _Router:
    """Mapea texto -> resultado. Lo que no este en el mapa no lo entiende."""

    def __init__(self, mapa: dict[str, RouteResult]) -> None:
        self._mapa = mapa
        self.consultas: list[str] = []

    def route(self, text: str) -> RouteResult:
        self.consultas.append(text)
        return self._mapa.get(text, RouteResult(stage=Stage.LLM))


class _Resultado:
    ok = True
    speech = "vale"


class _Registry:
    names = frozenset(
        {
            "media.next",
            "media.previous",
            "media.play_pause",
            "volume.step",
            "volume.set",
            "volume.mute",
            "volume.query",
            "open.target",
        }
    )

    def __init__(self) -> None:
        self.ejecutadas: list[str] = []

    def dispatch(self, call: ToolCall) -> _Resultado:
        self.ejecutadas.append(call.name)
        return _Resultado()


class _Speaker:
    def __init__(self) -> None:
        self.dicho: list[str] = []
        self.esperas = 0

    def say(self, text: str) -> None:
        self.dicho.append(text)

    def wait_until_done(self) -> None:
        self.esperas += 1


class _Gate:
    """Verificacion de locutor guionizada."""

    def __init__(self, veredictos: list[bool]) -> None:
        self._veredictos = list(veredictos)

    def check(self, audio: np.ndarray, sample_rate: int) -> object:
        aceptado = self._veredictos.pop(0) if self._veredictos else True
        return type("V", (), {"accepted": aceptado, "reason": "coseno de prueba"})()


def _orden(tool: str, stage: Stage = Stage.LITERAL) -> RouteResult:
    return RouteResult(stage=stage, tool_call=ToolCall(name=tool))


def _banco(
    *,
    guion: list[Utterance],
    textos: list[str],
    rutas: dict[str, RouteResult],
    follow_up: FollowUpConfig | None = None,
    speaker_gate: object | None = None,
) -> tuple[Assistant, _Mic, _Registry, _ScriptedRecorder]:
    stop = threading.Event()
    mic = _Mic()
    registry = _Registry()
    recorder = _ScriptedRecorder(guion, stop)
    assistant = Assistant(
        mic,  # type: ignore[arg-type]
        recorder,  # type: ignore[arg-type]
        _Transcriber(textos),  # type: ignore[arg-type]
        _Router(rutas),  # type: ignore[arg-type]
        registry,  # type: ignore[arg-type]
        _Speaker(),  # type: ignore[arg-type]
        keyphrase=object(),  # type: ignore[arg-type]
        # La SNR se calcula sobre audio sintetico, que no es voz: dejarla
        # activa haria fallar estos tests por una razon que no es la suya.
        vad_config=VadConfig(min_snr_db=0.0),
        speaker_gate=speaker_gate,  # type: ignore[arg-type]
        stop=stop,
        follow_up=follow_up,
    )
    return assistant, mic, registry, recorder


# --------------------------------------------------------------------------
# lo que la ventana acepta
# --------------------------------------------------------------------------


def test_a_simple_command_runs_without_the_wake_word() -> None:
    """El caso de uso entero: "Apolo, sube el volumen" y luego "mas" a secas."""
    assistant, _, registry, _ = _banco(
        guion=[VOZ],
        textos=["mas"],
        rutas={"mas": _orden("volume.step")},
    )

    assistant._follow_up_window(iter(range(10)), _orden("volume.step"))

    assert registry.ejecutadas == ["volume.step"]


def test_each_accepted_command_renews_the_window() -> None:
    """Encadenar cinco ordenes necesita cinco ventanas de cinco segundos, no
    una de veinticinco: si no se renovara, la tercera llegaria tarde."""
    assistant, _, registry, _ = _banco(
        guion=[VOZ, VOZ, VOZ],
        textos=["mas", "mas", "siguiente"],
        rutas={"mas": _orden("volume.step"), "siguiente": _orden("media.next")},
    )

    assistant._follow_up_window(iter(range(10)), _orden("volume.step"))

    assert registry.ejecutadas == ["volume.step", "volume.step", "media.next"]


def test_silence_inside_the_window_is_not_a_command() -> None:
    """Que no hables no cierra nada por si mismo: sigue escuchando hasta que
    se acabe el plazo. Lo que cierra la ventana es oir algo que no vale."""
    assistant, _, registry, recorder = _banco(
        guion=[SILENCIO, SILENCIO, VOZ],
        textos=["siguiente"],
        rutas={"siguiente": _orden("media.next")},
    )

    assistant._follow_up_window(iter(range(10)), _orden("volume.step"))

    assert registry.ejecutadas == ["media.next"]
    assert len(recorder.esperas) >= 3


# --------------------------------------------------------------------------
# lo que la ventana IGNORA. Es la parte que importa.
# --------------------------------------------------------------------------


def test_a_command_outside_the_allowlist_is_ignored() -> None:
    """"abre el navegador" es una orden perfectamente valida... con palabra
    clave. Aqui no: abriria una ventana encima de lo que estes haciendo."""
    assistant, _, registry, _ = _banco(
        guion=[VOZ],
        textos=["abre el navegador"],
        rutas={"abre el navegador": _orden("open.target")},
    )

    assistant._follow_up_window(iter(range(10)), _orden("volume.step"))

    assert registry.ejecutadas == []


def test_anything_that_needed_the_llm_is_ignored() -> None:
    """El filtro que mas trabaja. La clase `_fallback` del catalogo manda al
    LLM todo lo que no se parece a un comando, o sea toda la conversacion
    ajena que caiga dentro de la ventana. Se rechaza AUNQUE la accion que
    propone este en la allowlist: que hiciera falta el modelo ya dice que no
    era una orden simple.
    """
    assistant, _, registry, _ = _banco(
        guion=[VOZ],
        textos=["oye pues no se, sube un poco eso de ahi supongo"],
        rutas={
            "oye pues no se, sube un poco eso de ahi supongo": _orden(
                "volume.step", stage=Stage.LLM
            )
        },
    )

    assistant._follow_up_window(iter(range(10)), _orden("volume.step"))

    assert registry.ejecutadas == []


def test_the_first_thing_it_does_not_accept_closes_the_window() -> None:
    """Si lo que suena delante del microfono ya no son ordenes, se vuelve a
    exigir la palabra clave en vez de seguir escuchando por si acaso."""
    assistant, _, registry, _ = _banco(
        guion=[VOZ, VOZ],
        textos=["abre el navegador", "siguiente"],
        rutas={
            "abre el navegador": _orden("open.target"),
            "siguiente": _orden("media.next"),
        },
    )

    assistant._follow_up_window(iter(range(10)), _orden("volume.step"))

    # La segunda frase SI valia, y aun asi no se ejecuta: la ventana ya estaba
    # cerrada. Es deliberado, y es lo que hace acotado el peor caso.
    assert registry.ejecutadas == []


def test_the_speaker_gate_applies_inside_the_window() -> None:
    """Al reves que en el turno normal. Alli acababas de decir la palabra
    clave; aqui no ha dicho nada nadie, asi que es justo donde la verificacion
    de locutor se gana el sitio."""
    assistant, _, registry, _ = _banco(
        guion=[VOZ],
        textos=["siguiente"],
        rutas={"siguiente": _orden("media.next")},
        speaker_gate=_Gate([False]),
    )

    assistant._follow_up_window(iter(range(10)), _orden("volume.step"))

    assert registry.ejecutadas == []


# --------------------------------------------------------------------------
# cuando la ventana ni siquiera se abre
# --------------------------------------------------------------------------


def test_the_window_does_not_open_if_the_assistant_did_nothing() -> None:
    """Un turno que no acabo en accion ni en respuesta significa que lo que
    llego al microfono probablemente no iba dirigido al asistente. Es la peor
    situacion posible para dejar de exigir la palabra clave."""
    assistant, mic, registry, recorder = _banco(
        guion=[VOZ], textos=["siguiente"], rutas={"siguiente": _orden("media.next")}
    )

    assistant._follow_up_window(iter(range(10)), RouteResult(stage=Stage.LLM))

    assert recorder.esperas == []
    assert registry.ejecutadas == []
    assert mic.descartes == 0


def test_a_spoken_answer_does_open_the_window() -> None:
    """Preguntar la hora y encadenar "sube el volumen" es tan legitimo como
    encadenar dos ordenes."""
    assistant, _, registry, _ = _banco(
        guion=[VOZ],
        textos=["mas"],
        rutas={"mas": _orden("volume.step")},
    )

    assistant._follow_up_window(
        iter(range(10)), RouteResult(stage=Stage.SEMANTIC, reply=SpeechReply(text="las tres"))
    )

    assert registry.ejecutadas == ["volume.step"]


def test_disabling_it_removes_the_window_entirely() -> None:
    assistant, mic, _, recorder = _banco(
        guion=[VOZ],
        textos=["siguiente"],
        rutas={"siguiente": _orden("media.next")},
        follow_up=FollowUpConfig(enabled=False),
    )

    assistant._follow_up_window(iter(range(10)), _orden("volume.step"))

    assert recorder.esperas == []
    assert mic.descartes == 0


# --------------------------------------------------------------------------
# no oirse a si mismo
# --------------------------------------------------------------------------


def test_it_waits_for_its_own_voice_before_listening_again() -> None:
    """El fallo que haria inutilizable la ventana: Piper habla por los
    altavoces mientras el microfono sigue grabando, asi que sin esperar a que
    termine Y tirar lo grabado, el asistente se transcribiria a si mismo y se
    contestaria solo.
    """
    assistant, mic, _, _ = _banco(
        guion=[VOZ],
        textos=["mas"],
        rutas={"mas": _orden("volume.step")},
    )

    assistant._follow_up_window(iter(range(10)), _orden("volume.step"))

    speaker = assistant._speaker
    # Dos veces: al abrir la ventana y al renovarla tras la orden aceptada.
    assert speaker.esperas == 2  # type: ignore[attr-defined]
    # Y una vez mas al cerrar, para que el bucle principal no empiece
    # mordiendo audio viejo.
    assert mic.descartes == 3


def test_the_wait_never_overshoots_the_deadline() -> None:
    """La espera de cada trozo se acota con lo que queda de ventana: si no,
    una ventana de 5 s podria estirarse a 7 esperando a alguien que ya no va a
    decir nada."""
    assistant, _, _, recorder = _banco(
        guion=[SILENCIO, SILENCIO, SILENCIO],
        textos=[],
        rutas={},
        # Menos que los 2 s por defecto del recorder: asi el tope lo tiene que
        # poner lo que queda de ventana. Con una ventana mas larga el test
        # pasaria sin comprobar nada.
        follow_up=FollowUpConfig(window_s=1.5),
    )

    assistant._follow_up_window(iter(range(10)), _orden("volume.step"))

    assert recorder.esperas
    assert all(espera <= 1.5 for espera in recorder.esperas)
    # Y decrecen: cada vuelta queda menos ventana por delante.
    assert recorder.esperas == sorted(recorder.esperas, reverse=True)


# --------------------------------------------------------------------------
# configuracion
# --------------------------------------------------------------------------


def test_a_typo_in_the_allowlist_is_reported(caplog: pytest.LogCaptureFixture) -> None:
    """Un nombre mal escrito no da error: esa orden simplemente no se acepta
    nunca. Llega al usuario como "a veces no me hace caso", que es de lo mas
    caro de diagnosticar, asi que se dice al arrancar."""
    with caplog.at_level("WARNING"):
        _banco(
            guion=[],
            textos=[],
            rutas={},
            follow_up=FollowUpConfig(tools=frozenset({"media.next", "volumen.subir"})),
        )

    assert "volumen.subir" in caplog.text


def test_the_shipped_allowlist_only_names_real_skills() -> None:
    """Los valores por defecto tienen que valer tal cual: un nombre inventado
    aqui seria una funcion que no existe en ninguna maquina."""
    from asistente.config import Config

    config = Config.load("config.yaml")

    assert config.follow_up.tools <= _Registry.names | {"spotify.like"}
    assert config.follow_up.enabled is True


def test_the_shipped_allowlist_has_nothing_irreversible() -> None:
    """El criterio para entrar en la allowlist, escrito como test: si la tele
    lo dispara, tiene que poder deshacerse hablando. Abrir programas o buscar
    en la web no cumple."""
    from asistente.config import Config

    config = Config.load("config.yaml")

    prohibidas = {"open.target", "app.close", "web.search", "spotify.play"}
    assert not (config.follow_up.tools & prohibidas)


# --------------------------------------------------------------------------
# el buffer del microfono
# --------------------------------------------------------------------------


def test_discarding_pending_audio_empties_queue_and_preroll() -> None:
    """La pieza que impide que el asistente se oiga a si mismo.

    El callback del driver no para nunca: mientras Piper habla, la voz del
    propio asistente se va encolando aqui. Se cuenta lo descartado en segundos
    porque es el numero que dice si el problema es real -si aparece "0.1 s" no
    habia nada que tirar, si aparece "2.4 s" se estaba a punto de transcribir
    una frase entera de las suyas.
    """
    from asistente.audio.capture import MicrophoneStream

    mic = MicrophoneStream(sample_rate=SAMPLE_RATE, block_size=1280)
    bloque = np.zeros((1280, 1), dtype=np.float32)
    for _ in range(10):
        mic._on_audio(bloque, 1280, None, None)

    descartado = mic.discard_pending()

    assert descartado == pytest.approx(10 * 1280 / SAMPLE_RATE)
    assert mic._queue.empty()
    # El preroll tambien: es audio del mismo momento, y anteponerlo a la
    # siguiente frase volveria a meter justo lo que se acaba de tirar.
    assert mic.preroll_audio().size == 0


def test_discarding_nothing_is_not_an_error() -> None:
    """Se llama tras cada orden, tambien cuando el asistente no dijo nada."""
    from asistente.audio.capture import MicrophoneStream

    mic = MicrophoneStream(sample_rate=SAMPLE_RATE, block_size=1280)

    assert mic.discard_pending() == 0.0
