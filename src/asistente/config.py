"""Configuracion tipada del asistente.

Todo lo ajustable vive en `config.yaml`; los secretos, en variables de entorno
(`.env`). La separacion es deliberada: `config.yaml` se versiona, `.env` no.

Los valores por defecto de aqui son los del plan. Los umbrales marcados como
"tuning" son los que hay que barrer con `scripts/benchmark.py` en el PC Windows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AudioConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    #: None = dispositivo por defecto del sistema. Usa `python -m sounddevice`
    #: para listar los indices disponibles.
    input_device: int | None = None
    sample_rate: int = 16_000

    #: Amplificacion por software de la senal de entrada, para microfonos que
    #: capturan bajo incluso con el volumen de Windows al maximo.
    #:
    #: Afecta al VAD y a la palabra clave, que ven los bloques segun llegan. NO
    #: sustituye a la normalizacion previa al STT (`stt.normalize_audio`), que
    #: opera sobre la frase completa: son dos problemas distintos.
    #:
    #: Amplifica tambien el ruido de fondo, asi que no conviene pasarse. El
    #: valor exacto que necesita TU microfono lo calcula:
    #:     python scripts/diagnose_audio.py
    gain: float = Field(default=1.0, gt=0.0, le=50.0)
    #: Tamano de bloque del callback. 80 ms a 16 kHz: suficientemente pequeno
    #: para no anadir latencia perceptible, suficientemente grande para no
    #: saturar el hilo de audio.
    block_size: int = 1280


class WakeWordConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    #: "openwakeword": modelo dedicado siempre escuchando (1-2% de CPU), pero
    #:   solo funciona con palabras para las que exista modelo entrenado, y los
    #:   preentrenados son todos ingleses.
    #: "transcript": transcribe lo que oye y busca la clave en el texto. Acepta
    #:   CUALQUIER frase en espanol sin entrenar nada, a cambio de ejecutar
    #:   Whisper cada vez que alguien habla cerca del microfono.
    mode: Literal["openwakeword", "transcript"] = "transcript"

    #: Solo en modo "transcript". Varias formas porque Whisper no siempre
    #: transcribe igual un nombre propio.
    phrases: tuple[str, ...] = ("apolo", "apollo", "a polo")
    #: Similitud minima (0-100) para dar por dicha la clave.
    phrase_threshold: float = Field(default=82.0, ge=0.0, le=100.0)

    #: Solo en modo "openwakeword". Preentrenados: hey_jarvis, alexa,
    #: hey_mycroft, hey_rhasspy. Para uno propio, ruta a tu .onnx.
    model: str = "hey_jarvis"
    #: TUNING. Subir reduce falsos positivos y aumenta los fallos de activacion.
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    #: Ventana tras la activacion en la que se ignoran nuevas detecciones, para
    #: que una sola palabra clave no dispare varias veces.
    refractory_s: float = Field(default=2.0, gt=0.0)


class VadConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    #: TUNING - EL PARAMETRO MAS IMPORTANTE DEL SISTEMA. Es el 40-50% de la
    #: latencia total percibida. 0.35 s es el punto de partida seguro; 0.25 s se
    #: nota mucho y rara vez corta comandos cortos.
    silence_s: float = Field(default=0.35, gt=0.0)
    speech_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    #: Corta la grabacion pase lo que pase. Evita que un microfono ruidoso deje
    #: el pipeline grabando indefinidamente.
    max_utterance_s: float = Field(default=15.0, gt=0.0)
    #: Audio previo al inicio del habla que se conserva, para no comerse la
    #: primera silaba.
    preroll_s: float = Field(default=0.3, ge=0.0)

    #: Voz acumulada minima para que la grabacion merezca un STT. Un ventilador
    #: que roza `speech_threshold` produce grabaciones largas con casi nada de
    #: voz dentro; esto las descarta por unos microsegundos en vez de por los
    #: ~120 ms que cuesta Whisper.
    #:
    #: No subir mucho: "no" o "pausa" duran poco mas de 0.2 s.
    min_speech_s: float = Field(default=0.25, ge=0.0)
    #: SNR minima estimada de la grabacion, en dB. Por debajo, la voz no destaca
    #: sobre el fondo lo suficiente para transcribirse bien, y lo que sale es
    #: una alucinacion. 0 desactiva la comprobacion.
    min_snr_db: float = Field(default=6.0, ge=0.0)


class FollowUpConfig(BaseModel):
    """Ventana para encadenar ordenes sin repetir la palabra clave.

    Tras ejecutar una orden, el asistente sigue escuchando unos segundos. Es lo
    que convierte "Apolo, sube el volumen" / "Apolo, mas" / "Apolo, mas" en
    "Apolo, sube el volumen" / "mas" / "mas", que es como se ajusta el volumen
    de verdad: a tientas y en varios pasos.

    LO QUE SE ESTA QUITANDO
    -----------------------
    Dentro de la ventana NO hay palabra clave, y la palabra clave es la unica
    defensa real contra ejecutar lo que oye de la tele o de una conversacion
    ajena. Por eso la ventana no acepta cualquier cosa, sino tres filtros a la
    vez, y los tres tienen que pasar:

      1. la accion esta en `tools` (abajo). Nada de abrir programas, buscar en
         la web o reproducir algo concreto: solo lo reversible.
      2. el router la resolvio SIN LLM. Si hubo que escalar al modelo, por
         definicion no era una orden simple y no toca ejecutarla a ciegas.
      3. si tienes verificacion de locutor activada, tambien tiene que pasarla.
         Aqui SI se aplica -a diferencia del turno normal- porque aqui nadie ha
         dicho la palabra clave.

    El peor caso que queda es que la tele diga algo que suene a "siguiente" y
    salte una cancion. Se arregla diciendo "anterior". Ese es el listón: dentro
    de la ventana solo entra lo que se deshace hablando.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    #: TUNING. Cuanto se sigue escuchando tras cada orden. 5 s dan tiempo a
    #: pensar la siguiente sin dejar el microfono abierto de forma perceptible.
    #: Cada orden aceptada REINICIA la cuenta, asi que encadenar cinco ordenes
    #: no necesita una ventana de 25 s.
    #:
    #: El minimo es 1 s y no 0: por debajo de `pipeline._MIN_LISTEN_S` el bucle
    #: no llega a grabar nada, con lo que la ventana quedaria activada pero sin
    #: efecto. Rechazarlo al cargar la config es mejor que descubrirlo hablando.
    window_s: float = Field(default=5.0, ge=1.0, le=60.0)

    #: Allowlist por nombre de SKILL (no de intent): es lo que trae el
    #: `ToolCall` que sale del router. `volume.step` cubre subir y bajar.
    #:
    #: Anadir aqui es facil y por eso hay que pensarlo: la pregunta no es "es
    #: comodo?" sino "que pasa si lo dispara la tele?". `open.target` abriria
    #: ventanas encima de lo que estes haciendo; `web.search`, un navegador.
    tools: frozenset[str] = frozenset(
        {
            "media.next",
            "media.previous",
            "media.play_pause",
            "volume.step",
            "volume.set",
            "volume.mute",
            "volume.query",
        }
    )


class SttConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = "large-v3-turbo"
    device: str = "cuda"
    compute_type: str = "int8_float16"
    language: str = "es"
    #: Greedy en vez de beam search: ahorra 30-40% del tiempo con perdida
    #: despreciable en frases cortas de comando.
    #:
    #: MEDIDO, y por eso sigue en 1: sobre 24 clips de titulos en ingles dichos
    #: en espanol -el caso que peor se transcribia- subirlo a 5 movio el WER de
    #: 22.4% a 21.6% con large-v3-turbo, o sea dentro del ruido, a cambio de un
    #: 39% mas de tiempo. Con `small` daba lo mismo (39.6% -> 38.4%).
    #:
    #: Es la unica conclusion que aguanto con los dos modelos: subir el beam no
    #: arregla los titulos que el modelo no conoce.
    beam_size: int = 1

    #: Sesga el decodificador con los nombres de TU biblioteca de Spotify.
    #:
    #: DESACTIVADO POR UNA MEDIDA, y merece la pena contar la historia entera
    #: porque el resultado se dio la vuelta al usar el modelo de verdad.
    #:
    #: Con Whisper `small` (24 clips, voces ES diciendo titulos en ingles):
    #:     beam=1              WER 39.6%
    #:     beam=5              WER 38.4%
    #:     beam=1 + hotwords   WER 33.4%   <- parecia la solucion
    #:
    #: Con `large-v3-turbo`, QUE ES EL QUE CORRE AQUI, los mismos 24 clips:
    #:     beam=1              WER 22.4%   8/24 exactos
    #:     beam=5              WER 21.6%   7/24
    #:     beam=5 + hotwords   WER 24.1%   7/24
    #:     beam=1 + hotwords   WER 24.8%   6/24   <- PEOR que sin nada
    #:
    #: Turbo ya acierta los titulos en ingles por su cuenta (la mitad del WER de
    #: small), asi que el sesgo no tiene hueco donde ayudar y si donde estorbar.
    #: Y el modo de fallo es el peor posible: PEGA EL VERBO AL TITULO
    #: ("Ponlovers Rock de TV Girl"), con lo que la frase ya no casa con el
    #: patron de spotify.play y el comando no llega ni a enrutarse. Tambien
    #: convirtio un "de Metallica" correcto en ", The Metallica".
    #:
    #: Lo que si arregla el residuo -turbo escribe "TV Gery" por "TV Girl"- es
    #: la busqueda en tus me gusta, que puntua ese par a 90.9 sobre un umbral de
    #: 72. Ver `skills/spotify.py`.
    #:
    #: Se deja activable porque con otro modelo (small, medium) la tabla decia
    #: lo contrario. Si lo activas, comprueba que no te parte los verbos.
    hotwords_from_spotify: bool = False
    #: Cuantos terminos como maximo. Whisper acota el contexto del prompt a unos
    #: 224 tokens; pasarse de ahi no anade nada y empieza a desplazar al audio.
    hotwords_limit: int = Field(default=60, ge=0, le=200)
    #: Sin contexto entre turnos: cada comando es independiente, y arrastrar el
    #: anterior invita a alucinaciones.
    condition_on_previous_text: bool = False

    #: Normaliza el pico de cada frase antes de transcribirla. Whisper se
    #: entreno con audio a nivel normal y transcribe notablemente peor una
    #: grabacion muy floja. Es gratis (una multiplicacion) y no toca el ruido
    #: relativo, asi que conviene dejarlo activado.
    normalize_audio: bool = True
    #: Pico objetivo. 0.95 y no 1.0 para dejar margen y no recortar.
    normalize_peak: float = Field(default=0.95, gt=0.0, le=1.0)

    #: Resta espectral del ruido estacionario (ventilador, aire, zumbido) antes
    #: de transcribir. Ver `audio/denoise.py`.
    #:
    #: LO QUE ESTA MEDIDO (12 clips de voz sintetica + ruido simulado, Whisper
    #: small en CPU, que es un banco mas duro que una habitacion real):
    #:   - ruido de ventilador a  5 dB SNR: WER 97.9% -> 72.9%   mejora clara
    #:   - ruido de ventilador a 10 dB SNR: WER 66.0% -> 68.1%   igual
    #:   - sobre audio limpio: cambia el RMS en 2e-8, o sea nada
    #: Ayuda cuando hace falta y no estorba cuando no.
    denoise: bool = True
    #: Sobre-resta. Por encima de 1 se quita mas ruido del estimado, porque el
    #: percentil se queda corto. Pasarse abre agujeros en la voz.
    #:
    #: OJO: el barrido de este par de valores salio INCONCLUSO. Con 6 clips y un
    #: WER base del 80%, las diferencias entre configuraciones caian dentro de
    #: la varianza de la medida. Estos valores son un termino medio prudente, no
    #: un optimo: el unico barrido que vale es el que se haga en el PC, con el
    #: ventilador de verdad y con large-v3-turbo.
    denoise_strength: float = Field(default=1.5, ge=1.0, le=4.0)
    #: Atenuacion maxima por banda (0.15 = -16 dB). NO bajar a 0: los silencios
    #: absolutos son justo lo que dispara las alucinaciones de Whisper. Se elige
    #: 0.15 y no 0.05 -que midio algo mejor, dentro del ruido- porque el fallo
    #: que evita es el caro: alucinar produce una accion equivocada, no oir solo
    #: obliga a repetir.
    denoise_floor: float = Field(default=0.15, ge=0.01, le=1.0)

    #: Umbrales propios de Whisper para descartar lo que no es voz. Sin ellos su
    #: juicio se ignora y el ruido acaba convertido en texto con toda seguridad.
    #: Subir `no_speech` filtra mas; bajar `log_prob` filtra mas.
    no_speech_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    log_prob_threshold: float = -1.0


class SpeakerConfig(BaseModel):
    """Verificacion de locutor: responder solo a la voz enrolada.

    VIENE DESACTIVADA, y no por prudencia generica: esta MEDIDO que es la mas
    fragil de las tres defensas contra el ruido. Con ruido de ventilador el
    coseno de la propia voz baja de ~0.80 a 0.49-0.83 segun la voz, y con ruido
    de banda ancha se hunde a 0.29-0.44, que es territorio de "otra persona".
    O sea: falla precisamente cuando hay ruido, que es cuando se necesitaria.

    Contra el ventilador ya estan el denoise y las puertas del VAD, que son mas
    baratos y mas fiables. Esto solo aporta contra el ruido que HABLA -la tele,
    la musica cantada, otra persona-, y ahi si es la unica defensa que existe.

    Actívala con `python scripts/enroll_voice.py` y ajusta el umbral con los
    cosenos que el log imprime en cada turno, no a ojo.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    #: Fichero con el centroide de la voz. Lo genera `scripts/enroll_voice.py`.
    profile: Path = Path("voz.json")
    #: TUNING. Coseno minimo para aceptar la frase. Deliberadamente permisivo:
    #: un falso negativo deja el asistente sordo sin decir por que, que es mucho
    #: peor que atender de mas.
    threshold: float = Field(default=0.45, ge=-1.0, le=1.0)


class RouterConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    embedding_model: str = "intfloat/multilingual-e5-small"
    embedding_onnx_file: str = "onnx/model.onnx"
    embedding_on_gpu: bool = False
    #: TUNING. Red de seguridad para frases que no se parecen a nada. La
    #: decision principal la toma la clase negativa `_fallback` del catalogo,
    #: no este umbral. Medido: positivos desde 0.34, negativos por debajo.
    threshold: float = Field(default=0.25, ge=-1.0, le=1.0)


class LlmConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    #: UN SOLO MODELO PARA EL ROUTER Y PARA LA MUSICA, porque dos no caben.
    #:
    #: La cuenta, en una RTX de 8 GB:
    #:     Whisper large-v3-turbo   1.6 GB
    #:     qwen2.5 7B q4_K_M        4.7 GB
    #:     KV cache                 0.5 GB
    #:                              ------
    #:                              6.8 GB de 8
    #:
    #: Se sube de 3B a 7B por una sola razon: corregir "tibi guerl" -> "TV Girl"
    #: es conocimiento del mundo, y ahi el tamano del modelo SI se nota. Para el
    #: router daba igual —el catalogo resuelve el 85% de los turnos en 0-2 ms y
    #: el LLM es solo la red de seguridad—, asi que el 7B se paga en la unica
    #: capa que lo aprovecha.
    #:
    #: COMO REVERTIR: `llm.model: qwen2.5:3b-instruct-q4_K_M` en
    #: `config.local.yaml`. Hazlo si el log de Ollama menciona capas en CPU o si
    #: la latencia del turno se dispara: es que no cupo y esta descargando a RAM.
    model: str = "qwen2.5:7b-instruct-q4_K_M"
    host: str = "http://127.0.0.1:11434"
    #: -1 mantiene el modelo en VRAM indefinidamente. Sin esto, el primer
    #: comando tras un rato de inactividad tarda 5-10 s en volver a cargar.
    #:
    #: TIENE QUE SER NUMERO, NO CADENA. Ollama interpreta las cadenas como
    #: duraciones con unidad ("10m", "1h"); "-1" sin unidad da HTTP 400
    #: `missing unit in duration`. Como numero se lee en segundos, y -1 es el
    #: valor especial "no lo descargues nunca".
    keep_alive: int | str = -1
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    #: Timeout de una consulta normal. Corto a proposito: si el LLM tarda mas
    #: que esto, el comando ya se sintio lento y es mejor rendirse.
    timeout_s: float = Field(default=10.0, gt=0.0)
    #: Timeout del calentamiento. Mucho mas largo porque la primera peticion
    #: incluye cargar varios GB en VRAM, que con el disco frio pasa de un minuto.
    warmup_timeout_s: float = Field(default=180.0, gt=0.0)


class TtsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    #: Ruta al .onnx de la voz Piper (el .json debe estar junto a el).
    voice_model: Path = Path("models/es_MX-claude-high.onnx")
    speed: float = Field(default=1.0, gt=0.0)
    #: Sintetiza y reproduce por frases mientras el LLM sigue generando, para
    #: que el tiempo hasta el primer audio no dependa del largo de la respuesta.
    stream_by_sentence: bool = True


class WebConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    #: Navegador en el que se abre todo: busquedas y paginas. Conocidos:
    #: brave, chrome, firefox, edge, opera, vivaldi. Cadena vacia = el del
    #: sistema.
    browser: str = "brave"
    #: Ruta explicita, por si la deteccion automatica no lo encuentra.
    browser_path: Path | None = None
    #: Buscador. {query} se sustituye por lo que pidas, ya escapado.
    #:   Google       https://www.google.com/search?q={query}
    #:   Brave Search https://search.brave.com/search?q={query}
    #:   DuckDuckGo   https://duckduckgo.com/?q={query}
    search_url: str = "https://www.google.com/search?q={query}"


class DiscoveryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    #: Enumera al arrancar las aplicaciones instaladas y las anade a la
    #: allowlist. Lo declarado a mano en `apps:` SIEMPRE tiene prioridad.
    #:
    #: NOTA DE SEGURIDAD: con esto la allowlist deja de ser una lista curada y
    #: pasa a ser "todo lo que hay instalado". El limite sigue existiendo -solo
    #: se abre software ya presente en el equipo, y el LLM sigue sin poder
    #: ejecutar comandos arbitrarios- pero es una frontera mas ancha. Ponlo a
    #: false si prefieres controlar exactamente que se puede abrir.
    enabled: bool = True
    #: Incluir juegos de Steam.
    steam: bool = True
    #: Horas que vale el cache antes de volver a enumerar. Enumerar tarda unos
    #: segundos (lanza PowerShell), y las apps instaladas no cambian a menudo.
    cache_hours: float = Field(default=24.0, ge=0.0)


class AppSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    #: Comando o ruta a ejecutar. ALLOWLIST: solo se abre lo que este aqui.
    command: str
    #: Nombre del proceso para poder cerrarlo (sin .exe).
    process: str | None = None
    #: Como te refieres a ella al hablar. Se comparan normalizados.
    aliases: tuple[str, ...] = ()


class SpotifyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    redirect_uri: str = "http://127.0.0.1:8888/callback"
    #: Permisos que se piden al autorizar. Si anades uno DESPUES de haber
    #: autorizado, hay que borrar el token cacheado para que vuelva a pedirlo:
    #: el guardado solo sirve para los permisos con los que se creo.
    scopes: tuple[str, ...] = (
        "user-modify-playback-state",     # play, pausa, siguiente, volumen
        "user-read-playback-state",       # saber que dispositivo esta activo
        "user-read-currently-playing",    # "que cancion es esta"
        "user-library-modify",            # "me gusta esta cancion"
        "user-library-read",              # "pon mis me gusta"
        # Sin estos dos, `current_user_playlists` solo devuelve tus playlists
        # PUBLICAS, y "pon mi playlist de gym" no encontraria las privadas —que
        # son la mayoria— y acabaria reproduciendo una del catalogo publico que
        # se llame parecido.
        #
        # Anadir un permiso a esta lista NO obliga a borrar el token a mano:
        # comprobado en el codigo de spotipy 2.26, `validate_token` descarta el
        # cacheado si sus permisos no cubren los pedidos y vuelve a abrir el
        # navegador. Solo hay que borrarlo si eso no llegara a pasar.
        "playlist-read-private",
        "playlist-read-collaborative",
    )
    #: Si la API falla o no hay dispositivo activo, se recurre a las teclas
    #: multimedia de Windows. Menos capaz pero nunca deja el comando sin efecto.
    fallback_to_media_keys: bool = True
    #: Deja que el LLM local corrija el titulo cuando Spotify no encuentra nada.
    #: Ver `skills/music_ai.py`. Solo se paga en las peticiones que YA habian
    #: fallado, asi que apagarlo no acelera nada de lo que hoy funciona: lo
    #: unico que hace es devolver el asistente a "solo encuentra lo que escribes
    #: como Spotify lo tiene escrito".
    resolve_with_llm: bool = True
    #: Indexar tus Me Gusta al arrancar (hasta 20 peticiones) para poder
    #: emparejar contra ellos antes de tocar el catalogo publico.
    #:
    #: Sigue mereciendo la pena con el LLM detras, porque es el paso 2 de la
    #: cascada: local, gratis e instantaneo para lo que ya tienes, y evita los
    #: 600-900 ms del modelo justo en las canciones que mas escuchas. Con esto
    #: en `false` esas peticiones siguen funcionando, pero pagando el LLM.
    index_library: bool = True
    #: Procesos de Spotify. Solo lo usa `scripts/diagnose_volume.py`, para
    #: senalar a Spotify en la lista del mezclador de Windows y poder confirmar
    #: que esta sonando. El volumen NO se controla por ahi: ver skills/volume.py.
    process_names: tuple[str, ...] = ("Spotify.exe",)


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    audio: AudioConfig = AudioConfig()
    web: WebConfig = WebConfig()
    discovery: DiscoveryConfig = DiscoveryConfig()
    wake_word: WakeWordConfig = WakeWordConfig()
    vad: VadConfig = VadConfig()
    follow_up: FollowUpConfig = FollowUpConfig()
    stt: SttConfig = SttConfig()
    speaker: SpeakerConfig = SpeakerConfig()
    router: RouterConfig = RouterConfig()
    llm: LlmConfig = LlmConfig()
    tts: TtsConfig = TtsConfig()
    spotify: SpotifyConfig = SpotifyConfig()

    #: Allowlist de aplicaciones. Lo que no este aqui no se puede abrir.
    apps: dict[str, AppSpec] = Field(default_factory=dict)
    #: Sitios conocidos: alias -> URL. Lo que no este aqui cae a busqueda web.
    sites: dict[str, str] = Field(default_factory=dict)

    @property
    def search_url(self) -> str:
        return self.web.search_url

    @model_validator(mode="after")
    def _check_site_urls(self) -> Self:
        for alias, url in self.sites.items():
            if not url.startswith(("http://", "https://")):
                raise ValueError(f"sitio '{alias}': la URL debe ser http(s), no {url!r}")
        return self

    @classmethod
    def load(cls, path: Path | str) -> Config:
        """Carga `config.yaml` y le superpone `config.local.yaml` si existe.

        POR QUE HAY DOS FICHEROS
        ------------------------
        `config.yaml` esta versionado y trae los valores por defecto del
        proyecto. Pero varios ajustes son propios de CADA maquina -el indice del
        microfono, la ganancia que necesita, la ruta de las apps- y editarlos
        directamente hace que `git pull` choque una y otra vez con conflictos
        sobre un fichero que en realidad no es compartido.

        `config.local.yaml` esta en .gitignore y se fusiona ENCIMA, clave a
        clave. Solo hay que escribir en el lo que difiera del valor por defecto:

            audio:
              gain: 12.0        # solo esta clave; el resto de `audio` se hereda

        Asi los ajustes de tu maquina sobreviven a cualquier actualizacion.
        """
        base_path = Path(path)
        raw = yaml.safe_load(base_path.read_text(encoding="utf-8")) or {}

        local_path = base_path.with_name(f"{base_path.stem}.local{base_path.suffix}")
        if local_path.is_file():
            overrides = yaml.safe_load(local_path.read_text(encoding="utf-8")) or {}
            raw = _deep_merge(raw, overrides)

        return cls.model_validate(raw)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Fusiona recursivamente `override` sobre `base`.

    Recursiva y no plana: con una fusion plana, poner `audio: {gain: 12}` en el
    fichero local borraria `input_device`, `sample_rate` y `block_size`, y habria
    que repetir la seccion entera para cambiar un solo valor.
    """
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = value
    return merged


class Secrets(BaseSettings):
    """Credenciales. Nunca se escriben en `config.yaml` ni se versionan."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    spotify_client_id: str = ""
    spotify_client_secret: str = ""
