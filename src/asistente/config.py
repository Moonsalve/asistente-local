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


class SttConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = "large-v3-turbo"
    device: str = "cuda"
    compute_type: str = "int8_float16"
    language: str = "es"
    #: Greedy en vez de beam search: ahorra 30-40% del tiempo con perdida
    #: despreciable en frases cortas de comando.
    beam_size: int = 1
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

    model: str = "qwen2.5:3b-instruct-q4_K_M"
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
        "user-modify-playback-state",   # play, pausa, siguiente, volumen
        "user-read-playback-state",     # saber que dispositivo esta activo
        "user-read-currently-playing",  # "que cancion es esta"
        "user-library-modify",          # "me gusta esta cancion"
        "user-library-read",
    )
    #: Si la API falla o no hay dispositivo activo, se recurre a las teclas
    #: multimedia de Windows. Menos capaz pero nunca deja el comando sin efecto.
    fallback_to_media_keys: bool = True


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    audio: AudioConfig = AudioConfig()
    web: WebConfig = WebConfig()
    discovery: DiscoveryConfig = DiscoveryConfig()
    wake_word: WakeWordConfig = WakeWordConfig()
    vad: VadConfig = VadConfig()
    stt: SttConfig = SttConfig()
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
