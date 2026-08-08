"""Configuracion tipada del asistente.

Todo lo ajustable vive en `config.yaml`; los secretos, en variables de entorno
(`.env`). La separacion es deliberada: `config.yaml` se versiona, `.env` no.

Los valores por defecto de aqui son los del plan. Los umbrales marcados como
"tuning" son los que hay que barrer con `scripts/benchmark.py` en el PC Windows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AudioConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    #: None = dispositivo por defecto del sistema. Usa `python -m sounddevice`
    #: para listar los indices disponibles.
    input_device: int | None = None
    sample_rate: int = 16_000
    #: Tamano de bloque del callback. 80 ms a 16 kHz: suficientemente pequeno
    #: para no anadir latencia perceptible, suficientemente grande para no
    #: saturar el hilo de audio.
    block_size: int = 1280


class WakeWordConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    #: Modelo preentrenado de openWakeWord. Para un wake word propio, apunta
    #: aqui al .onnx entrenado con el pipeline sintetico de openWakeWord.
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
    timeout_s: float = Field(default=10.0, gt=0.0)


class TtsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    #: Ruta al .onnx de la voz Piper (el .json debe estar junto a el).
    voice_model: Path = Path("models/es_MX-claude-high.onnx")
    speed: float = Field(default=1.0, gt=0.0)
    #: Sintetiza y reproduce por frases mientras el LLM sigue generando, para
    #: que el tiempo hasta el primer audio no dependa del largo de la respuesta.
    stream_by_sentence: bool = True


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
    scopes: tuple[str, ...] = (
        "user-modify-playback-state",
        "user-read-playback-state",
    )
    #: Si la API falla o no hay dispositivo activo, se recurre a las teclas
    #: multimedia de Windows. Menos capaz pero nunca deja el comando sin efecto.
    fallback_to_media_keys: bool = True


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    audio: AudioConfig = AudioConfig()
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
    search_url: str = "https://duckduckgo.com/?q={query}"

    @model_validator(mode="after")
    def _check_site_urls(self) -> Self:
        for alias, url in self.sites.items():
            if not url.startswith(("http://", "https://")):
                raise ValueError(f"sitio '{alias}': la URL debe ser http(s), no {url!r}")
        return self

    @classmethod
    def load(cls, path: Path) -> Config:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls.model_validate(raw)


class Secrets(BaseSettings):
    """Credenciales. Nunca se escriben en `config.yaml` ni se versionan."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    spotify_client_id: str = ""
    spotify_client_secret: str = ""
