"""Cliente de Spotify y skill `spotify.play`.

AUTENTICACION
-------------
Se usa el flujo Authorization Code con PKCE. Autorizas UNA vez desde el
navegador; a partir de ahi spotipy guarda un refresh token en disco y lo renueva
solo. El cache va a `%LOCALAPPDATA%\\asistente-local`, nunca al repositorio: es
una credencial de larga duracion con permiso para controlar tu reproduccion.

PKCE en vez del flujo con client secret porque esto es una aplicacion de
escritorio: no hay servidor donde esconder un secreto, y PKCE esta disenado
exactamente para ese caso.

DEGRADACION
-----------
La Web API necesita un dispositivo activo. Si Spotify lleva rato cerrado no hay
ninguno, y la llamada falla. Por eso `media.*` cae a las teclas multimedia: el
comando siempre hace algo. `spotify.play` no puede degradar (buscar por nombre
requiere la API), asi que ahi si se avisa al usuario.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from asistente.config import Config, Secrets
from asistente.skills.base import Skill, SkillResult

log = logging.getLogger(__name__)


def _cache_path() -> Path:
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~/.cache")
    directory = Path(base) / "asistente-local"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / "spotify-token.json"


class SpotifyClient:
    """Envoltorio fino sobre spotipy con degradacion controlada.

    Ningun metodo lanza: devuelven bool. El asistente tiene que seguir vivo
    aunque la red se caiga a mitad de un comando.
    """

    def __init__(self, config: Config, secrets: Secrets) -> None:
        self._config = config.spotify
        self._client: Any | None = None
        self._secrets = secrets

    @property
    def available(self) -> bool:
        return self._config.enabled and bool(self._secrets.spotify_client_id)

    def connect(self) -> bool:
        """Crea el cliente. Abre el navegador solo la primera vez."""
        if not self.available:
            return False
        try:
            import spotipy
            from spotipy.oauth2 import SpotifyPKCE

            auth = SpotifyPKCE(
                client_id=self._secrets.spotify_client_id,
                redirect_uri=self._config.redirect_uri,
                scope=",".join(self._config.scopes),
                cache_handler=spotipy.CacheFileHandler(cache_path=str(_cache_path())),
                open_browser=True,
            )
            self._client = spotipy.Spotify(auth_manager=auth, requests_timeout=5)
        except Exception:
            log.exception("no se pudo conectar con Spotify")
            self._client = None
            return False
        return True

    def _active_device(self) -> str | None:
        if self._client is None:
            return None
        try:
            devices = self._client.devices().get("devices", [])
        except Exception:
            log.exception("no se pudieron listar los dispositivos de Spotify")
            return None
        for device in devices:
            if device.get("is_active"):
                return str(device["id"])
        # Sin dispositivo activo pero con alguno disponible: usar el primero
        # permite que "pon rock" funcione con Spotify abierto pero parado.
        return str(devices[0]["id"]) if devices else None

    def play_query(self, query: str) -> bool:
        """Busca y reproduce. Prueba playlist, luego album, luego cancion.

        El orden importa: "pon rock" casi siempre significa una playlist, no la
        primera cancion que se llame "Rock".
        """
        if self._client is None or (device := self._active_device()) is None:
            return False

        try:
            for kind, key in (("playlist", "playlists"), ("album", "albums")):
                results = self._client.search(q=query, type=kind, limit=1)
                if items := results.get(key, {}).get("items"):
                    self._client.start_playback(device_id=device, context_uri=items[0]["uri"])
                    return True

            results = self._client.search(q=query, type="track", limit=1)
            if items := results.get("tracks", {}).get("items"):
                self._client.start_playback(device_id=device, uris=[items[0]["uri"]])
                return True
        except Exception:
            log.exception("fallo la reproduccion de %r", query)
        return False

    def next_track(self) -> bool:
        return self._simple_action("next_track")

    def previous_track(self) -> bool:
        return self._simple_action("previous_track")

    def toggle_playback(self) -> bool:
        if self._client is None or (device := self._active_device()) is None:
            return False
        try:
            state = self._client.current_playback()
            if state is not None and state.get("is_playing"):
                self._client.pause_playback(device_id=device)
            else:
                self._client.start_playback(device_id=device)
        except Exception:
            log.exception("fallo el play/pause de Spotify")
            return False
        return True

    def _simple_action(self, method: str) -> bool:
        if self._client is None or (device := self._active_device()) is None:
            return False
        try:
            getattr(self._client, method)(device_id=device)
        except Exception:
            log.exception("fallo %s en Spotify", method)
            return False
        return True


class PlayArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=200)


class SpotifyPlaySkill(Skill):
    name = "spotify.play"
    args_model = PlayArgs
    description = (
        "Reproduce en Spotify una playlist, album, artista o cancion por nombre. "
        "Usar para 'pon X', 'reproduce X', 'quiero escuchar X'."
    )

    def __init__(self, client: SpotifyClient) -> None:
        self._client = client

    def execute(self, args: BaseModel) -> SkillResult:
        assert isinstance(args, PlayArgs)
        if not self._client.play_query(args.query):
            return SkillResult.failed(f"No pude reproducir {args.query} en Spotify.")
        return SkillResult.silent()
