"""Skills de reproduccion: siguiente, anterior, play/pausa.

Cada una intenta la Web API de Spotify primero (mas fiable cuando hay sesion
activa: sabe que esta sonando) y cae a las teclas multimedia si no hay
dispositivo o la red falla. Las teclas funcionan con cualquier reproductor, asi
que "pasa la cancion" sigue haciendo lo esperado con YouTube Music o VLC.

Estas son las skills mas frecuentes del sistema y todas son silenciosas: la
accion es su propia confirmacion. Que el asistente diga "listo" cada vez que
pasas de cancion es ruido.
"""

from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel

from asistente.skills.base import NoArgs, Skill, SkillResult
from asistente.skills.spotify import SpotifyClient
from asistente.skills.winkeys import VirtualKey, press


class _MediaSkill(Skill):
    """Base comun: intenta Spotify, cae a tecla multimedia."""

    args_model = NoArgs
    _key: VirtualKey
    _error: str

    def __init__(self, client: SpotifyClient | None, use_media_keys: bool = True) -> None:
        self._client = client
        self._use_media_keys = use_media_keys

    def _spotify_action(self) -> Callable[[], bool] | None:
        raise NotImplementedError

    def execute(self, args: BaseModel) -> SkillResult:
        if self._client is not None and (action := self._spotify_action()) is not None:
            if action():
                return SkillResult.silent()

        if self._use_media_keys and press(self._key):
            return SkillResult.silent()

        return SkillResult.failed(self._error)


class NextTrackSkill(_MediaSkill):
    name = "media.next"
    description = "Pasa a la siguiente cancion."
    _key = VirtualKey.MEDIA_NEXT_TRACK
    _error = "No pude cambiar de canción."

    def _spotify_action(self) -> Callable[[], bool] | None:
        return self._client.next_track if self._client else None


class PreviousTrackSkill(_MediaSkill):
    name = "media.previous"
    description = "Vuelve a la cancion anterior."
    _key = VirtualKey.MEDIA_PREV_TRACK
    _error = "No pude volver a la canción anterior."

    def _spotify_action(self) -> Callable[[], bool] | None:
        return self._client.previous_track if self._client else None


class PlayPauseSkill(_MediaSkill):
    name = "media.play_pause"
    description = "Pausa o reanuda la reproduccion."
    _key = VirtualKey.MEDIA_PLAY_PAUSE
    _error = "No pude pausar la música."

    def _spotify_action(self) -> Callable[[], bool] | None:
        return self._client.toggle_playback if self._client else None
