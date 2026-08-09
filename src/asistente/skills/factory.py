"""Construccion del registro de skills.

Un unico sitio donde se decide que acciones existen. Cambiar la lista de aqui es
lo unico que hace falta para anadir o quitar una capacidad del asistente, y es
tambien la lista que hay que revisar en una auditoria de seguridad.
"""

from __future__ import annotations

from asistente.config import Config, Secrets
from asistente.skills.apps import CloseAppSkill
from asistente.skills.base import Skill
from asistente.skills.browser import Browser
from asistente.skills.media import NextTrackSkill, PlayPauseSkill, PreviousTrackSkill
from asistente.skills.opening import OpenTargetSkill
from asistente.skills.registry import SkillRegistry
from asistente.skills.spotify import (
    LikeSkill,
    ShuffleSkill,
    SpotifyClient,
    SpotifyPlaySkill,
    WhatSongSkill,
)
from asistente.skills.system import DateSkill, TimeSkill
from asistente.skills.volume import (
    MuteSkill,
    VolumeController,
    VolumeQuerySkill,
    VolumeSetSkill,
    VolumeStepSkill,
)
from asistente.skills.web import WebSearchSkill


def build_registry(config: Config, secrets: Secrets) -> tuple[SkillRegistry, SpotifyClient]:
    """Instancia todas las skills. Devuelve tambien el cliente de Spotify para
    que el arranque pueda conectarlo y calentarlo antes del primer comando."""
    spotify = SpotifyClient(config, secrets)
    browser = Browser.from_path(config.web.browser, config.web.browser_path)
    use_keys = config.spotify.fallback_to_media_keys
    volume = VolumeController(spotify)

    skills: list[Skill] = [
        NextTrackSkill(spotify, use_keys),
        PreviousTrackSkill(spotify, use_keys),
        PlayPauseSkill(spotify, use_keys),
        SpotifyPlaySkill(spotify),
        WhatSongSkill(spotify),
        LikeSkill(spotify),
        ShuffleSkill(spotify),
        VolumeStepSkill(volume),
        VolumeSetSkill(volume),
        MuteSkill(volume),
        VolumeQuerySkill(volume),
        TimeSkill(),
        DateSkill(),
        OpenTargetSkill(config, browser),
        CloseAppSkill(config),
        WebSearchSkill(config, browser),
    ]
    return SkillRegistry(skills), spotify
