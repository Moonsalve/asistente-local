"""Skill `app.close`: cerrar aplicaciones de la allowlist."""

from __future__ import annotations

import logging
import subprocess
import sys

from pydantic import BaseModel, ConfigDict, Field

from asistente.config import Config
from asistente.skills.base import Skill, SkillResult
from asistente.skills.resolve import build_alias_index, resolve

log = logging.getLogger(__name__)


class CloseArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    app: str = Field(min_length=1, max_length=120)


class CloseAppSkill(Skill):
    name = "app.close"
    args_model = CloseArgs
    description = "Cierra una aplicacion abierta. Usar para 'cierra X', 'mata X'."

    def __init__(self, config: Config) -> None:
        self._config = config
        self._index = build_alias_index(config.apps)

    def execute(self, args: BaseModel) -> SkillResult:
        assert isinstance(args, CloseArgs)

        app_name = resolve(args.app, self._index)
        if app_name is None:
            # No se cierra a ciegas: sin entrada en la allowlist no hay nombre
            # de proceso fiable, y matar el proceso equivocado es destructivo.
            return SkillResult.failed(f"No tengo {args.app} en la lista de aplicaciones.")

        spec = self._config.apps[app_name]
        if spec.process is None:
            return SkillResult.failed(f"No se cual es el proceso de {app_name}.")

        if sys.platform != "win32":
            log.info("cierre de %s omitido: no estamos en Windows", spec.process)
            return SkillResult.failed("Cerrar aplicaciones solo funciona en Windows.")

        # /T cierra tambien los procesos hijos (los navegadores abren uno por
        # pestana). /F fuerza: sin el, una app con dialogo abierto ignora la
        # peticion y el comando parece no hacer nada.
        result = subprocess.run(  # noqa: S603
            ["taskkill", "/F", "/T", "/IM", f"{spec.process}.exe"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            log.debug("taskkill %s: %s", spec.process, result.stderr.strip())
            return SkillResult.failed(f"{app_name} no estaba abierto.")
        return SkillResult.silent()
