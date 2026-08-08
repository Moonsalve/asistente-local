"""Registro y despacho de skills: la unica puerta hacia la ejecucion.

Aplica dos controles, en este orden:

1. **Allowlist por nombre.** Si `ToolCall.name` no esta registrado, se rechaza.
   No hay resolucion dinamica, ni `getattr`, ni import por nombre: un LLM que
   alucine `system.shutdown` o `os.system` no encuentra nada que invocar.
2. **Validacion de argumentos.** Cada skill declara su modelo Pydantic con
   `extra='forbid'`. Argumentos de mas, de menos o del tipo equivocado abortan
   la ejecucion.

Cierra la via de inyeccion por voz: alguien que grite un comando cerca del
microfono solo puede disparar acciones que ya estaban autorizadas.
"""

from __future__ import annotations

import logging

from pydantic import ValidationError

from asistente.router.schema import ToolCall
from asistente.skills.base import Skill, SkillResult

log = logging.getLogger(__name__)


class SkillRegistry:
    def __init__(self, skills: list[Skill]) -> None:
        self._skills: dict[str, Skill] = {}
        for skill in skills:
            if skill.name in self._skills:
                raise ValueError(f"skill duplicada: {skill.name}")
            self._skills[skill.name] = skill

    @property
    def names(self) -> frozenset[str]:
        return frozenset(self._skills)

    def describe(self) -> list[dict[str, object]]:
        """Catalogo de herramientas para el prompt del LLM."""
        return [
            {
                "name": skill.name,
                "description": skill.description,
                "args": skill.args_model.model_json_schema().get("properties", {}),
            }
            for skill in self._skills.values()
        ]

    def verify_catalog(self, tools_referenced: set[str]) -> None:
        """Comprueba que todo `tool` del YAML tiene skill registrada.

        Se llama al arrancar. Un typo en `commands.yaml` se manifestaria si no
        como un comando que se reconoce pero no hace nada, que es de los fallos
        mas confusos de diagnosticar.
        """
        if missing := tools_referenced - self.names:
            raise ValueError(
                f"commands.yaml referencia tools sin skill registrada: {sorted(missing)}"
            )

    def dispatch(self, call: ToolCall) -> SkillResult:
        skill = self._skills.get(call.name)
        if skill is None:
            log.warning("tool desconocida rechazada: %s", call.name)
            return SkillResult.failed("No conozco esa acción.")

        try:
            args = skill.args_model.model_validate(call.args)
        except ValidationError as exc:
            log.warning("argumentos invalidos para %s: %s", call.name, exc)
            return SkillResult.failed("No entendí bien qué querías que hiciera.")

        try:
            return skill.execute(args)
        except Exception:
            # Una skill que revienta no debe tumbar el asistente: el usuario
            # sigue hablando y el bucle tiene que seguir vivo.
            log.exception("la skill %s fallo", call.name)
            return SkillResult.failed("Algo falló al ejecutar eso.")
