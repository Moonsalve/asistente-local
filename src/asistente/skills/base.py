"""Contrato de las skills.

Cada skill declara su propio modelo Pydantic de argumentos. El registro valida
contra ese modelo ANTES de ejecutar nada, asi que una skill nunca recibe
argumentos que no haya declarado, vengan del catalogo o de un LLM.

Esta es la frontera de seguridad del sistema: por aqui pasa todo lo que llega a
ejecutarse, y es el unico sitio donde hay que auditar.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

from pydantic import BaseModel


@dataclass(frozen=True, slots=True)
class SkillResult:
    """Resultado de ejecutar una skill.

    `speech` es lo que el asistente dira en voz alta. None significa silencio:
    para acciones obvias (pasar de cancion, subir volumen) confirmar hablando
    es ruido, la accion ya es su propia confirmacion.
    """

    ok: bool
    speech: str | None = None

    @classmethod
    def silent(cls) -> SkillResult:
        return cls(ok=True, speech=None)

    @classmethod
    def says(cls, text: str) -> SkillResult:
        return cls(ok=True, speech=text)

    @classmethod
    def failed(cls, reason: str) -> SkillResult:
        return cls(ok=False, speech=reason)


class NoArgs(BaseModel):
    """Modelo vacio para skills sin argumentos. `extra='forbid'` es deliberado:
    si el LLM inventa argumentos para una skill que no los tiene, falla aqui."""

    model_config = {"extra": "forbid"}


class Skill(ABC):
    """Una accion ejecutable.

    `name` debe coincidir con el `tool` declarado en `commands.yaml`; el registro
    verifica esa correspondencia al arrancar para que un typo en el YAML no se
    manifieste como un comando que no hace nada.
    """

    name: ClassVar[str]
    args_model: ClassVar[type[BaseModel]] = NoArgs
    #: Descripcion en lenguaje natural. La consume el prompt del LLM para saber
    #: que herramientas existen; mantenerla corta y orientada a la intencion.
    description: ClassVar[str] = ""

    @abstractmethod
    def execute(self, args: BaseModel) -> SkillResult: ...
