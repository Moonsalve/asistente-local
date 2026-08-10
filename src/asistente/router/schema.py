"""Contratos tipados del router.

Todo lo que sale del router (venga de un regex, de un embedding o del LLM) tiene
que pasar por `ToolCall`. Es el unico punto por el que se llega a ejecutar una
accion, y por eso valida con Pydantic en modo estricto: un LLM de 3B produce
JSON malformado o nombres de skill inventados con cierta frecuencia, y queremos
que eso reviente aqui y no dentro de un `subprocess`.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Stage(StrEnum):
    """Etapa del router que resolvio la frase. Se registra para el benchmark."""

    LITERAL = "literal"
    PATTERN = "pattern"
    SEMANTIC = "semantic"
    LLM = "llm"


class ToolCall(BaseModel):
    """Una accion concreta a ejecutar, ya validada.

    `name` se resuelve contra el registro de skills, que aplica la allowlist.
    `args` queda como dict porque cada skill valida sus propios argumentos con su
    modelo Pydantic (ver `skills/base.py`); centralizar aqui todos los schemas
    acoplaria el router a cada skill.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$")
    args: dict[str, str | int | float | bool] = Field(default_factory=dict)


class SpeechReply(BaseModel):
    """Respuesta hablada sin accion asociada (preguntas abiertas al LLM)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["speech"] = "speech"
    text: str


class RouteResult(BaseModel):
    """Lo que devuelve el router, con la telemetria necesaria para el tuning."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    stage: Stage
    tool_call: ToolCall | None = None
    reply: SpeechReply | None = None
    # Coseno centrado en la etapa semantica, 1.0 en la literal. Puede ser
    # negativo: tras centrar, dos frases opuestas apuntan en sentidos contrarios.
    score: float = Field(default=1.0, ge=-1.0, le=1.0)
    latency_s: float = Field(default=0.0, ge=0.0)


class IntentSpec(BaseModel):
    """Un intent tal y como se declara en `commands.yaml`.

    `examples` son las frases que se codifican a vectores al arrancar. No son
    patrones: son anclas semanticas. Cuantas mas y mas variadas, mejor cubre las
    paraphrasis que nunca escribiste.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    tool: str | None = Field(default=None, pattern=r"^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$")
    examples: tuple[str, ...] = Field(min_length=1)
    # Frases exactas (ya normalizadas) que resuelve la etapa literal sin coste.
    literal: tuple[str, ...] = ()
    # Clase negativa: sus ejemplos son charla y preguntas abiertas. Si gana el
    # coseno, la frase escala al LLM. Ver la nota de diseno en `semantic.py`
    # sobre por que esto sustituye al umbral absoluto.
    fallback: bool = False
    # Regex que DECIDEN el intent, no que extraen de uno ya decidido. Existen
    # para las frases cuyo contenido discriminante es sintactico y no semantico:
    # en "reproduce loving machine de tv girl" lo unico reconocible es el molde,
    # porque el titulo son nombres propios que el encoder nunca ha visto y su
    # vector es practicamente ruido.
    #
    # Se prueban despues de la etapa literal y antes de la semantica, en el
    # orden en que aparecen los intents en el YAML: lo especifico tiene que ir
    # declarado antes que lo generico. Sus grupos nombrados se usan como slots.
    #
    # Alto coste de equivocarse: un patron aqui se salta el juicio del encoder,
    # asi que solo deben declararse los que sean de precision practicamente
    # total. `tests/test_router.py` comprueba que ninguno reclama frases de
    # otro intent del catalogo.
    patterns: tuple[str, ...] = ()
    # Regex con grupos nombrados para extraer argumentos de la frase original.
    # Se prueban en orden; el primero que casa gana.
    slot_patterns: tuple[str, ...] = ()
    # Argumentos fijos, utiles para reusar una skill con distinta intencion
    # (p. ej. volume.step con delta=+10 y delta=-10 son dos intents).
    fixed_args: dict[str, str | int | float | bool] = Field(default_factory=dict)
    # Si es True y ningun slot_pattern casa, la frase escala al LLM en vez de
    # ejecutarse sin argumento.
    slots_required: bool = False
    # Frase corta que confirma por voz. Si es None, la skill decide.
    confirmation: str | None = None

    @model_validator(mode="after")
    def _check_tool_presence(self) -> Self:
        if self.fallback and self.tool is not None:
            raise ValueError("un intent con fallback=true no puede declarar 'tool'")
        if not self.fallback and self.tool is None:
            raise ValueError("falta 'tool' (obligatorio salvo en intents de fallback)")
        return self
