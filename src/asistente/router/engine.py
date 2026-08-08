"""Orquestador del router de tres etapas.

Cada etapa es mas lenta y mas flexible que la anterior, y solo se ejecuta si la
anterior no resolvio:

    1. literal   (< 0.001 s)  frases exactas del catalogo
    2. semantica (0.01-0.02 s) coseno contra los ejemplos del catalogo
    3. LLM       (0.25-0.50 s) lo genuinamente nuevo o compuesto

La metrica que gobierna la latencia media del sistema es el porcentaje de
frases que llegan a la etapa 3. `RouteResult.stage` existe para poder medirlo:
si sube del 15%, al catalogo le faltan ejemplos.
"""

from __future__ import annotations

import logging
from time import perf_counter
from typing import Protocol

from asistente.router.catalog import Catalog
from asistente.router.literal import match_literal
from asistente.router.schema import RouteResult, SpeechReply, Stage, ToolCall
from asistente.router.semantic import SemanticMatcher, extract_slots
from asistente.router.text import normalize

log = logging.getLogger(__name__)


class LlmFallback(Protocol):
    """Etapa 3. Se inyecta para que el router sea testeable sin Ollama."""

    def route(self, text: str) -> ToolCall | SpeechReply | None: ...


class Router:
    def __init__(
        self,
        catalog: Catalog,
        matcher: SemanticMatcher,
        llm: LlmFallback | None = None,
    ) -> None:
        self._catalog = catalog
        self._matcher = matcher
        self._llm = llm

    def route(self, text: str) -> RouteResult:
        started = perf_counter()
        normalized = normalize(text)

        if (intent := match_literal(self._catalog, normalized)) is not None:
            call = self._build_call(intent, slots={})
            return RouteResult(
                stage=Stage.LITERAL,
                tool_call=call,
                score=1.0,
                latency_s=perf_counter() - started,
            )

        if (match := self._matcher.match(normalized)) is not None:
            spec = self._catalog.intents[match.intent]
            slots = extract_slots(self._catalog, match.intent, normalized)

            if slots is None and spec.slots_required:
                # El intent esta claro pero falta el argumento: solo este caso
                # concreto paga el LLM, no toda la categoria de comandos.
                log.debug(
                    "intent %s reconocido (%.3f) pero sin slots; escalando al LLM",
                    match.intent,
                    match.score,
                )
            else:
                return RouteResult(
                    stage=Stage.SEMANTIC,
                    tool_call=self._build_call(match.intent, slots or {}),
                    score=match.score,
                    latency_s=perf_counter() - started,
                )

        return self._route_with_llm(text, started)

    def _route_with_llm(self, text: str, started: float) -> RouteResult:
        if self._llm is None:
            return RouteResult(
                stage=Stage.LLM, score=0.0, latency_s=perf_counter() - started
            )

        outcome = self._llm.route(text)
        return RouteResult(
            stage=Stage.LLM,
            tool_call=outcome if isinstance(outcome, ToolCall) else None,
            reply=outcome if isinstance(outcome, SpeechReply) else None,
            score=0.0,
            latency_s=perf_counter() - started,
        )

    def _build_call(self, intent: str, slots: dict[str, str]) -> ToolCall:
        spec = self._catalog.intents[intent]
        # Garantizado por el validador de IntentSpec: solo los intents de
        # fallback pueden no declarar tool, y esos nunca llegan hasta aqui.
        assert spec.tool is not None, f"intent '{intent}' sin tool"
        # Los slots extraidos de la frase pisan a los fijos: si el catalogo
        # declara un valor por defecto y el usuario dice otro, manda el usuario.
        args: dict[str, str | int | float | bool] = {**spec.fixed_args, **slots}
        return ToolCall(name=spec.tool, args=args)
