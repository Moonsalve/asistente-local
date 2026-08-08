"""Etapa 3 del router: fallback con LLM local via Ollama.

Solo entra cuando las dos etapas anteriores no resuelven. Es la ruta cara
(0.25-0.50 s) y por eso el diseno entero gira alrededor de no necesitarla.

DECISIONES
----------
**Structured output, no texto libre.** Se pasa un JSON Schema en `format`, asi
que Ollama restringe el decodificado y el modelo no puede devolver prosa donde
esperabamos una accion. Un 3B sin esta restriccion produce JSON malformado con
frecuencia suficiente para ser un problema.

**El modelo elige un nombre de una lista cerrada.** El schema declara `name` como
enum con las skills registradas. Aun asi el registro revalida: `format` guia el
decodificado, no lo garantiza formalmente.

**Nunca genera codigo.** No hay una skill que ejecute shell, ni rutas de archivo
libres. Lo peor que puede hacer un modelo confundido es lanzar una accion que ya
estaba autorizada.

**temperature=0.** Esto es clasificacion, no escritura creativa: queremos que la
misma frase produzca siempre la misma accion.
"""

from __future__ import annotations

import json
import logging
from time import perf_counter
from typing import Any

from pydantic import ValidationError

from asistente.config import LlmConfig
from asistente.router.schema import SpeechReply, ToolCall

log = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
Eres el cerebro de un asistente de voz local que corre en un PC con Windows.
Recibes la transcripcion de lo que el usuario acaba de decir en espanol.

Decide una de dos cosas:
- Si pide una ACCION que puedas ejecutar con las herramientas disponibles,
  responde con kind="tool", el nombre de la herramienta y sus argumentos.
- Si es una pregunta o conversacion, responde con kind="speech" y un texto
  BREVE (una o dos frases). Se va a leer en voz alta: nada de listas, markdown
  ni enlaces.

Herramientas disponibles:
{tools}

Responde solo con el JSON pedido.\
"""

# Few-shot corto y en espanol. Con un 3B los ejemplos pesan mas que las
# instrucciones: dos de cada tipo bastan y mantienen el prompt barato.
_EXAMPLES: list[dict[str, str]] = [
    {"role": "user", "content": "ponme algo de musica tranquila"},
    {
        "role": "assistant",
        "content": '{"kind":"tool","name":"spotify.play","args":{"query":"musica tranquila"},"text":""}',
    },
    {"role": "user", "content": "cuantos habitantes tiene colombia"},
    {
        "role": "assistant",
        "content": '{"kind":"speech","name":"","args":{},"text":"Colombia tiene unos 52 millones de habitantes."}',
    },
]


def _describe(exc: Exception) -> str:
    """Mensaje corto y accionable para fallos de servicio."""
    name = type(exc).__name__
    if "Timeout" in name:
        return "timeout (Ollama tarda demasiado; puede estar cargando el modelo)"
    if "Connect" in name:
        return "no se pudo conectar (¿esta Ollama arrancado? `ollama serve`)"
    return f"{name}: {exc}".strip().splitlines()[0][:200]


def _response_schema(tool_names: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "enum": ["tool", "speech"]},
            # Enum cerrado: el decodificado restringido no puede inventar una
            # herramienta que no exista.
            "name": {"type": "string", "enum": [*tool_names, ""]},
            "args": {"type": "object"},
            "text": {"type": "string"},
        },
        "required": ["kind", "name", "args", "text"],
    }


class OllamaFallback:
    def __init__(self, config: LlmConfig, tool_descriptions: list[dict[str, object]]) -> None:
        from ollama import Client

        self._config = config
        self._client = Client(host=config.host, timeout=config.timeout_s)
        self._tool_names = [str(t["name"]) for t in tool_descriptions]
        self._schema = _response_schema(self._tool_names)
        self._system = _SYSTEM_PROMPT.format(
            tools="\n".join(
                f"- {t['name']}: {t['description']} args={list(t['args'])}"  # type: ignore[arg-type]
                for t in tool_descriptions
            )
        )

    def warmup(self) -> None:
        """Carga el modelo en VRAM antes del primer comando real.

        Usa un timeout MUCHO mas largo que una consulta normal: esta peticion
        incluye leer varios GB de disco y subirlos a la VRAM, que en frio pasa
        del minuto. Con el timeout de runtime (10 s) el calentamiento fallaba
        siempre, y entonces el primer comando real pagaba la carga completa.
        """
        from ollama import Client

        started = perf_counter()
        client = Client(host=self._config.host, timeout=self._config.warmup_timeout_s)
        try:
            self._chat(client, "hola")
        except Exception as exc:
            log.warning(
                "no se pudo calentar el LLM (%s); el primer comando ira lento",
                _describe(exc),
            )
            return
        log.info("LLM calentado en %.1f s", perf_counter() - started)

    def route(self, text: str) -> ToolCall | SpeechReply | None:
        try:
            payload = self._chat(self._client, text)
        except Exception as exc:
            # Sin traceback: los fallos aqui son de servicio (timeout, Ollama
            # caido, modelo sin descargar), no bugs. El traceback de httpx
            # ocupa 40 lineas y no dice nada util.
            log.warning("fallo la consulta al LLM: %s", _describe(exc))
            return None

        return self._to_outcome(payload)

    def _chat(self, client: object, text: str) -> dict[str, Any]:
        response = client.chat(  # type: ignore[attr-defined]
            model=self._config.model,
            messages=[
                {"role": "system", "content": self._system},
                *_EXAMPLES,
                {"role": "user", "content": text},
            ],
            format=self._schema,
            keep_alive=self._config.keep_alive,
            options={"temperature": self._config.temperature},
        )
        return dict(json.loads(response["message"]["content"]))

    def _to_outcome(self, payload: dict[str, Any]) -> ToolCall | SpeechReply | None:
        if payload.get("kind") == "speech":
            text = str(payload.get("text", "")).strip()
            return SpeechReply(text=text) if text else None

        name = str(payload.get("name", ""))
        if name not in self._tool_names:
            # Ocurre pese al enum: `format` guia el decodificado pero no es una
            # garantia formal. El registro volveria a rechazarlo de todos modos.
            log.warning("el LLM devolvio una herramienta desconocida: %r", name)
            return None

        try:
            return ToolCall(name=name, args=payload.get("args") or {})
        except ValidationError:
            log.warning("argumentos invalidos del LLM para %s: %r", name, payload.get("args"))
            return None
