"""Skill `web.search`: buscar en internet.

Abre el navegador con la consulta en vez de scrapear resultados. Es deliberado:
scrapear un buscador es fragil, se rompe con cada cambio de maquetado, y para
"busca X" lo que quieres es la pagina, no que el asistente te la lea.

Para preguntas que SI quieres que te responda hablando ("cuanto mide la torre
Eiffel"), la ruta es otra: el catalogo las manda al LLM via `_fallback`.
"""

from __future__ import annotations

import webbrowser
from urllib.parse import quote_plus

from pydantic import BaseModel, ConfigDict, Field

from asistente.config import Config
from asistente.skills.base import Skill, SkillResult


class SearchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=300)


class WebSearchSkill(Skill):
    name = "web.search"
    args_model = SearchArgs
    description = "Busca algo en internet y abre los resultados en el navegador."

    def __init__(self, config: Config) -> None:
        self._search_url = config.search_url

    def execute(self, args: BaseModel) -> SkillResult:
        assert isinstance(args, SearchArgs)
        webbrowser.open(self._search_url.format(query=quote_plus(args.query)))
        return SkillResult.silent()
