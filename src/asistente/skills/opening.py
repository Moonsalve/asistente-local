"""Skill `open.target`: abrir aplicaciones y paginas web.

POR QUE ES UNA SOLA SKILL Y NO DOS
---------------------------------
"abre spotify" (app) y "abre youtube" (web) son la misma intencion en la cabeza
del usuario. Separarlas en dos intents fallaba de forma medible: "abreme el
chrome" ganaba `web.open` con 0.942 frente a `app.open` con 0.922, porque los
embeddings no tienen forma de saber si "chrome" es un programa o un sitio.

La ambiguedad no es semantica, es de resolucion. Asi que se decide aqui, con
datos en vez de con similitud:

    allowlist de apps  ->  mapa de sitios  ->  busqueda web

El ultimo escalon garantiza que el comando nunca se queda sin efecto: si dices
"abre lo que sea", acabas con una busqueda en vez de con un silencio.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, ConfigDict, Field

from asistente.config import Config
from asistente.router.text import normalize
from asistente.skills.base import Skill, SkillResult
from asistente.skills.browser import Browser
from asistente.skills.launcher import launch
from asistente.skills.resolve import build_alias_index, resolve

log = logging.getLogger(__name__)


class OpenArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target: str = Field(min_length=1, max_length=120)


class OpenTargetSkill(Skill):
    name = "open.target"
    args_model = OpenArgs
    description = (
        "Abre una aplicacion instalada o una pagina web. Usar para 'abre X', "
        "'entra a X', 'llevame a X'."
    )

    def __init__(self, config: Config, browser: Browser) -> None:
        self._config = config
        self._browser = browser
        self._app_index = build_alias_index(config.apps)
        self._site_index = {normalize(alias): url for alias, url in config.sites.items()}

    def execute(self, args: BaseModel) -> SkillResult:
        assert isinstance(args, OpenArgs)
        target = args.target

        if (app_name := resolve(target, self._app_index)) is not None:
            return self._launch_app(app_name)

        if (url := self._site_index.get(normalize(target))) is not None:
            self._browser.open(url)
            return SkillResult.silent()

        # Ni app conocida ni sitio conocido: buscarlo es mas util que no hacer nada.
        query = self._config.search_url.format(query=_url_quote(target))
        self._browser.open(query)
        return SkillResult.says(f"No conozco {target}, te lo busco.")

    def _launch_app(self, app_name: str) -> SkillResult:
        spec = self._config.apps[app_name]
        if not launch(spec.command):
            return SkillResult.failed(f"No pude abrir {app_name}.")
        return SkillResult.silent()


def _url_quote(text: str) -> str:
    from urllib.parse import quote_plus

    return quote_plus(text)
