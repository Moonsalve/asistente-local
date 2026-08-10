"""Skill `app.close`: cerrar aplicaciones de la allowlist.

EL NOMBRE DEL PROCESO NO SE ADIVINA, SE BUSCA ENTRE LOS QUE CORREN
------------------------------------------------------------------
La version anterior mataba `<spec.process>.exe` y punto. Ese campo es opcional
y a menudo falso: las apps de la Store llegan con `process=None` y los .lnk del
menu Inicio con un nombre inventado a partir del titulo. El resultado era una
skill que fallaba en silencio o decia "no estaba abierto" cuando el problema
era otro.

Ahora se construye una lista de nombres plausibles y se cruza con los procesos
vivos. El nombre que se mata es uno que EXISTE, no uno que deberia existir.

La frontera de seguridad no cambia: todos los candidatos salen de la entrada de
la allowlist (su `process`, su comando, su clave y sus alias), nunca del texto
transcrito. Lo hablado solo elige QUE entrada de la allowlist se usa.
"""

from __future__ import annotations

import logging
from pathlib import PureWindowsPath

from pydantic import BaseModel, ConfigDict, Field

from asistente.config import AppSpec, Config
from asistente.skills.base import Skill, SkillResult
from asistente.skills.launcher import APPS_FOLDER
from asistente.skills.resolve import build_alias_index, resolve
from asistente.skills.winproc import (
    can_list_processes,
    kill_image,
    normalize_image,
    running_images,
)

log = logging.getLogger(__name__)


def candidate_images(app_name: str, spec: AppSpec) -> tuple[str, ...]:
    """Nombres de imagen plausibles, del mas fiable al mas especulativo.

    El orden importa porque el primero que este corriendo es el que se mata:

    1. `spec.process`, si esta declarado a mano. Es el unico dato escrito por
       una persona que sabia lo que hacia.
    2. El nombre del ejecutable del comando. Cubre `chrome`, las rutas
       absolutas y los .lnk, cuyo titulo suele coincidir con su .exe.
    3. La clave de la app y sus alias. Es lo que salva a las apps de la Store,
       que no traen ningun nombre de proceso pero corren como `<Nombre>.exe`.

    Los comandos que no son ejecutables —AppsFolder y URIs como
    `steam://rungameid/440`— no aportan nombre y se saltan: su `stem` seria
    basura ("rungameid").
    """
    crudos: list[str] = []
    if spec.process:
        crudos.append(spec.process)
    # `PureWindowsPath` y no `Path`: estos comandos son rutas de Windows se
    # ejecute esto donde se ejecute, y `Path` en macOS no ve la barra invertida
    # como separador, asi que se quedaria la ruta entera.
    es_ejecutable = not spec.command.startswith(APPS_FOLDER) and "://" not in spec.command
    if es_ejecutable and (stem := PureWindowsPath(spec.command).stem):
        crudos.append(stem)
    crudos.append(app_name)
    crudos.extend(spec.aliases)

    vistos: set[str] = set()
    unicos: list[str] = []
    for nombre in crudos:
        clave = normalize_image(nombre)
        if clave and clave not in vistos:
            vistos.add(clave)
            unicos.append(nombre)
    return tuple(unicos)


def pick_running(candidates: tuple[str, ...], running: tuple[str, ...]) -> str | None:
    """Primer candidato que este de verdad en marcha, con su nombre real.

    La comparacion es exacta salvo mayusculas y la extension. Nada de difuso:
    aqui un acierto de mas no abre una ventana equivocada, mata un proceso
    ajeno, y "Code" contra "Discord" o "Steam" contra "SteamService" estan lo
    bastante cerca como para que un umbral difuso acabe costando trabajo sin
    guardar.
    """
    vivos = {normalize_image(imagen): imagen for imagen in running}
    for candidato in candidates:
        if (real := vivos.get(normalize_image(candidato))) is not None:
            return real
    return None


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

        if not can_list_processes():
            return SkillResult.failed("Cerrar aplicaciones solo funciona en Windows.")

        spec = self._config.apps[app_name]
        candidatos = candidate_images(app_name, spec)
        objetivo = self._target(app_name, spec, candidatos)
        if objetivo is None:
            return SkillResult.failed(f"{app_name} no está abierto.")

        ok, motivo = kill_image(objetivo)
        if not ok:
            log.warning("no se pudo cerrar %s (%s.exe): %s", app_name, objetivo, motivo)
            return SkillResult.failed(f"No pude cerrar {app_name}.")
        log.info("cerrado %s -> %s", app_name, objetivo)
        return SkillResult.silent()

    def _target(self, app_name: str, spec: AppSpec, candidatos: tuple[str, ...]) -> str | None:
        """Que proceso matar, o None si no hay nada que matar.

        `running_images()` devuelve vacio tanto si `tasklist` falla como si no
        hay procesos, pero en Windows lo segundo es imposible: una tupla vacia
        significa siempre "no se pudo preguntar". En ese caso se sigue adelante
        con lo que dice la config —degradar al comportamiento anterior es mejor
        que negarse— y se deja constancia en el log.
        """
        if vivos := running_images():
            return pick_running(candidatos, vivos)

        log.warning("no se pudo listar los procesos; se cierra %s a ciegas", app_name)
        return spec.process or (candidatos[0] if candidatos else None)
