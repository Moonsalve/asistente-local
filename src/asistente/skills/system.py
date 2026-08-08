"""Skills de sistema: volumen y hora.

El volumen usa la API de Windows (pycaw) cuando esta disponible porque permite
fijar un porcentaje exacto. Si falla, se recurre a las teclas multimedia, que
solo suben o bajan en pasos fijos pero nunca dejan el comando sin efecto.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from asistente.numbers import parse_spanish_number
from asistente.skills.base import NoArgs, Skill, SkillResult
from asistente.skills.winkeys import VirtualKey, get_volume_percent, press, set_volume_percent

#: Cada pulsacion de VOLUME_UP/DOWN mueve el volumen de Windows un 2%.
_KEY_STEP_PERCENT = 2


class VolumeStepArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    #: Positivo sube, negativo baja. Viene de `fixed_args` en el catalogo.
    delta: int = Field(ge=-100, le=100)


class VolumeStepSkill(Skill):
    name = "system.volume_step"
    args_model = VolumeStepArgs
    description = "Sube o baja el volumen del sistema una cantidad relativa."

    def execute(self, args: BaseModel) -> SkillResult:
        assert isinstance(args, VolumeStepArgs)

        current = get_volume_percent()
        if current is not None and set_volume_percent(current + args.delta):
            return SkillResult.silent()

        # Sin pycaw: aproximar con pulsaciones. Se redondea al alza para que
        # "sube el volumen" siempre haga algo perceptible.
        key = VirtualKey.VOLUME_UP if args.delta > 0 else VirtualKey.VOLUME_DOWN
        presses = max(1, round(abs(args.delta) / _KEY_STEP_PERCENT))
        if not all(press(key) for _ in range(presses)):
            return SkillResult.failed("No pude cambiar el volumen.")
        return SkillResult.silent()


class VolumeSetArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    #: Llega como texto porque el router extrae la frase cruda: puede ser "40"
    #: o "cuarenta" segun como lo transcriba Whisper. La conversion es
    #: responsabilidad de la skill, no del router.
    level: str = Field(min_length=1, max_length=40)


class VolumeSetSkill(Skill):
    name = "system.volume_set"
    args_model = VolumeSetArgs
    description = "Fija el volumen del sistema a un porcentaje concreto (0-100)."

    def execute(self, args: BaseModel) -> SkillResult:
        assert isinstance(args, VolumeSetArgs)

        level = parse_spanish_number(args.level)
        if level is None:
            return SkillResult.failed("No entendí a qué volumen lo pongo.")
        if not set_volume_percent(level):
            return SkillResult.failed("No pude cambiar el volumen.")
        return SkillResult.silent()


class MuteSkill(Skill):
    name = "system.mute"
    args_model = NoArgs
    description = "Silencia o quita el silencio del sistema."

    def execute(self, args: BaseModel) -> SkillResult:
        if not press(VirtualKey.VOLUME_MUTE):
            return SkillResult.failed("No pude silenciar el sonido.")
        return SkillResult.silent()


class TimeSkill(Skill):
    name = "system.time"
    args_model = NoArgs
    description = "Dice la hora actual."

    def execute(self, args: BaseModel) -> SkillResult:
        now = datetime.now()
        # Formato hablado: "las tres y veinte" se lee peor que "3:20" en Piper,
        # que expande los digitos correctamente en espanol.
        return SkillResult.says(f"Son las {now.hour}:{now.minute:02d}.")


#: Nombres en espanol escritos a mano. `locale` en Windows es poco fiable: el
#: idioma del sistema puede no ser espanol, y `setlocale` es global al proceso,
#: asi que tocarlo afectaria a todo lo demas.
_DIAS = ("lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo")
_MESES = (
    "enero", "febrero", "marzo", "abril", "mayo", "junio",
    "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre",
)


class DateSkill(Skill):
    name = "system.date"
    args_model = NoArgs
    description = "Dice la fecha de hoy: dia de la semana, dia del mes y mes."

    def execute(self, args: BaseModel) -> SkillResult:
        now = datetime.now()
        return SkillResult.says(
            f"Hoy es {_DIAS[now.weekday()]} {now.day} de {_MESES[now.month - 1]}."
        )
