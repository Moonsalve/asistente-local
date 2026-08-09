"""Skills de sistema: hora y fecha.

El volumen vivia aqui hasta que dejo de ser solo "del sistema": ahora tambien
apunta a Spotify y tiene su propio modulo, `skills/volume.py`.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel

from asistente.skills.base import NoArgs, Skill, SkillResult


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
