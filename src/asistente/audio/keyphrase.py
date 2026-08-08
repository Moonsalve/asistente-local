"""Deteccion de la palabra clave sobre la transcripcion.

POR QUE EXISTE
--------------
openWakeWord solo reconoce palabras para las que hay un modelo entrenado, y los
preentrenados son todos ingleses ("hey jarvis", "alexa"...). Para una palabra
propia en espanol -"Apolo"- hay dos caminos:

  1. Entrenar un modelo de openWakeWord. Es lo eficiente (1-2% de CPU en
     reposo) pero cuesta ~1 h de entrenamiento y hay que rehacerlo si cambias
     de palabra.
  2. Este modulo: transcribir lo que se dice y mirar si empieza por la palabra
     clave. Funciona con CUALQUIER frase en espanol al instante y sin entrenar.

TRADE-OFF, que no es menor: en modo transcripcion Whisper corre CADA VEZ que
alguien habla cerca del microfono, no solo cuando le hablas al asistente. Con
GPU son ~150 ms por frase y es asumible; en CPU es pesado. El wake word
entrenado sigue siendo la opcion buena a largo plazo.

VENTAJA INESPERADA: al transcribir la frase entera de una vez, "Apolo, pon
musica" se resuelve con UNA sola transcripcion, mientras que el wake word
clasico obliga a detectar y luego grabar la orden por separado. Cuando hablas
del tiron, este modo es en realidad mas rapido.
"""

from __future__ import annotations

import logging

from rapidfuzz import fuzz

from asistente.router.text import normalize

log = logging.getLogger(__name__)

#: Similitud minima (0-100) para dar por dicha la palabra clave. Alto: un falso
#: positivo aqui ejecuta una accion que no pediste.
DEFAULT_THRESHOLD = 82.0

#: Palabras de cortesia que suelen colarse entre la clave y la orden.
_FILLERS = frozenset({"por", "favor", "oye", "eh", "y", "ok", "vale"})


class KeyphraseGate:
    """Decide si una transcripcion va dirigida al asistente."""

    def __init__(self, phrases: tuple[str, ...], threshold: float = DEFAULT_THRESHOLD) -> None:
        if not phrases:
            raise ValueError("hace falta al menos una palabra clave")
        self._phrases = tuple(normalize(p) for p in phrases)
        # Cuantas palabras ocupa la clave mas larga: es lo maximo que hay que
        # mirar al principio de la frase.
        self._max_words = max(len(p.split()) for p in self._phrases)
        self._threshold = threshold

    def match(self, text: str) -> str | None:
        """Devuelve la orden sin la palabra clave, o None si no va dirigida a ti.

        Cadena vacia significa "dijo solo la palabra clave": el asistente debe
        quedarse escuchando la orden que viene despues.
        """
        normalized = normalize(text)
        if not normalized:
            return None

        words = normalized.split()
        # Se prueba con 1, 2... n palabras iniciales porque Whisper a veces
        # parte la clave ("a polo") y a veces la pega a la siguiente palabra.
        for take in range(1, min(self._max_words, len(words)) + 1):
            candidate = " ".join(words[:take])
            for phrase in self._phrases:
                score = fuzz.ratio(candidate, phrase)
                if score >= self._threshold:
                    log.debug("clave %r reconocida en %r (%.0f)", phrase, candidate, score)
                    rest = words[take:]
                    # "Apolo, por favor, pon musica" -> "pon musica"
                    while rest and rest[0] in _FILLERS:
                        rest.pop(0)
                    return " ".join(rest)
        return None
