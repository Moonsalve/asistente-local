"""Etapa 1.5: intents que se reconocen por su forma, no por su significado.

POR QUE EXISTE ESTA ETAPA
-------------------------
La etapa semantica compara significados, y para eso necesita que la frase TENGA
significado que comparar. Hay una familia de comandos donde no lo hay:

    "reproduce loving machine de tv girl"

El titulo y el grupo son nombres propios que el encoder no ha visto nunca. Su
vector es practicamente ruido, y el coseno lo asigna a cualquier intent con un
score mediocre. MEDIDO sobre este catalogo:

    reproduce loving machine de tv girl   -> spotify.liked    0.461
    pon despacito de luis fonsi           -> media.play_pause 0.323
    pon blinding lights de the weeknd     -> spotify.liked    0.453

No es que falten anclas: es que no hay nada que anclar. Los titulos varian en
cada peticion, y lo unico constante de la frase es el MOLDE.

Y no se puede arreglar con un umbral. Medido: las frases secuestradas puntuan
entre 0.291 y 0.628, y las que el catalogo acierta, entre 0.290 y 0.68. Se
solapan enteras — el mismo hallazgo que hizo descartar el umbral absoluto de
coseno, un piso mas arriba.

Asi que la decision se mueve a donde SI es fiable: un regex sobre la sintaxis.

EL PELIGRO, Y COMO SE CONTIENE
------------------------------
Un patron aqui se salta el juicio del encoder, asi que uno demasiado goloso
secuestra comandos legitimos: "pon el volumen al 50" tambien empieza por "pon".
Tres cosas lo contienen:

1. El patron generico lleva un guardia negativo con las palabras que el
   asistente YA RECLAMA para si (volumen, pausa, siguiente, mis me gusta...).
   No es una lista de frases que el usuario deba aprender: es el vocabulario
   propio del asistente, y solo crece cuando un intent nuevo reclama una forma
   con "pon".
2. El orden de declaracion manda, asi que lo especifico va antes que lo
   generico y cada intent reclama sus propias formas en su propio bloque.
3. `tests/test_router.py` recorre TODOS los ejemplos del catalogo y comprueba
   que ningun patron reclama frases de otro intent. El guardia no puede quedarse
   obsoleto en silencio: si alguien anade "pon el modo repeticion" a otro
   intent, el test lo caza.

Lo que el regex NO decide es el significado. "pon X de Y" entrega una hipotesis
—titulo X, artista Y—; que eso sea de verdad una cancion lo decide Spotify al
buscarla. Es el mismo reparto que en `open.target`: la forma la reconoce el
router, el sentido lo resuelven los datos.
"""

from __future__ import annotations

from asistente.router.catalog import Catalog


def match_pattern(catalog: Catalog, normalized: str) -> tuple[str, dict[str, str]] | None:
    """`(intent, slots)` del primer patron que casa, o None.

    Los grupos nombrados del patron son los slots. Se devuelven aparte de
    `extract_slots` porque aqui el mismo regex hace las dos cosas: si ha sabido
    reconocer el molde, ya sabe donde empieza el titulo.
    """
    for pattern, intent in catalog.command_patterns:
        if (m := pattern.search(normalized)) is None:
            continue
        slots = {
            key: value.strip()
            for key, value in m.groupdict().items()
            if value and value.strip()
        }
        return intent, slots
    return None
