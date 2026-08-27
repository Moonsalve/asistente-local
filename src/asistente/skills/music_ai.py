"""Etapa 3 de la busqueda de musica: el LLM local corrige el nombre.

EL PROBLEMA
-----------
Whisper escribe los titulos en ingles como suenan en espanol: "lovin machin de
tibi guerl" por "Loving Machine de TV Girl". Con ese texto la Web API de Spotify
no encuentra nada, porque busca letras y no fonetica.

Lo que habia era emparejar contra TUS me gusta, y funciona — pero solo para lo
que ya tienes. Pedir algo que no esta en tu biblioteca caia a una busqueda con
el texto destrozado y no aparecia nada. Era una red, no una solucion.

Aqui el LLM hace lo unico que ningun emparejamiento difuso puede hacer:
reconocer el nombre. "Tibi guerl" solo se convierte en "TV Girl" si sabes que
TV Girl existe. Eso es conocimiento del mundo, no distancia de edicion.

EL REPARTO
----------
    Router      ¿que quiere?              catalogo — 0 ms, determinista
    Resolucion  ¿como se escribe eso?     ESTE MODULO — conocimiento del mundo
    Ejecucion   ¿existe de verdad?        Spotify + verificacion de cobertura

El catalogo de comandos NO pasa por aqui. Para "sube el volumen" o "pausa" el
catalogo cuesta 0-2 ms, es determinista y no puede alucinar una accion; un LLM
son 250-900 ms por comando y la posibilidad de inventarse una herramienta. El
LLM entra solo donde aporta algo que el catalogo no tiene: los nombres propios.

LA CASCADA POR COSTE, QUE ES EL DISENO ENTERO
---------------------------------------------
El LLM NO se llama en cada peticion de musica. Solo cuando la via barata ya
fallo:

    1. Busqueda directa en Spotify + verificacion de cobertura    0 ms extra
    2. Emparejamiento contra tus me gusta (local)                 0 ms
    3. El LLM corrige {titulo, artista} y se vuelve a buscar    600-900 ms

Es el mismo reparto que el router (literal -> patrones -> semantico -> LLM) y
que las defensas contra el ruido: lo caro solo lo paga quien lo necesita. Sin
esto, cada "pon musica" que hoy ya funciona costaria casi un segundo de mas para
nada.

CORREGIR NO ES INVENTAR
-----------------------
La trampa de todo esto, y la razon de que exista `_plausible`. Si el modelo no
conoce la cancion, no dice "no la conozco": **produce un titulo plausible y
distinto**, con la misma confianza. Y eso es peor que no encontrar nada, porque
suena algo y parece que funciono. Es exactamente el mismo argumento por el que
el volumen de Spotify no degrada al mezclador de Windows.

Dos defensas, y la segunda es la que importa:

1. **El prompt se lo pide.** "Si no reconoces la cancion, reconstruye
   literalmente en vez de inventarte otra." Ayuda; no basta.

2. **Se mide cuanto se ha alejado de lo que se OYO.** Whisper destroza la
   ortografia pero conserva el sonido, asi que una correccion de verdad se
   queda cerca del texto oido ("lovin machin" -> "loving machine"). Una
   invencion se va lejos ("loving machine" -> "love machine de the miracles").
   Por debajo del umbral, la correccion se tira y el comando falla como habria
   fallado de todos modos.

   El umbral no es un numero inventado: es el mismo `LIBRARY_THRESHOLD` de
   emparejar contra tu biblioteca, porque es LITERALMENTE la misma comparacion
   —transcripcion destrozada contra titulo bien escrito— y ya esta medida sobre
   transcripciones reales: el peor acierto puntua 91.9 y el mejor fallo, 65.0.

La asimetria es deliberada. Rechazar una correccion buena cuesta un "no lo
encontre", que es lo que habria pasado sin este modulo. Aceptar una mala pone
una cancion que nadie pidio. En este sistema fallar siempre es mejor que mentir.

Y queda una tercera red que no esta aqui: el resultado corregido vuelve a pasar
por `play_query`, que verifica la COBERTURA del resultado de Spotify contra el
texto corregido. Un titulo inventado que ademas Spotify no encuentre no llega a
sonar.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from asistente.config import LlmConfig
from asistente.router.text import normalize
from asistente.skills.spotify import LIBRARY_THRESHOLD, similitud

log = logging.getLogger(__name__)

#: Cuanto puede alejarse la correccion del texto que se oyo (0-100).
#:
#: Vale lo mismo que `LIBRARY_THRESHOLD` porque es la misma comparacion —texto
#: transcrito de oido contra titulo bien escrito— y alli ya estaba medida sobre
#: transcripciones reales: peor acierto 91.9, mejor fallo 65.0.
#:
#: COMPROBADO ADEMAS PARA ESTE USO, con `similitud` sobre destrozos reales de
#: Whisper y sobre invenciones del estilo que produce un modelo cuando no
#: conoce la cancion:
#:
#:     correcciones de verdad      73.9 - 92.0   (la peor: "blain ding lights
#:                                                de uiquen" -> Blinding Lights
#:                                                de The Weeknd, 73.9)
#:     invenciones                 35.9 - 63.4   (la mejor: "loving machine de
#:                                                tv gery" -> Love Machine de
#:                                                The Miracles, 63.4)
#:
#: El hueco es de 10.5 puntos y 72 cae dentro, mas cerca del lado bueno. Si
#: empieza a rechazar correcciones validas, bajarlo hacia 66 es lo primero que
#: probar —y sabiendo lo que se compra: cada punto que baja acerca el umbral a
#: las invenciones—. `test_the_threshold_sits_between_the_two_measurements` fija
#: el hueco para que nadie lo mueva a ciegas.
#:
#: Se declara aparte y no se usa `LIBRARY_THRESHOLD` directamente para que se
#: puedan afinar por separado: el espacio de busqueda de alli son unos cientos
#: de canciones tuyas, y el de aqui, todo lo que el modelo se pueda imaginar.
PLAUSIBLE_THRESHOLD = LIBRARY_THRESHOLD

#: El prompt de `scripts/probe_music_ai.py`, PALABRA POR PALABRA.
#:
#: No se toca sin volver a correr la sonda. La pregunta que decidio este cambio
#: —¿reconoce el modelo los grupos que escuchas, o se los inventa?— se respondio
#: con este texto exacto. Cambiar una linea invalida esa medida.
_PROMPT = """Eres un corrector de nombres de canciones. Recibes una peticion de \
musica transcrita por un sistema de voz en espanol, que escribe los nombres en \
ingles como suenan y comete errores.

Devuelve SOLO un objeto JSON con dos claves:
  "titulo":  el nombre real de la cancion, escrito correctamente
  "artista": el nombre real del grupo o cantante, o "" si no se dice

Si no reconoces la cancion, escribe tu mejor reconstruccion literal en vez de \
inventarte otra distinta. No expliques nada.

Peticion: {frase}"""

#: Decodificado restringido, igual que en el router. La sonda uso `format="json"`
#: a secas; el esquema es MAS estricto y solo fija la FORMA (dos claves de
#: texto), no el contenido, asi que lo que la sonda midio —si el modelo conoce o
#: no el grupo— se mantiene igual. Lo que desaparece es la otra mitad de los
#: fallos posibles: JSON con una clave de mas, o con el titulo dentro de un
#: objeto anidado.
_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "titulo": {"type": "string"},
        "artista": {"type": "string"},
    },
    "required": ["titulo", "artista"],
}


class MusicQuery(BaseModel):
    """Lo que el LLM tiene permitido devolver.

    `extra="forbid"` por la misma razon que en `ToolCall`: el modelo emite datos
    validados contra un modelo cerrado, nunca campos libres que alguien aguas
    abajo pueda interpretar.
    """

    model_config = ConfigDict(extra="forbid")

    titulo: str = Field(min_length=1, max_length=200)
    artista: str = Field(default="", max_length=120)

    @property
    def pedido(self) -> str:
        """Titulo y artista juntos, como se compararian con lo que se oyo."""
        return f"{self.titulo} {self.artista}".strip()


def _describe(exc: Exception) -> str:
    """Mensaje corto para fallos de servicio (mismo criterio que `router/llm`)."""
    name = type(exc).__name__
    if "Timeout" in name:
        return "timeout (el modelo tarda mas de lo que aguanta un comando de voz)"
    if "Connect" in name:
        return "no se pudo conectar (¿esta Ollama arrancado?)"
    return f"{name}: {exc}".strip().splitlines()[0][:200]


class MusicResolver:
    """Convierte lo que se oyo en un titulo y un artista que existan.

    No habla con Spotify: devuelve texto corregido y quien lo pidio decide que
    hacer con el. Mantenerlo asi es lo que permite probarlo sin red, y lo que
    deja `SpotifyClient` como lo que debe seguir siendo, un envoltorio fino de
    la Web API.
    """

    def __init__(self, config: LlmConfig) -> None:
        from ollama import Client

        self._config = config
        self._client = Client(host=config.host, timeout=config.timeout_s)

    def resolve(self, query: str, artist: str | None = None) -> MusicQuery | None:
        """La correccion en la que se puede confiar, o None.

        None significa las tres cosas que acaban en el mismo sitio —no hubo
        respuesta, no era valida, o no era creible— porque para quien llama son
        la misma: no hay nada nuevo que buscar.
        """
        oido = f"{query} de {artist}" if artist else query
        payload = self._ask(oido)
        if payload is None:
            return None

        try:
            corregido = MusicQuery.model_validate(payload)
        except ValidationError:
            log.warning("el LLM devolvio una correccion invalida: %r", payload)
            return None

        if not self._plausible(query, artist, corregido):
            return None
        return corregido

    def _ask(self, oido: str) -> dict[str, Any] | None:
        try:
            response = self._client.chat(
                model=self._config.model,
                messages=[{"role": "user", "content": _PROMPT.format(frase=oido)}],
                format=_SCHEMA,
                keep_alive=self._config.keep_alive,
                options={"temperature": self._config.temperature},
            )
            return dict(json.loads(response["message"]["content"]))
        except Exception as exc:
            # Se atrapa `Exception` a proposito: lo que puede fallar aqui es el
            # servicio (Ollama caido, timeout, modelo sin descargar), y ninguno
            # de esos debe tumbar un comando de voz.
            #
            # Sin traceback, igual que en el router: aqui los fallos son de
            # servicio (Ollama caido, timeout) y el traceback de httpx ocupa 40
            # lineas que no dicen nada util.
            log.warning("no se pudo corregir %r con el LLM: %s", oido, _describe(exc))
            return None

    @staticmethod
    def _plausible(query: str, artist: str | None, corregido: MusicQuery) -> bool:
        """¿Corrigio lo que se oyo, o se invento otra cancion?

        Se comparan los dos lados completos cuando la frase dijo el artista, y
        solo los titulos cuando no lo dijo. La distincion importa: si nadie dijo
        el grupo, el que el modelo anade es informacion NUEVA y no hay nada oido
        contra lo que contrastarla. Metiendola igualmente en la comparacion,
        "pon creep" -> "Creep de Radiohead" puntuaria 53 y se rechazaria una
        correccion perfecta solo porque el nombre del grupo es mas largo que el
        titulo.
        """
        if artist:
            oido, propuesto = f"{query} {artist}", corregido.pedido
        else:
            oido, propuesto = query, corregido.titulo

        score = similitud(normalize(oido), normalize(propuesto))
        if score < PLAUSIBLE_THRESHOLD:
            # INFO y no DEBUG a proposito: esta linea es la sonda funcionando
            # sola. Si aparece a menudo, el modelo no conoce lo que escuchas y
            # esta rellenando huecos, que es justo lo que habia que vigilar.
            log.info(
                "correccion descartada por inverosimil: %r -> %r (%.0f < %.0f)",
                oido, propuesto, score, PLAUSIBLE_THRESHOLD,
            )
            return False

        log.info("el LLM corrige %r -> %r (%.0f)", oido, propuesto, score)
        return True


def build_resolver(config: LlmConfig) -> MusicResolver | None:
    """El resolvedor, o None si no se puede construir.

    Degradar en silencio es correcto AQUI y no lo seria en el router: sin esto
    la busqueda de musica sigue funcionando exactamente como funcionaba antes,
    asi que un Ollama ausente no puede impedir que el asistente arranque.
    """
    try:
        return MusicResolver(config)
    except Exception as exc:
        # Incluye el `ImportError` de no tener el paquete `ollama` instalado,
        # que es el caso de la Mac de desarrollo.
        log.warning("sin correccion de titulos con LLM: %s", _describe(exc))
        return None
