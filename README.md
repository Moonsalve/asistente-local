# Asistente local de voz

Asistente de voz 100% local para Windows. En español, activado por palabra
clave, sin enviar audio a ninguna nube. Abre aplicaciones, controla Spotify,
abre páginas web y busca en internet.

**Latencia objetivo:** 0.45–0.82 s desde que dejas de hablar hasta que suena la
respuesta, en la ruta que cubre la mayoría de comandos.

---

## Estado

| Fase | Qué es | Estado |
|---|---|---|
| 0 | Andamiaje, configuración, contratos tipados | Hecha |
| 1 | Audio: wake word, VAD, STT | **En marcha en el PC** (STT en CUDA, 0.12 s) |
| 2 | Router de 4 etapas + skills | **Hecha y verificada** (337 tests) |
| 3 | Spotify OAuth + control de sistema | **En marcha en el PC** |
| 4 | Fallback con LLM (Ollama) | En marcha; falta medir cuánto se usa |
| 4.5 | El LLM corrige el nombre de la canción | Escrito y con tests; **sin verificar en el PC** |
| 5 | Benchmark y tuning | Pendiente |

Sin verificar todavía en Windows: el autodescubrimiento de aplicaciones, los
juegos de Steam, Brave como navegador y la corrección de títulos con LLM.

El router y las skills se prueban en macOS; todo lo que toca micrófono, CUDA,
`pycaw` o `SendInput` requiere el PC Windows.

---

## Puesta en marcha (PC Windows)

> **Activa el venv en CADA terminal nueva.** Es el fallo número uno: instalas
> las dependencias en `.venv`, abres otra ventana, y `python -m asistente` corre
> con el Python del sistema y no las encuentra. El asistente lo detecta al
> arrancar y te avisa, pero es mejor no tropezar.

```powershell
# 1. Entorno  (el activate no es opcional)
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
pip install -e ".[gpu]"            # CUDA: onnxruntime-gpu + cuBLAS + cuDNN

# 2. Modelo del LLM (uno solo: router y corrección de títulos)
ollama pull qwen2.5:7b-instruct-q4_K_M

# 3. Voz de Piper (descarga el .onnx y su .json, que van siempre juntos)
python scripts/download_voice.py
python scripts/download_voice.py --list    # ver otras voces en español

# 4. Credenciales de Spotify
copy .env.example .env             # y rellena SPOTIFY_CLIENT_ID

# 5. Arrancar
python -m asistente
```

La primera ejecución descarga el encoder de embeddings (~470 MB) y el modelo de
Whisper (~1.6 GB). A partir de ahí quedan en caché.

Al arrancar se comprueba en un segundo que estén el intérprete correcto, la voz
de Piper, las librerías CUDA y Ollama con su modelo. Los problemas se reportan
todos a la vez, antes de cargar nada pesado.

### Modo texto (para desarrollar)

Escribes los comandos en vez de hablarlos. Salta micrófono, STT y TTS, así que
es lo único que corre fuera de Windows y es la forma rápida de iterar el
catálogo:

```bash
python -m asistente --text --no-llm
```

---

## Cómo se añade un comando

Casi siempre es una línea en `commands.yaml`, sin tocar código:

```yaml
media.next:
  examples:
    - pasa la cancion
    - esta no me gusta
    - tu nueva forma de decirlo aqui   # <- esto
```

Reinicia y ya. Si un comando escala al LLM (lo verás en el log como
`stage=llm`), esa frase es exactamente la que falta en `examples`.

Para una acción **nueva**: crea la skill en `src/asistente/skills/`, regístrala
en `factory.py` y declara el intent en `commands.yaml`. El arranque verifica que
cada `tool` del YAML tiene skill registrada, así que un typo se detecta al
instante en vez de convertirse en un comando que no hace nada.

---

## Tests

```bash
PYTHONPATH=src pytest -q                           # 337 tests, ~3 s
PYTHONPATH=src python scripts/diagnose_router.py   # scores del router frase a frase
PYTHONPATH=src python scripts/diagnose_apps.py     # qué apps se abren y cuáles se cierran
PYTHONPATH=src python scripts/diagnose_spotify.py  # dispositivos, tus playlists, me gusta
PYTHONPATH=src python scripts/diagnose_volume.py   # qué mecanismo de volumen falla
PYTHONPATH=src python scripts/diagnose_noise.py    # ruido de la sala y umbrales
PYTHONPATH=src python scripts/diagnose_music_ai.py # qué corrige el LLM y qué descarta
```

`tests/test_router.py` mide el router contra frases que **no** están en
`commands.yaml`: probar con las del propio catálogo no demostraría nada.

---

## Aplicaciones: autodescubrimiento

**Es automático.** Al arrancar, el asistente enumera lo instalado —apps de la
Microsoft Store, de escritorio y juegos de Steam— y lo añade a la allowlist. No
hay que ejecutar nada ni mantener listas.

El resultado se cachea 24 h, así que solo la primera ejecución del día paga el
coste de enumerar (unos segundos: lanza PowerShell y recorre el menú Inicio).

```yaml
# config.local.yaml — para ajustarlo
discovery:
  enabled: true       # false = solo lo que declares a mano
  steam: true
  cache_hours: 24.0
```

**Lo declarado a mano en `apps:` siempre gana.** Si pusiste la ruta exacta de
Spotify porque la detección fallaba, el descubrimiento no te la pisa.

> **Nota de seguridad.** Con el autodescubrimiento la allowlist deja de ser una
> lista curada y pasa a ser "todo lo instalado". El límite sigue existiendo
> —solo se abre software ya presente en el equipo, y el LLM sigue sin poder
> ejecutar comandos arbitrarios— pero es una frontera más ancha que antes. Pon
> `discovery.enabled: false` si prefieres controlar exactamente qué se abre.

Para ver qué encontró:

```powershell
python scripts/diagnose_apps.py
```

### Cuatro fuentes, porque ninguna las ve todas

| Fuente | Qué aporta |
|---|---|
| `Get-StartApps` | apps de la Store y de escritorio; la más completa |
| Menú Inicio (`.lnk`) | respaldo si PowerShell está restringido |
| App Paths (registro) | rutas de `.exe` exactas, permite cerrar la app |
| Steam | los juegos, que **no aparecen en ninguna de las otras** |

### Juegos de Steam

Se leen los manifiestos de Steam en disco (`appmanifest_*.acf`), incluyendo
bibliotecas repartidas en varios discos. Solo se listan los **instalados**:
ofrecer abrir algo que solo tienes comprado produciría fallos.

Se lanzan con `steam://rungameid/<id>`, no con el `.exe` del juego. Ejecutar el
ejecutable directamente se salta al cliente: no cuenta horas, no funciona el
overlay, no se sincronizan las partidas en la nube, y los juegos con DRM de
Steam se niegan a arrancar.

Se generan alias automáticos para los títulos largos, que nadie dice enteros:
*"Counter-Strike 2: Something"* → `Counter-Strike 2`, y siglas cuando son tres
palabras o más. Puedes añadir los tuyos en `config.local.yaml`:

```yaml
apps:
  counter_strike_2:
    command: 'steam://rungameid/730'
    aliases: ['Counter-Strike 2', 'cs', 'counter', 'el conter']
```

---

## Spotify: guía de configuración

Necesitas **Premium**: la Web API bloquea el control de reproducción en cuentas
gratuitas.

### 1. Crear la aplicación en Spotify

1. Entra en <https://developer.spotify.com/dashboard> con tu cuenta.
2. **Create app**. Nombre y descripción, lo que quieras.
3. En **Redirect URI** pon exactamente esto y pulsa *Add*:

   ```
   http://127.0.0.1:8888/callback
   ```

   Tiene que coincidir carácter por carácter con `spotify.redirect_uri` de
   `config.yaml`. Es el error más común: `localhost` **no** vale, Spotify exige
   `127.0.0.1`.
4. En **APIs used**, marca *Web API*. Guarda.
5. Copia el **Client ID** desde *Settings*.

### 2. Poner el Client ID

```powershell
copy .env.example .env
```

Edita `.env`:

```
SPOTIFY_CLIENT_ID=el_client_id_que_copiaste
SPOTIFY_CLIENT_SECRET=
```

El secret se deja **vacío a propósito**. Se usa el flujo PKCE, diseñado para
aplicaciones de escritorio: no hay servidor donde esconder un secreto, así que
no se usa ninguno. `.env` está en `.gitignore`.

### 3. Autorizar (una sola vez)

Arranca el asistente. Se abrirá el navegador pidiendo permiso; acepta. A partir
de ahí spotipy guarda un *refresh token* en `%LOCALAPPDATA%\asistente-local\`
y lo renueva solo. No vuelve a preguntar.

### 4. Comandos disponibles

Con Spotify abierto (hace falta un dispositivo activo):

| Dices | Hace |
|---|---|
| *"Apolo, reproduce Loving Machine de TV Girl"* | esa canción de ese grupo |
| *"Apolo, pon la canción Despacito de Luis Fonsi"* | igual, diciéndolo largo |
| *"Apolo, pon mi playlist de gym"* | **tu** playlist, no una pública que se llame igual |
| *"Apolo, pon mis me gusta"* / *"mis favoritas"* | tus canciones guardadas |
| *"Apolo, pon la playlist de rock"* | busca y reproduce una playlist |
| *"Apolo, pon música de los 80"* | idem por género o época |
| *"Apolo, quiero escuchar a Shakira"* | busca por artista |
| *"Apolo, ponme el álbum de Pink Floyd"* | busca por álbum |
| *"Apolo, pásala"* / *"esta no me gusta"* | siguiente canción |
| *"Apolo, qué canción es esta"* / *"quién canta esto"* | lo dice en voz alta |
| *"Apolo, me gusta esta canción"* / *"guárdala"* | la añade a tus favoritos |
| *"Apolo, pon el modo aleatorio"* | activa shuffle |

**Tus playlists van antes que el catálogo público.** Si dices *"pon mi playlist
de gym"* es porque existe y sabes cómo se llama; que exista una pública con el
mismo nombre es casualidad. Se buscan con `current_user_playlists` y se
emparejan por **palabras completas**: *"gym"* casa con "Gym mix" y no con
"Gimnasio". Ponle a la playlist el nombre que dices en voz alta.

**Los Me Gusta no son una playlist**: no tienen URI con la que arrancar la
reproducción, así que se leen las 50 guardadas más recientemente y se ponen en
cola. Si tienes el aleatorio puesto en Spotify, se aplica a esa cola.

**Decir de quién es la canción cambia la búsqueda entera.** Con artista se usan
los filtros de campo de Spotify (`track:"..." artist:"..."`) y se va directo a
la canción, sin pasar por playlists ni álbumes. Si eso no devuelve nada —porque
Whisper transcribió el título algo distinto de como está escrito— se reintenta
en texto libre, y por último con la frase entera tal cual la dijiste.

**No hace falta decir "la canción".** *"Reproduce loving machine de tv girl"*
funciona igual que *"pon la canción loving machine de tv girl"*: la primera la
resuelve la etapa de patrones por su forma, porque un título que el modelo no ha
visto nunca no se puede clasificar por significado (ver `ARCHITECTURE.md`).

**Tus Me Gusta se buscan primero.** Antes que tus playlists y antes que el
catálogo público. Además de ser lo que sueles querer, es donde se absorben los
títulos que Whisper transcribe mal: contra unos cientos de canciones tuyas, un
emparejamiento difuso acierta donde la Web API no encontraría nada.

**Los títulos en inglés** dichos en español se transcriben como suenan
(*"TV Gery"* por *"TV Girl"*), y ahí es donde entra la búsqueda en tu
biblioteca: ese par puntúa 90.9 sobre un umbral de 72.

**Y si la canción no está en tu biblioteca, la corrige el LLM.** Buscar en tus
Me Gusta solo encuentra lo que ya tienes; pedir algo nuevo con el título
destrozado no daba nada. Cuando Spotify no encuentra nada, el modelo local
reescribe el par {título, artista} —*"lovin machin de tibi guerl"* →
*"Loving Machine de TV Girl"*— y se busca otra vez.

Tres cosas que conviene saber de cómo se comporta:

- **Solo se paga cuando hace falta.** Lo que ya funcionaba sigue costando lo
  mismo; los 600–900 ms del modelo los paga únicamente la petición que iba a
  fallar de todos modos. Tampoco entra si el problema era otro: sin Spotify
  abierto, escribir mejor el título no arregla nada.
- **Antes de sonar, se comprueba que corrigió y no inventó.** Si el modelo no
  conoce la canción se saca un título plausible de la manga, y eso es peor que
  no encontrarla. Una corrección que se aleja demasiado de lo que oíste se tira
  y el asistente dice que no la encontró.
- **Se puede apagar** con `spotify.resolve_with_llm: false` en
  `config.local.yaml`. Vuelve a funcionar como antes, sin ningún otro efecto.

En el log, con `-v`, verás `el LLM corrige ...` cuando entra, y
`correccion descartada por inverosimil: ...` cuando el modelo se estaba
inventando el título. Si esa segunda línea sale a menudo, no conoce la música
que escuchas.

Se probó además sesgar a Whisper con los nombres de tu biblioteca
(`stt.hotwords_from_spotify`) y **viene desactivado porque con `large-v3-turbo`
empeora**: WER 22.4% → 24.8%, y llega a pegar el verbo al título
(*"Ponlovers Rock"*), con lo que el comando ni se enruta. Con `small` mejoraba
mucho, así que la opción sigue ahí para quien use otro modelo. Subir
`beam_size` no ayuda con ninguno de los dos.

**Si no encuentra lo que pediste, lo dice.** Antes se quedaba con el primer
resultado de la búsqueda, y como `search()` siempre devuelve algo, pedir *"loving
machine de tv girl"* podía acabar poniendo una playlist llamada "TV Girl". Ahora
el resultado tiene que **cubrir** las palabras que dijiste; si ninguno lo hace,
no suena nada y te lo dice. Es preferible repetir la orden a que suene otra cosa
y parezca que funcionó.

### Volumen: el del PC y el de Spotify son dos mandos distintos

Sin mencionar destino se controla **Windows**. Diciendo *"de Spotify"*, *"la
música"* o *"la canción"* se controla **solo Spotify**, que es la barra que sale
en el mezclador de volumen de Windows.

| Dices | Hace |
|---|---|
| *"Apolo, sube el volumen"* / *"súbele"* | volumen del PC, +10% |
| *"Apolo, pon el volumen al 40"* / *"sube el volumen al 40"* | volumen del PC, exacto |
| *"Apolo, silencio"* | silencia todo el PC |
| *"Apolo, sube el volumen de Spotify"* / *"súbele a Spotify"* | solo Spotify |
| *"Apolo, baja la música"* | solo Spotify |
| *"Apolo, pon Spotify al 30"* | volumen de Spotify, exacto |
| *"Apolo, silencia Spotify"* | mutea solo Spotify |
| *"Apolo, a qué volumen está la música"* | lo dice en voz alta |

**Con número fija ese valor; sin número se mueve un paso** (10%). Vale para los
dos destinos: *"sube el volumen"* sube el del PC de 10 en 10, *"sube el volumen
al 60"* lo pone en 60.

### Por qué Spotify va siempre por la app y no por el mezclador

Windows tiene un segundo mando —la barra de cada programa en el mezclador de
volumen— y es **otra cosa distinta**: solo afecta a lo que este PC saca por los
altavoces, no se ve desde ninguna parte de Spotify y no se sincroniza con el
móvil. El volumen de Spotify es el de **dentro de Spotify**, y ese solo se toca
por la Web API.

Se pagan dos cosas por hacerlo bien: un viaje de red (unas décimas frente a
milisegundos) y que deja de funcionar si no hay dispositivo activo. **No hay
degradación al mezclador a propósito**: sería cambiar un volumen distinto del
que pediste, y en silencio. Si no hay dispositivo, el asistente lo dice.

Algunos dispositivos de Connect y el reproductor web rechazan el cambio de
volumen con un 403; eso es de Spotify, no del asistente.

Si algo de esto no responde:

```powershell
python scripts/diagnose_volume.py
```

Prueba los cuatro mecanismos por separado (teclas multimedia, volumen maestro,
mezclador y Web API) y dice cuál falla y por qué. No deja nada cambiado:
restaura los valores que toca.

Si el diagnóstico sale en verde pero un comando concreto no hace lo esperado, el
propio asistente lo dice en el log en cada turno:

```
volumen spotify (paso +10): ok vía spotify-api (80 -> 90)
volumen system (paso -10): ok vía api-maestra (60 -> 50)
volumen spotify (fijar a 40): FALLO vía ninguna (? -> ?)
```

Con eso se ve de un vistazo **a qué destino fue**, **qué mecanismo actuó** y
**si el valor se movió de verdad**. Cuando no se mueve porque ya estaba al tope,
el asistente lo dice en voz alta en vez de quedarse callado.

La búsqueda prueba **playlist → álbum → canción**, en ese orden: *"pon rock"*
casi siempre significa una playlist, no la primera canción titulada "Rock".

### Si añades comandos que necesitan permisos nuevos

El token guardado solo sirve para los permisos con los que se creó. **Spotipy lo
detecta solo**: comprueba que los permisos del token cacheado cubran los pedidos
en `spotify.scopes` y, si no, lo descarta y vuelve a abrir el navegador. O sea
que al actualizar el asistente basta con aceptar otra vez.

Si por lo que sea no lo hiciera, se fuerza borrando el token:

```powershell
del "$env:LOCALAPPDATA\asistente-local\spotify-token.json"
```

Los permisos `playlist-read-private` y `playlist-read-collaborative` se añadieron
el 2026-08-09 para poder ver **tus** playlists. Sin ellos solo se ven las
públicas, y *"pon mi playlist de gym"* acabaría reproduciendo una ajena.

### Sin Premium, o si falla

Los comandos de reproducción caen a las **teclas multimedia de Windows**, que
funcionan con cualquier reproductor sin autenticación. Pierdes lo que requiere
la API: buscar por nombre, saber qué suena y guardar favoritos.

---

## Configuración: no edites `config.yaml`

`config.yaml` está versionado y trae los valores por defecto del proyecto. Los
ajustes propios de **tu** máquina —índice del micrófono, ganancia, rutas de
apps— van en `config.local.yaml`, que está en `.gitignore` y se fusiona encima
clave a clave:

```yaml
# config.local.yaml
audio:
  input_device: 2
  gain: 12.0        # solo esta clave; el resto de `audio` se hereda
```

Parte de `config.local.yaml.example`. Editar `config.yaml` directamente hace
que `git pull` falle con un conflicto sobre un fichero que en realidad no es
compartido:

```
error: Your local changes to the following files would be overwritten by merge:
        config.yaml
```

Si ya te ha pasado, mueve tus cambios a `config.local.yaml` y deja `config.yaml`
como está en el repo:

```powershell
git diff config.yaml        # mira qué cambiaste y cópialo a config.local.yaml
git checkout config.yaml
git pull
```

---

## Palabra clave

Por defecto: **"Apolo"**. Dile la clave y la orden del tirón:

> *"Apolo, pon música"* · *"Apolo, pásala"* · *"Apolo, qué hora es"*

Si dices solo *"Apolo"*, se queda escuchando la orden que venga después.

### Dos modos de activación

| | `transcript` (por defecto) | `openwakeword` |
|---|---|---|
| Palabras admitidas | **cualquiera, en español** | solo las que tengan modelo entrenado |
| Entrenamiento | ninguno | ~1 h en Colab para una propia |
| CPU en reposo | Whisper por cada frase cercana | 1–2% de un núcleo |
| "Apolo" | sí | requiere entrenarlo |

Se elige en `config.yaml`. `transcript` es lo que permite usar "Apolo" hoy: no
existe modelo preentrenado para esa palabra, y los que vienen con openWakeWord
(`hey_jarvis`, `alexa`, `hey_mycroft`, `hey_rhasspy`) están entrenados con voces
inglesas, así que hay que pronunciarlos a la inglesa.

**El coste de `transcript` es real**: Whisper se ejecuta cada vez que alguien
habla cerca del micrófono, no solo cuando le hablas al asistente. Con la GPU son
~150 ms por frase y no se nota; en CPU sí. Cuando el sistema esté rodado, la
opción eficiente es entrenar un modelo propio de "Apolo" y pasar a
`openwakeword`.

Si el asistente ignora frases que sí eran para él, mira el log en modo `-v`:
verás cómo transcribió tu palabra clave. Añade esa forma a `phrases`.

### `'spotify' is not recognized as an internal or external command`

La app no se encuentra. Casi ningún instalador de Windows mete su ejecutable en
el `PATH`; lo que hacen es registrarse en **App Paths** del registro, que es por
lo que `Win+R → spotify` sí funciona.

El asistente busca por cuatro vías, en este orden: ruta que exista → `PATH` →
App Paths → carpetas habituales de instalación (`%APPDATA%\Spotify\Spotify.exe`
y compañía). La cuarta se añadió porque App Paths **no es obligatorio**: hay
instaladores que no escriben la clave y actualizaciones que la dejan apuntando a
una versión que ya no existe.

Y si tu `config.yaml` apunta a algo que no se puede lanzar aquí, el
autodescubrimiento **sustituye el comando** por el que sí funciona (conservando
tus alias y tu nombre de proceso) y lo avisa en el log:

```
apps.spotify: 'spotify' no se puede lanzar aqui; se usa lo descubierto
(start_apps): shell:AppsFolder\SpotifyAB.SpotifyMusic_zpdnekdrzrea0!Spotify
```

Comprueba toda la allowlist de una vez:

```powershell
python scripts/diagnose_apps.py          # qué se abre, qué se cierra
python scripts/diagnose_apps.py --fix    # además, intenta adivinar las rutas
```

Las que fallen, con ruta completa en **`config.local.yaml`**:

```yaml
apps:
  spotify:
    command: 'C:\Users\TU_USUARIO\AppData\Roaming\Spotify\Spotify.exe'
    process: Spotify
```

### *"Cierra X"* no cierra nada

Abrir y cerrar son dos problemas distintos: abrir necesita una **ruta** y cerrar
necesita el **nombre del proceso**, que a menudo no aparece en ningún sitio de la
config. Las apps de la Microsoft Store llegan del descubrimiento sin proceso
alguno, y los accesos directos del menú Inicio con uno adivinado pegando el
título ("Visual Studio Code" → `VisualStudioCode.exe`, que no existe: es
`Code.exe`).

Por eso el nombre ya no se adivina: se cruzan los nombres plausibles de la
entrada (su `process`, su comando, su clave y sus alias) con la lista de
procesos que **están corriendo**, y se mata uno que existe. Si no coincide
ninguno, te dice que no está abierto en vez de fallar en silencio.

`python scripts/diagnose_apps.py` imprime, para lo que hay abierto ahora mismo,
qué proceso se mataría con cada app.

### El asistente no entiende una frase

Cuando una frase llega al LLM en vez de resolverse en el catálogo, el log lo
dice con nombre y apellidos:

```
INFO  asistente.pipeline: AL CATALOGO LE FALTA ESTA FRASE: 'ponme algo tranquilo'
      -> anadela a `examples` del intent correspondiente en commands.yaml
```

Copia esa frase a `examples` del intent que corresponda y reinicia. Es la forma
práctica de afinarlo: nadie recuerda después cómo lo dijo exactamente.

Si además **ignora** frases que sí eran para él, arranca con `-v`: verás en el
log cómo transcribió tu palabra clave, y podrás añadir esa forma a
`wake_word.phrases`.

**Cuidado con las negaciones.** *"Me gusta esta canción"* (guardar) y *"no me
gusta esta"* (siguiente) se diferencian en una palabra, y los embeddings manejan
mal la negación. Si añades ejemplos a un intent con negación, revisa el opuesto.

### Ruido de fondo: ventilador, música, la tele

Empieza midiendo tu habitación en lugar de tocar valores a ciegas:

```powershell
python scripts/diagnose_noise.py
```

Graba 4 s de silencio y 4 s hablando, calcula la SNR real de tu sala y te dice
qué números poner en `config.local.yaml`.

Hay **tres defensas**, y actúan por orden de coste. Las dos primeras vienen
activadas:

| Defensa | Contra qué | Coste |
|---|---|---|
| Puertas del VAD (`min_speech_s`, `min_snr_db`) | picos sueltos que no son voz | microsegundos |
| Supresión de ruido (`stt.denoise`) | ventilador, aire, zumbido | ~5 ms |
| Verificación de locutor (`speaker.enabled`) | la tele, música cantada, otra persona | ~20 ms |

**Lo que está medido** (12 clips, ruido simulado, Whisper `small` en CPU — un
banco más duro que una habitación real):

- Ventilador a **5 dB** de SNR: WER medio **97.9% → 72.9%**.
- Ventilador a **10 dB**: 66.0% → 68.1%, o sea igual dentro del ruido de la medida.
- Sobre audio limpio: cambia el RMS en 2e-8. No toca nada.

Es decir: ayuda cuando hace falta y no estorba cuando no. El barrido de
`denoise_strength` y `denoise_floor` salió **inconcluso** con esa muestra, así
que los valores por defecto son un término medio prudente, no un óptimo. El
barrido que vale es el que hagas en el PC con tu ventilador de verdad.

Whisper además **inventa texto** cuando solo oye ruido — en español casi siempre
"Subtítulos realizados por la comunidad de Amara.org" y parientes. Se filtran
por lista cerrada, más los umbrales `no_speech_threshold` y `log_prob_threshold`
del propio modelo.

### Que solo te haga caso a ti

```powershell
python scripts/enroll_voice.py          # graba 6 frases y guarda tu perfil
python scripts/enroll_voice.py --probar # comprobar cómo te puntúa
```

Al terminar te recomienda un umbral calculado con **tus** grabaciones. Actívalo
en `config.local.yaml`:

```yaml
speaker:
  enabled: true
  threshold: 0.50     # el que te haya dicho el script
```

> **Viene desactivado, y por una razón medida.** Con ruido de ventilador el
> coseno de tu propia voz baja de ~0.80 a 0.49–0.83, y con ruido de banda ancha
> se hunde a 0.29–0.44 — territorio de "otra persona". Falla justo cuando haría
> falta. Contra el ventilador funcionan mejor el denoise y las puertas del VAD,
> que ya están activados. Esto sirve contra el ruido que **habla**.
>
> Enrola **en tu sitio y en tus condiciones**: el perfil recoge la sala y el
> micrófono, no solo la voz. Si el ventilador suele estar encendido, déjalo
> encendido al enrolar.
>
> Si el asistente deja de oírte, **baja el umbral** o pon `enabled: false`. Cada
> turno registra el coseno en el log; ajusta con esos números, no a ojo.

### El micrófono capta muy bajo

```powershell
python scripts/diagnose_audio.py --list    # ver los micrófonos disponibles
python scripts/diagnose_audio.py           # grabar 5 s y diagnosticar
```

Distingue las cuatro causas posibles —micrófono mudo, nivel bajo, VAD que no
oye voz, o umbral demasiado alto— y **calcula la ganancia exacta** que necesita
tu micrófono. Compruébala antes de fijarla:

```powershell
python scripts/diagnose_audio.py --gain 8
```

**Dos ajustes distintos, para dos problemas distintos.** Es fácil confundirlos:

| Ajuste | Cuándo actúa | A quién ayuda |
|---|---|---|
| `audio.gain` | sobre los bloques según llegan | VAD y palabra clave |
| `stt.normalize_audio` | sobre la frase ya grabada | Whisper |

Aplicar solo uno deja el otro problema sin resolver. Medido sobre voz atenuada
a 1/50: el VAD seguía detectando habla (0.951 de pico) pero su probabilidad
media caía de 0.859 a **0.589** — con ruido de fondo real ese margen desaparece.
Con `gain: 10` vuelve a 0.816.

No te pases con la ganancia: amplifica también el ruido, y si satura verás
`el audio satura con gain=...` en el log. **La voz distorsionada se transcribe
peor que la floja.**

---

## Problemas conocidos

### `RuntimeError: Library cublas64_12.dll is not found or cannot be loaded`

CTranslate2 (el motor de faster-whisper) necesita cuBLAS y cuDNN de CUDA 12, y
**no vienen con el driver de NVIDIA**:

```powershell
pip install -e ".[gpu]"
```

Además hay que decirle a Windows dónde están, y **por dos vías distintas**:
`os.add_dll_directory()` (que usan las extensiones de Python) y el `PATH` (que
usa `LoadLibrary` cuando lo llama código nativo). CTranslate2 carga cuBLAS desde
su propio C++, o sea por la segunda: registrar solo los directorios no basta —
el log dice "registrado" y la carga falla igual. `src/asistente/cuda_setup.py`
hace las dos y comprueba con `ctypes` que la DLL carga de verdad.

Si aun así falla, el asistente **degrada el STT a CPU automáticamente** y avisa
en el log. Funciona, pero más lento: sirve para probar el resto del pipeline
mientras arreglas CUDA.

### `ollama ResponseError: time: missing unit in duration "-1"`

`keep_alive` en `config.yaml` tiene que ser número, no cadena. Ollama interpreta
las cadenas como duraciones con unidad (`10m`, `1h`); un `"-1"` entrecomillado
da HTTP 400. Correcto:

```yaml
llm:
  keep_alive: -1     # sin comillas
```

### El silencio y las teclas multimedia no hacían nada (corregido)

Síntoma: *"Apolo, silencio"* respondía *"no pude silenciar el sonido"*, y las
teclas de reproducción no llegaban nunca a Spotify. Sin excepción y sin log.

Causa: `SendInput` valida que `cbSize` sea **exactamente** `sizeof(INPUT)` y
devuelve 0 si no cuadra. La unión `INPUT` estaba declarada solo con
`KEYBDINPUT` —lo único que el asistente usa— y eso da 32 bytes; Windows espera
40 en x64, porque el miembro que fija el tamaño es `MOUSEINPUT`. La llamada se
rechazaba entera y el único indicio era un valor de retorno que nadie miraba.

Se notaba poco porque los comandos de reproducción intentan la Web API de
Spotify primero y las teclas son solo el respaldo; el silencio, en cambio, no
tenía otra vía. Ahora `winkeys.py` declara la unión completa, registra
`GetLastError()` cuando falla, y `tests/test_winkeys.py` fija el tamaño. El
silencio además pasa por la API de audio y solo usa la tecla como respaldo.

Si `scripts/diagnose_volume.py` dice que las pulsaciones se rechazan pese al
tamaño correcto, la causa suele ser **UIPI**: la ventana en primer plano corre
como administrador y el asistente no, así que Windows bloquea la inyección.

### `UserWarning: cache-system uses symlinks... your machine does not support them`

Inofensivo. Hugging Face cachea sin symlinks y ocupa algo más de disco. Para
silenciarlo, `setx HF_HUB_DISABLE_SYMLINKS_WARNING 1`. Si prefieres los
symlinks, activa el Modo Desarrollador de Windows.

---

## Documentación

- [`ARCHITECTURE.md`](ARCHITECTURE.md) — diseño, decisiones y mediciones.
- `commands.yaml` — catálogo de comandos, comentado.
- `config.yaml` — parámetros; los marcados `TUNING` son los del barrido de la Fase 5.
