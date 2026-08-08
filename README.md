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
| 2 | Router de 3 etapas + skills | **Hecha y verificada** (81 tests) |
| 4 | Fallback con LLM (Ollama) | Código escrito, sin probar contra Ollama |
| 1 | Audio: wake word, VAD, STT | Código escrito, **requiere el PC Windows** |
| 3 | Spotify OAuth + control de sistema | Código escrito, **requiere el PC Windows** |
| 5 | Benchmark y tuning | Pendiente |

Lo verificado hasta ahora se ejecutó en macOS, que es donde se puede probar el
router. Todo lo que toca micrófono, CUDA, `pycaw` o `SendInput` solo funciona en
Windows y está sin ejecutar.

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

# 2. Modelo del LLM
ollama pull qwen2.5:3b-instruct-q4_K_M

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
PYTHONPATH=src pytest -q                      # 81 tests, ~2 s
PYTHONPATH=src python scripts/diagnose_router.py   # scores del router frase a frase
```

`tests/test_router.py` mide el router contra frases que **no** están en
`commands.yaml`: probar con las del propio catálogo no demostraría nada.

---

## Aplicaciones: autodescubrimiento

No hace falta escribir la lista a mano. Enumera lo instalado —apps de la
Microsoft Store, de escritorio y juegos de Steam— y genera el bloque `apps:`:

```powershell
python scripts/discover_apps.py            # ver qué encuentra
python scripts/discover_apps.py --write    # escribirlo en config.local.yaml
python scripts/diagnose_apps.py            # comprobar que todo se localiza
```

**Revisa la lista antes de `--write`**: lo que quede ahí es la allowlist, o sea
lo único que el asistente puede abrir. Para acotarla:

```powershell
python scripts/discover_apps.py --filter steam --write   # solo juegos
python scripts/discover_apps.py --no-steam --limit 30 --write
```

`--write` reemplaza solo la sección `apps:` y hace copia de seguridad, así que
la ganancia del micrófono y el resto de tus ajustes se conservan.

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
| *"Apolo, pon la playlist de rock"* | busca y reproduce una playlist |
| *"Apolo, pon música de los 80"* | idem por género o época |
| *"Apolo, quiero escuchar a Shakira"* | busca por artista |
| *"Apolo, ponme el álbum de Pink Floyd"* | busca por álbum |
| *"Apolo, pásala"* / *"esta no me gusta"* | siguiente canción |
| *"Apolo, qué canción es esta"* / *"quién canta esto"* | lo dice en voz alta |
| *"Apolo, me gusta esta canción"* / *"guárdala"* | la añade a tus favoritos |
| *"Apolo, pon el modo aleatorio"* | activa shuffle |
| *"Apolo, sube el volumen"* | volumen del sistema |

La búsqueda prueba **playlist → álbum → canción**, en ese orden: *"pon rock"*
casi siempre significa una playlist, no la primera canción titulada "Rock".

### Si añades comandos que necesitan permisos nuevos

El token guardado solo sirve para los permisos con los que se creó. Al añadir
`spotify.like` o `spotify.what_song` sobre una autorización antigua, borra el
token para que Spotify vuelva a pedirlos:

```powershell
del "$env:LOCALAPPDATA\asistente-local\spotify-token.json"
```

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
lo que `Win+R → spotify` sí funciona. El asistente consulta esa clave, pero si
una app no está ahí hay que darle la ruta.

Comprueba toda la allowlist de una vez:

```powershell
python scripts/diagnose_apps.py          # cuáles se localizan y cuáles no
python scripts/diagnose_apps.py --fix    # además, intenta adivinar las rutas
```

Las que fallen, con ruta completa en **`config.local.yaml`**:

```yaml
apps:
  spotify:
    command: 'C:\Users\TU_USUARIO\AppData\Roaming\Spotify\Spotify.exe'
    process: Spotify
```

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

### `UserWarning: cache-system uses symlinks... your machine does not support them`

Inofensivo. Hugging Face cachea sin symlinks y ocupa algo más de disco. Para
silenciarlo, `setx HF_HUB_DISABLE_SYMLINKS_WARNING 1`. Si prefieres los
symlinks, activa el Modo Desarrollador de Windows.

---

## Documentación

- [`ARCHITECTURE.md`](ARCHITECTURE.md) — diseño, decisiones y mediciones.
- `commands.yaml` — catálogo de comandos, comentado.
- `config.yaml` — parámetros; los marcados `TUNING` son los del barrido de la Fase 5.
