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

### El micrófono no detecta nada

```powershell
python scripts/diagnose_audio.py --list    # ver los micrófonos disponibles
python scripts/diagnose_audio.py           # grabar 5 s y diagnosticar
```

Distingue las cuatro causas posibles —micrófono mudo, ganancia baja, VAD que no
oye voz, o umbral demasiado alto— y dice cuál es la tuya.

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
