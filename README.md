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

```powershell
# 1. Entorno
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
pip install -e ".[gpu]"            # CUDA: onnxruntime-gpu + cuBLAS + cuDNN

# 2. Modelo del LLM
ollama pull qwen2.5:3b-instruct-q4_K_M

# 3. Voz de Piper: descarga el .onnx y su .json a models/
#    https://huggingface.co/rhasspy/piper-voices/tree/main/es

# 4. Credenciales de Spotify
copy .env.example .env             # y rellena SPOTIFY_CLIENT_ID

# 5. Arrancar
python -m asistente
```

La primera ejecución descarga el encoder de embeddings (~470 MB) y el modelo de
Whisper. A partir de ahí quedan en caché.

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

## Problemas conocidos

### `RuntimeError: Library cublas64_12.dll is not found or cannot be loaded`

CTranslate2 (el motor de faster-whisper) necesita cuBLAS y cuDNN de CUDA 12, y
**no vienen con el driver de NVIDIA**:

```powershell
pip install -e ".[gpu]"
```

Además, desde Python 3.8 Windows ya no busca DLL en el `PATH`: hay que declarar
los directorios explícitamente. De eso se encarga `src/asistente/cuda_setup.py`,
que corre antes de cargar CTranslate2.

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
