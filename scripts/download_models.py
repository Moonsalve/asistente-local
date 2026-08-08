"""Descarga los modelos de openWakeWord.

El paquete pip de openWakeWord trae solo el codigo: los .onnx hay que bajarlos
aparte. Son cuatro y los necesita todos el pipeline:

    melspectrogram.onnx    preprocesado de audio comun a cualquier wake word
    embedding_model.onnx   idem
    hey_jarvis_v0.1.onnx   la palabra clave en si
    silero_vad.onnx        el VAD que usa nuestro endpointing

Sin ellos el sintoma es un NO_SUCHFILE de onnxruntime apuntando a un fichero
dentro de site-packages, que parece una instalacion corrupta y no lo es.

El asistente los descarga solo la primera vez que arranca; este script sirve
para hacerlo por adelantado o para reparar una descarga a medias.

Uso:  python scripts/download_models.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    from asistente.audio.wakeword import download_models, models_dir

    download_models()

    directory = models_dir()
    print(f"\nModelos en {directory}:")
    for path in sorted(directory.glob("*.onnx")):
        print(f"  {path.name:<28} {path.stat().st_size / 1024:>8.0f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
