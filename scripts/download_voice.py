"""Descarga una voz de Piper desde Hugging Face.

Una voz son SIEMPRE dos archivos que tienen que ir juntos:

    es_MX-claude-high.onnx        el modelo
    es_MX-claude-high.onnx.json   su configuracion (sample rate, fonemas...)

Piper falla con un FileNotFoundError sobre el .json aunque el que falte sea el
.onnx, asi que bajarlos a mano es facil de hacer mal.

Uso:
    python scripts/download_voice.py                    # voz por defecto
    python scripts/download_voice.py --voice es_ES-davefx-medium
    python scripts/download_voice.py --list             # ver las disponibles
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_ID = "rhasspy/piper-voices"

#: Voces en espanol. La ruta dentro del repo sigue el patron
#: <idioma>/<locale>/<nombre>/<calidad>/<archivo>.
VOICES: dict[str, str] = {
    "es_MX-claude-high": "es/es_MX/claude/high",
    "es_MX-ald-medium": "es/es_MX/ald/medium",
    "es_ES-davefx-medium": "es/es_ES/davefx/medium",
    "es_ES-sharvard-medium": "es/es_ES/sharvard/medium",
    "es_ES-carlfm-x_low": "es/es_ES/carlfm/x_low",
}

DEFAULT_VOICE = "es_MX-claude-high"


def main() -> int:
    parser = argparse.ArgumentParser(description="Descarga una voz de Piper")
    parser.add_argument("--voice", default=DEFAULT_VOICE, choices=sorted(VOICES))
    parser.add_argument("--out", type=Path, default=ROOT / "models")
    parser.add_argument("--list", action="store_true", help="lista las voces y sale")
    args = parser.parse_args()

    if args.list:
        print("Voces disponibles:")
        for name in sorted(VOICES):
            marca = "  (por defecto)" if name == DEFAULT_VOICE else ""
            print(f"  {name}{marca}")
        return 0

    from huggingface_hub import hf_hub_download

    args.out.mkdir(parents=True, exist_ok=True)
    directory = VOICES[args.voice]

    for suffix in (".onnx", ".onnx.json"):
        filename = f"{args.voice}{suffix}"
        print(f"descargando {filename}...")
        downloaded = hf_hub_download(REPO_ID, f"{directory}/{filename}")
        target = args.out / filename
        # copy2 en vez de symlink: el cache de HF en Windows ya avisa de que no
        # soporta symlinks, y una voz son ~60 MB, no merece complicarse.
        import shutil

        shutil.copy2(downloaded, target)
        print(f"  -> {target}")

    print(f"\nListo. En config.yaml:\n  tts:\n    voice_model: models/{args.voice}.onnx")
    return 0


if __name__ == "__main__":
    sys.exit(main())
