"""Diagnostico del microfono y del wake word.

"No me detecta la voz" tiene al menos cuatro causas posibles y hay que
distinguirlas antes de tocar nada:

  1. El microfono no captura (dispositivo equivocado, silenciado, sin permisos).
  2. Captura pero muy bajo (ganancia baja: el VAD no lo considera voz).
  3. Captura bien pero el wake word no puntua (pronunciacion, umbral).
  4. Todo bien pero el umbral esta demasiado alto.

Este script las separa: lista dispositivos, graba, mide el nivel, pasa el audio
por el VAD y por el wake word, y te dice cual de las cuatro es.

Uso:
    python scripts/diagnose_audio.py --list          # ver dispositivos
    python scripts/diagnose_audio.py                 # prueba de 5 segundos
    python scripts/diagnose_audio.py --device 2      # probar otro microfono
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SAMPLE_RATE = 16_000
BLOCK = 1280


def list_devices() -> int:
    import sounddevice as sd

    print("Dispositivos de ENTRADA disponibles:\n")
    default_in = sd.default.device[0]
    for index, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] < 1:
            continue
        marca = "  <-- por defecto" if index == default_in else ""
        print(f"  [{index}] {dev['name']}")
        print(f"       canales={dev['max_input_channels']} sr={dev['default_samplerate']:.0f}{marca}")
    print("\nPara usar uno concreto, en config.yaml:\n  audio:\n    input_device: <numero>")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnostico de microfono y wake word")
    parser.add_argument("--list", action="store_true", help="lista los microfonos y sale")
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--model", default="hey_jarvis")
    parser.add_argument("--gain", type=float, default=1.0, help="ganancia a aplicar en la prueba")
    args = parser.parse_args()

    if args.list:
        return list_devices()

    import sounddevice as sd

    from asistente.audio.vad import SileroVad
    from asistente.audio.wakeword import WakeWordDetector

    device = args.device
    info = sd.query_devices(device if device is not None else sd.default.device[0])
    print(f"Microfono: {info['name']}")
    print(f"\nGrabando {args.seconds:.0f} s. DI LA PALABRA CLAVE en voz alta, como hablarias normalmente.\n")

    frames: list[np.ndarray] = []
    vad = SileroVad(SAMPLE_RATE)
    detector = WakeWordDetector(args.model, threshold=0.0, refractory_s=0.0)
    scores: list[float] = []
    probs: list[float] = []

    with sd.InputStream(
        samplerate=SAMPLE_RATE, blocksize=BLOCK, device=device, channels=1, dtype="float32"
    ) as stream:
        for i in range(int(args.seconds * SAMPLE_RATE / BLOCK)):
            block, overflowed = stream.read(BLOCK)
            if overflowed:
                print("  aviso: overflow del buffer de entrada")
            mono = np.clip(block[:, 0] * args.gain, -1.0, 1.0) if args.gain != 1.0 else block[:, 0]
            frames.append(mono)

            if (prob := vad.speech_probability(mono)) is not None:
                probs.append(prob)
            raw = detector._model.predict((mono * 32767).astype(np.int16))  # noqa: SLF001
            scores.append(max(raw.values()) if raw else 0.0)

            rms = float(np.sqrt(np.mean(mono**2)))
            barra = "#" * min(40, int(rms * 400))
            print(f"\r  [{i * BLOCK / SAMPLE_RATE:4.1f}s] nivel {rms:.4f} |{barra:<40}|", end="")

    print("\n")
    audio = np.concatenate(frames)
    rms = float(np.sqrt(np.mean(audio**2)))
    peak = float(np.max(np.abs(audio)))

    # Nivel de la parte hablada, no de toda la grabacion: promediar los
    # silencios hunde el RMS y da una ganancia recomendada demasiado alta.
    bloques_rms = np.array([float(np.sqrt(np.mean(f**2))) for f in frames])
    umbral_voz = max(bloques_rms.mean(), 1e-6)
    voz = bloques_rms[bloques_rms > umbral_voz]
    rms_voz = float(voz.mean()) if voz.size else rms

    print("=" * 70)
    print("RESULTADOS")
    print("=" * 70)
    if args.gain != 1.0:
        print(f"  ganancia aplicada     : x{args.gain:.1f}")
    print(f"  nivel medio (RMS)     : {rms:.4f}")
    print(f"  nivel al hablar (RMS) : {rms_voz:.4f}   <- el que importa")
    print(f"  pico                  : {peak:.4f}")
    print(f"  VAD  max / media      : {max(probs):.3f} / {sum(probs) / len(probs):.3f}")
    print(f"  wake word max         : {max(scores):.3f}   (umbral por defecto 0.5)")

    # Objetivo: voz alrededor de RMS 0.05, que es un nivel de conversacion
    # normal y donde el VAD y Whisper trabajan comodos.
    objetivo = 0.05
    if rms_voz > 1e-6:
        sugerida = objetivo / rms_voz * args.gain
        # Techo por el pico: no tiene sentido una ganancia que sature.
        if peak > 1e-6:
            sugerida = min(sugerida, 0.95 / peak * args.gain)
        sugerida = max(1.0, min(sugerida, 50.0))
        print(f"\n  GANANCIA RECOMENDADA  : {sugerida:.1f}")
        if sugerida > 1.2:
            print("\n  En config.yaml:")
            print("    audio:")
            print(f"      gain: {sugerida:.1f}")
            print(f"\n  Compruebalo con:  python scripts/diagnose_audio.py --gain {sugerida:.1f}")

    print("\n" + "=" * 70)
    print("VEREDICTO")
    print("=" * 70)
    if peak < 0.001:
        print("  1. EL MICROFONO NO CAPTURA NADA.")
        print("     - Comprueba que no este silenciado en Windows.")
        print("     - Ajustes > Privacidad > Microfono: permitir a las apps de escritorio.")
        print("     - Prueba otro dispositivo:  python scripts/diagnose_audio.py --list")
    elif rms_voz < 0.005:
        print("  2. CAPTURA MUY BAJO.")
        print("     Si ya tienes el volumen de Windows al maximo, usa la ganancia por")
        print("     software: mira la GANANCIA RECOMENDADA de arriba y ponla en config.yaml.")
    elif max(probs) < 0.5:
        print("  2b. HAY SENAL PERO EL VAD NO DETECTA VOZ.")
        print("     - Puede ser ruido de fondo constante en vez de habla.")
    elif max(scores) < 0.2:
        print("  3. EL AUDIO ES BUENO PERO EL WAKE WORD NO RECONOCE LA PALABRA.")
        print("     Los modelos preentrenados son INGLESES: 'hey jarvis' hay que")
        print("     pronunciarlo a la inglesa. Opciones:")
        print("     - Prueba 'alexa', que suena igual en ambos idiomas:")
        print("         python scripts/diagnose_audio.py --model alexa")
        print("     - O usa el modo transcripcion, que acepta cualquier palabra")
        print("       en espanol (ver README, seccion Palabra clave).")
    elif max(scores) < 0.5:
        print(f"  4. CASI. El wake word puntuo {max(scores):.3f}, por debajo del umbral 0.5.")
        print(f"     Baja el umbral en config.yaml:\n         wake_word:\n           threshold: {max(scores) * 0.8:.2f}")
    else:
        print(f"  TODO BIEN. El wake word puntuo {max(scores):.3f} (>= 0.5).")
        print("     Si aun asi no funciona en el asistente, revisa que config.yaml")
        print("     apunte al mismo dispositivo que has probado aqui.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
