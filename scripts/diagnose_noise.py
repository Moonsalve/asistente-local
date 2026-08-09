"""Mide el ruido de TU habitacion y recomienda umbrales.

Los valores por defecto de `config.yaml` salen de simulaciones, no de tu cuarto.
Este script mide lo que hay de verdad —el ventilador, el aire, el zumbido del
PC— y dice que numeros poner.

    python scripts/diagnose_noise.py

Hace dos grabaciones: una en silencio (para el suelo de ruido) y otra hablando
(para saber cuanto destaca tu voz sobre el). Con las dos calcula la SNR real y
comprueba que las puertas del pipeline dejan pasar tu voz y paran el ruido.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asistente.audio.denoise import denoise, snr_db  # noqa: E402
from asistente.audio.vad import SileroVad  # noqa: E402
from asistente.config import Config  # noqa: E402

OK = "  OK "
AVISO = "AVISO"
MAL = " MAL "


def grabar(segundos: float, config: Config) -> np.ndarray:
    import sounddevice as sd

    audio = sd.rec(
        int(segundos * config.audio.sample_rate),
        samplerate=config.audio.sample_rate,
        channels=1,
        dtype="float32",
        device=config.audio.input_device,
    )
    sd.wait()
    return audio.reshape(-1) * config.audio.gain


def cuenta_atras(mensaje: str) -> None:
    print(f"\n{mensaje}")
    for i in (3, 2, 1):
        print(f"   {i}...", end="\r", flush=True)
        time.sleep(0.7)
    print("   GRABANDO      ", end="\r", flush=True)


def proporcion_de_voz(audio: np.ndarray, config: Config) -> float:
    """Fraccion de bloques que el VAD da por voz. Sobre una grabacion en
    silencio deberia ser ~0; si no lo es, el ventilador esta pasando por voz."""
    vad = SileroVad(config.audio.sample_rate)
    bloque = config.audio.block_size
    total = habla = 0
    for i in range(0, len(audio) - bloque, bloque):
        prob = vad.speech_probability(audio[i : i + bloque])
        if prob is None:
            continue
        total += 1
        habla += prob >= config.vad.speech_threshold
    return habla / total if total else 0.0


def main() -> int:
    config = Config.load(ROOT / "config.yaml")

    print("=" * 70)
    print("DIAGNOSTICO DE RUIDO DE LA SALA")
    print("=" * 70)
    print("\nDeja todo como lo tienes normalmente: si el ventilador suele estar")
    print("encendido, dejalo encendido. Se trata de medir tu situacion real.")

    cuenta_atras("1/2  NO HABLES. Grabando solo el ruido de fondo (4 s).")
    ruido = grabar(4.0, config)
    print("     hecho                    ")

    cuenta_atras('2/2  Ahora DI: "Apolo, sube el volumen de la música" (4 s).')
    voz = grabar(4.0, config)
    print("     hecho                    ")

    rms_ruido = float(np.sqrt(np.mean(ruido**2)))
    rms_voz = float(np.sqrt(np.mean(voz**2)))
    pico_voz = float(np.max(np.abs(voz)))

    print("\n" + "-" * 70)
    print("NIVELES")
    print(f"     ruido de fondo   RMS {rms_ruido:.5f}")
    print(f"     con tu voz       RMS {rms_voz:.5f}   pico {pico_voz:.3f}")

    if rms_ruido < 1e-6:
        print(f"{MAL} el micrófono no captó nada. ¿Dispositivo correcto? "
              "python scripts/diagnose_audio.py --list")
        return 1

    snr_real = 20 * np.log10(rms_voz / rms_ruido) if rms_voz > rms_ruido else 0.0
    print(f"\n     SNR real de la sala: {snr_real:.1f} dB")
    if snr_real >= 20:
        print(f"{OK} margen amplio: tu voz destaca de sobra sobre el fondo")
    elif snr_real >= 10:
        print(f"{OK} margen suficiente")
    else:
        print(f"{AVISO} margen escaso. Acércate al micrófono o aléjalo del ventilador:")
        print("      ninguna limpieza por software compensa un micrófono mal colocado.")

    if pico_voz > 0.99:
        print(f"{AVISO} la voz satura (pico {pico_voz:.2f}). Baja `audio.gain`.")

    print("\n" + "-" * 70)
    print("PUERTA DEL VAD  (¿el ruido pasa por voz?)")
    frac_ruido = proporcion_de_voz(ruido, config)
    frac_voz = proporcion_de_voz(voz, config)
    print(f"     bloques dados por voz    en silencio: {frac_ruido:>5.1%}"
          f"     hablando: {frac_voz:>5.1%}")
    if frac_ruido > 0.15:
        sugerido = min(0.9, config.vad.speech_threshold + 0.15)
        print(f"{AVISO} el ruido dispara el VAD demasiado. Sube el umbral:")
        print(f"      vad:\n        speech_threshold: {sugerido:.2f}")
    else:
        print(f"{OK} el ruido no dispara el VAD")
    if frac_voz < 0.3:
        print(f"{AVISO} tu voz apenas dispara el VAD. Sube `audio.gain` o baja el umbral.")

    print("\n" + "-" * 70)
    print("EFECTO DEL DENOISE")
    antes, despues = snr_db(voz), snr_db(denoise(voz))
    print(f"     SNR estimada de la frase   {antes:.1f} dB  ->  {despues:.1f} dB")
    print(f"     (la puerta `vad.min_snr_db` está en {config.vad.min_snr_db:.1f} dB)")
    if antes < config.vad.min_snr_db:
        print(f"{AVISO} tu propia voz queda por debajo de la puerta y sería descartada.")
        print(f"      Baja min_snr_db a {max(0.0, antes - 2):.0f}, o mejora la colocación.")
    else:
        print(f"{OK} tu voz pasa la puerta con {antes - config.vad.min_snr_db:.1f} dB de margen")

    print("\n" + "=" * 70)
    print("Los ajustes van en config.local.yaml, NUNCA en config.yaml.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
