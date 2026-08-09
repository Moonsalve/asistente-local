"""Enrola tu voz para que el asistente solo responda a ti.

Graba varias frases, calcula el centroide de sus embeddings y lo guarda. A
partir de ahi, con `speaker.enabled: true` en `config.local.yaml`, las frases
que no se parezcan a la tuya se descartan antes de llegar a Whisper.

    python scripts/enroll_voice.py
    python scripts/enroll_voice.py --frases 8        # mas muestras, mas estable
    python scripts/enroll_voice.py --probar          # comprobar el perfil ya hecho

GRABA EN TU SITIO Y EN TUS CONDICIONES
--------------------------------------
El perfil recoge la sala y el microfono, no solo la voz. Enrolar en silencio
absoluto y luego usarlo con el ventilador puesto es la receta para que deje de
reconocerte: el ruido desplaza el embedding. Si el ventilador suele estar
encendido, enrola con el encendido.

ANTES DE ACTIVARLO, LEE ESTO
----------------------------
Esta medido que la verificacion de locutor se degrada con ruido de fondo, que es
justo cuando haria falta. Contra el ventilador funcionan mejor el denoise y las
puertas del VAD, que ya estan activados. Esto sirve contra el ruido que HABLA:
la television, la musica cantada, otra persona en la habitacion.

Al terminar, el script dice la separacion que ha medido con TUS grabaciones y
recomienda un umbral. Hazle caso a ese numero antes que al valor por defecto.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asistente.audio.speaker import (  # noqa: E402
    SpeakerEmbedder,
    build_profile,
    download_model,
)
from asistente.config import Config  # noqa: E402

FRASES_SUGERIDAS = (
    "Apolo, sube el volumen",
    "Apolo, pon música de los ochenta",
    "Apolo, qué hora es",
    "Apolo, abre Spotify",
    "Apolo, baja el volumen de la música",
    "Apolo, pasa la canción",
    "Apolo, busca el clima de hoy",
    "Apolo, silencia el sonido",
)


def grabar(segundos: float, sample_rate: int, device: int | None) -> np.ndarray:
    import sounddevice as sd

    audio = sd.rec(
        int(segundos * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        device=device,
    )
    sd.wait()
    return audio.reshape(-1)


def recortar_silencio(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Quita el silencio de los extremos por energia en ventanas de 20 ms.

    Importa mas de lo que parece: el embedding se calcula sobre la frase entera,
    asi que un segundo de silencio al final lo arrastra hacia "sala vacia" y
    hace el perfil menos discriminante.
    """
    ventana = int(0.02 * sample_rate)
    if audio.size < ventana * 3:
        return audio
    trozos = audio[: len(audio) // ventana * ventana].reshape(-1, ventana)
    energia = np.sqrt((trozos**2).mean(axis=1))
    activos = np.where(energia > max(float(energia.max()) * 0.1, 1e-4))[0]
    if activos.size == 0:
        return audio
    return audio[activos[0] * ventana : min((activos[-1] + 1) * ventana, len(audio))]


def enrolar(args: argparse.Namespace, config: Config) -> int:
    sample_rate = config.audio.sample_rate
    device = config.audio.input_device

    print("=" * 68)
    print("ENROLAMIENTO DE VOZ")
    print("=" * 68)
    print(f"\nVas a grabar {args.frases} frases de {args.segundos:.0f} segundos.")
    print("Habla como hablas normalmente al asistente, desde donde sueles estar.")
    print("Si el ventilador suele estar encendido, dejalo encendido.\n")

    print("Descargando el modelo de locutor si hace falta...")
    embedder = SpeakerEmbedder(download_model())

    muestras: list[np.ndarray] = []
    for i in range(args.frases):
        frase = FRASES_SUGERIDAS[i % len(FRASES_SUGERIDAS)]
        print(f"\n[{i + 1}/{args.frases}] Di:  \"{frase}\"")
        for cuenta in (3, 2, 1):
            print(f"    {cuenta}...", end="\r", flush=True)
            time.sleep(0.6)
        print("    GRABANDO        ", end="\r", flush=True)

        audio = recortar_silencio(grabar(args.segundos, sample_rate, device), sample_rate)
        duracion = len(audio) / sample_rate
        pico = float(np.max(np.abs(audio))) if audio.size else 0.0

        if duracion < 0.6:
            print(f"    descartada: solo {duracion:.2f} s de voz. Repite.")
            continue
        if pico < 0.02:
            print(f"    descartada: nivel muy bajo (pico {pico:.3f}). ¿Micrófono correcto?")
            continue

        muestras.append(audio)
        print(f"    ok  ({duracion:.1f} s, pico {pico:.2f})            ")

    if len(muestras) < 3:
        print("\nHacen falta al menos 3 grabaciones utiles. No se ha guardado nada.")
        return 1

    centroide = build_profile(embedder, muestras, sample_rate)
    vectores = [embedder.encode(m, sample_rate) for m in muestras]
    propios = [float(np.dot(v, centroide)) for v in vectores]

    destino = Path(args.salida or config.speaker.profile)
    destino.write_text(
        json.dumps(
            {
                "centroid": centroide.tolist(),
                "samples": len(muestras),
                "self_scores": propios,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 68)
    print(f"Perfil guardado en {destino}  ({len(muestras)} frases)")
    print(f"Coseno de tus propias frases: min {min(propios):.3f}  media "
          f"{np.mean(propios):.3f}")

    # El umbral se deriva de SUS grabaciones, no de un valor de catalogo. Se
    # deja un margen generoso por debajo del peor caso propio: el error caro es
    # rechazarle a el, no aceptar de mas.
    sugerido = max(0.30, round(min(propios) - 0.20, 2))
    print(f"\nUmbral recomendado para TU voz y TU sala: {sugerido:.2f}")
    print("\nPara activarlo, en config.local.yaml:\n")
    print("speaker:")
    print("  enabled: true")
    print(f"  threshold: {sugerido:.2f}")
    print("\nDespues arranca con -v y mira las lineas 'locutor:' de cada turno.")
    print("Si te ignora a ti, BAJA el umbral. Si atiende a la tele, subelo.")
    return 0


def probar(config: Config) -> int:
    """Graba una frase y dice cuanto se parece al perfil guardado."""
    from asistente.audio.speaker import SpeakerGate

    perfil = Path(config.speaker.profile)
    if not perfil.is_file():
        print(f"No existe {perfil}. Ejecuta el enrolamiento primero.")
        return 1

    embedder = SpeakerEmbedder(download_model())
    gate = SpeakerGate.from_profile(embedder, perfil, config.speaker.threshold)

    print(f"Perfil: {perfil}   umbral: {config.speaker.threshold:.2f}")
    print("Di algo cuando aparezca GRABANDO (Ctrl+C para salir).\n")
    try:
        while True:
            for cuenta in (3, 2, 1):
                print(f"  {cuenta}...", end="\r", flush=True)
                time.sleep(0.6)
            print("  GRABANDO   ", end="\r", flush=True)
            audio = recortar_silencio(
                grabar(3.0, config.audio.sample_rate, config.audio.input_device),
                config.audio.sample_rate,
            )
            veredicto = gate.check(audio, config.audio.sample_rate)
            marca = "ACEPTA" if veredicto.accepted else "RECHAZA"
            print(f"  {marca}  {veredicto.reason}                    ")
    except KeyboardInterrupt:
        print("\nadios")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frases", type=int, default=6, help="cuantas grabar (por defecto 6)")
    parser.add_argument("--segundos", type=float, default=3.0, help="duracion de cada una")
    parser.add_argument("--salida", type=str, default=None, help="donde guardar el perfil")
    parser.add_argument("--probar", action="store_true", help="probar el perfil ya existente")
    args = parser.parse_args()

    config = Config.load(ROOT / "config.yaml")
    return probar(config) if args.probar else enrolar(args, config)


if __name__ == "__main__":
    raise SystemExit(main())
