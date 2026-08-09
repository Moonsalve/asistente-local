"""Diagnostico del control de volumen.

Responde a "el volumen no funciona" sin adivinar: hay cuatro mecanismos
independientes y cada uno falla por su cuenta. Este script los prueba uno a uno
y dice cual esta roto y por que.

    1. Teclas multimedia (SendInput)     -> mute y respaldo de subir/bajar
    2. Volumen maestro (IAudioEndpointVolume) -> el volumen del PC
    3. Mezclador por aplicacion (ISimpleAudioVolume) -> el volumen de Spotify
    4. Web API de Spotify                -> volumen cuando suena en otro aparato

Uso (en el PC Windows, con el venv activado):
    python scripts/diagnose_volume.py

No cambia nada de forma permanente: donde escribe, restaura el valor anterior.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asistente.config import Config, Secrets  # noqa: E402
from asistente.skills import winaudio, winkeys  # noqa: E402
from asistente.skills.spotify import SpotifyClient  # noqa: E402

OK = "  OK "
MAL = " MAL "
NA = "  -- "


def check_send_input() -> bool:
    print("\n1. TECLAS MULTIMEDIA (SendInput)")
    size = winkeys.input_struct_size()
    expected = winkeys.EXPECTED_INPUT_SIZE
    if size != expected:
        print(f"{MAL} sizeof(INPUT) = {size}, Windows espera {expected}")
        print("      SendInput rechaza toda llamada con cbSize incorrecto: fallan")
        print("      el silencio y todas las teclas de reproduccion.")
        return False
    print(f"{OK} sizeof(INPUT) = {size} bytes")

    if not winkeys.IS_WINDOWS:
        print(f"{NA} no estamos en Windows; las pulsaciones no se prueban")
        return True

    # VOLUME_MUTE es reversible: se pulsa dos veces y todo queda igual.
    ida = winkeys.press(winkeys.VirtualKey.VOLUME_MUTE)
    vuelta = winkeys.press(winkeys.VirtualKey.VOLUME_MUTE)
    if ida and vuelta:
        print(f"{OK} pulsacion de prueba aceptada (mute y de vuelta)")
        return True
    print(f"{MAL} SendInput no inyecto los eventos.")
    print("      Causa tipica: la ventana en primer plano corre como")
    print("      administrador y el asistente no. Windows bloquea la inyeccion")
    print("      de un proceso menos privilegiado (UIPI).")
    return False


def check_master() -> bool:
    print("\n2. VOLUMEN MAESTRO DEL PC (IAudioEndpointVolume)")
    print(f"      pycaw {winaudio.pycaw_version()}")
    current = winaudio.master_percent()
    if current is None:
        print(f"{MAL} no se pudo leer: {winaudio.last_error()}")
        print("      Sin esto el volumen solo se mueve a saltos de 2% con las teclas.")
        print("      Si el error menciona 'AudioDevice' o 'Activate', pycaw ha")
        print("      vuelto a cambiar la forma de GetSpeakers(): ver winaudio._master.")
        return False
    print(f"{OK} volumen actual: {current}%   silenciado: {winaudio.master_muted()}")

    objetivo = 30 if current > 50 else 70
    if not winaudio.set_master_percent(objetivo):
        print(f"{MAL} no se pudo escribir: {winaudio.last_error()}")
        return False
    leido = winaudio.master_percent()
    winaudio.set_master_percent(current)
    if leido != objetivo:
        print(f"{MAL} se escribio {objetivo}% pero se leyo {leido}%")
        return False
    print(f"{OK} escritura verificada ({objetivo}%) y valor original restaurado")
    return True


def check_mixer(process_names: tuple[str, ...]) -> bool:
    """Informativo. El volumen de Spotify NO se controla por aqui —se decidio
    que "el volumen de Spotify" es el de dentro de la app, no el del mezclador—
    pero ver a Spotify en esta lista confirma que esta sonando de verdad, que es
    la causa mas comun de que el punto 4 falle."""
    print("\n3. MEZCLADOR DE WINDOWS (informativo: quien tiene audio abierto)")
    sessions = winaudio.list_sessions()
    if not sessions:
        print(f"{NA} el mezclador esta vacio: {winaudio.last_error() or 'sin sesiones'}")
        print("      Solo aparecen aplicaciones con audio abierto.")
        return True

    buscados = {n.lower() for n in process_names}
    visto = False
    for s in sessions:
        marca = " "
        if s.process.lower() in buscados:
            marca, visto = "*", True
        print(f"       {marca} {s.process:<32} {s.percent:>3}%  silenciada={s.muted}")

    if visto:
        print(f"{OK} Spotify esta sonando en este PC")
    else:
        print(f"{NA} Spotify no tiene audio abierto aqui (puede estar en el movil)")
    return True


def check_web_api(config: Config) -> bool:
    """Esta es LA via del volumen de Spotify. Si falla, "baja el volumen de
    Spotify" no tiene alternativa: no se degrada al mezclador a proposito,
    porque seria hacer justo lo que no se pidio y sin avisar."""
    print("\n4. WEB API DE SPOTIFY  <-- la via del volumen de Spotify")
    client = SpotifyClient(config, Secrets())
    if not client.available:
        print(f"{NA} Spotify desactivado o sin client_id; no aplica")
        return True
    if not client.connect():
        print(f"{MAL} no se pudo conectar (ver el log con -v)")
        return False

    volume = client.get_volume_percent()
    if volume is None:
        print(f"{MAL} sin dispositivo activo, o el dispositivo no reporta volumen.")
        print("      Normal si Spotify lleva rato cerrado. Dale al play y repite.")
        return False
    print(f"{OK} volumen del dispositivo activo: {volume}%")

    objetivo = 30 if volume > 50 else 70
    if not client.set_volume_percent(objetivo):
        print(f"{MAL} el dispositivo rechazo el cambio de volumen.")
        print("      Spotify devuelve 403 en el reproductor web y en varios")
        print("      altavoces de Connect. No es un fallo del asistente.")
        return False
    client.set_volume_percent(volume)
    print(f"{OK} escritura verificada y volumen original restaurado")
    return True


def main() -> int:
    config = Config.load(ROOT / "config.yaml")
    procesos = config.spotify.process_names

    print("=" * 72)
    print("DIAGNOSTICO DEL CONTROL DE VOLUMEN")
    print("=" * 72)

    resultados = {
        "teclas multimedia": check_send_input(),
        "volumen del PC": check_master(),
        "mezclador (informativo)": check_mixer(procesos),
        "volumen de Spotify": check_web_api(config),
    }

    print("\n" + "=" * 72)
    for nombre, ok in resultados.items():
        print(f"{OK if ok else MAL} {nombre}")

    if not resultados["volumen del PC"] and not resultados["teclas multimedia"]:
        print("\nEl volumen del PC no tiene ninguna via disponible. Empieza por el")
        print("punto 2: sin pycaw/comtypes instalados en el venv activo, no hay nada")
        print("que hacer (pip install -e .).")
    elif not resultados["volumen de Spotify"]:
        print("\nEl volumen de Spotify no tiene via alternativa: se decidio que sea")
        print("siempre el mando de dentro de la app, y no se degrada al mezclador")
        print("porque seria cambiar otro volumen distinto sin avisar. Pon musica a")
        print("sonar en Spotify y repite.")

    return 0 if all(resultados.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
