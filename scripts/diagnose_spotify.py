"""Comprueba lo que el asistente puede ver de TU cuenta de Spotify.

Cuando "pon mi playlist de X" no hace lo que esperas, casi siempre es una de
estas cuatro y no se distinguen oyendo el fallo:

    1. no hay conexion (falta el client id, o el token cacheado se quedo sin
       los permisos que se anadieron despues),
    2. no hay ningun dispositivo activo donde sonar,
    3. tus playlists no se ven (falta el permiso de lectura),
    4. la playlist que dices no se parece bastante a ninguna de las tuyas.

Uso:
    python scripts/diagnose_spotify.py
    python scripts/diagnose_spotify.py --buscar "gym"

No reproduce nada: solo mira.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OK = "  OK "
AVISO = "AVISO"
MAL = " MAL "


def main() -> int:
    parser = argparse.ArgumentParser(description="Que ve el asistente de tu Spotify")
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--buscar", metavar="NOMBRE", help="probar el emparejamiento de playlist")
    args = parser.parse_args()

    from asistente.config import Config, Secrets
    from asistente.skills.spotify import LIKED_LIMIT, PLAYLIST_THRESHOLD, SpotifyClient

    config = Config.load(args.config)
    secrets = Secrets()

    print("=" * 70)
    print("DIAGNOSTICO DE SPOTIFY")
    print("=" * 70)

    client = SpotifyClient(config, secrets)
    if not client.available:
        tiene_id = "puesto" if secrets.spotify_client_id else "VACIO"
        print(f"{MAL} sin configurar.")
        print("      spotify.enabled esta en", config.spotify.enabled)
        print("      SPOTIFY_CLIENT_ID en .env:", tiene_id)
        return 1

    print("\nConectando (la primera vez se abre el navegador para autorizar)...")
    if not client.connect():
        print(f"{MAL} no se pudo conectar. Mira el log de arriba.")
        return 1
    print(f"{OK} conectado")

    # --- dispositivos ------------------------------------------------------
    print("\n" + "-" * 70)
    print("DISPOSITIVOS")
    estado = client.volume_state()
    raw = client._client  # noqa: SLF001 - diagnostico: se mira por dentro a proposito
    try:
        devices = raw.devices().get("devices", [])
    except Exception as exc:  # noqa: BLE001
        print(f"{MAL} no se pudieron listar: {exc}")
        devices = []

    if not devices:
        print(f"{MAL} ninguno. Abre Spotify en el PC y vuelve a ejecutar esto.")
        print("      Sin dispositivo, TODA la Web API falla: es la causa mas comun.")
    for device in devices:
        marca = "ACTIVO" if device.get("is_active") else "      "
        volumen = device.get("volume_percent")
        print(f"  {marca}  {device.get('name', '?'):<28} {device.get('type', '?'):<12} "
              f"volumen {volumen if volumen is not None else 'n/d'}")
    if estado is not None:
        print(f"{OK} el volumen se puede leer y escribir en el dispositivo activo")
    elif devices:
        print(f"{AVISO} no se pudo leer el volumen del dispositivo activo")

    # --- tus playlists -----------------------------------------------------
    print("\n" + "-" * 70)
    print("TUS PLAYLISTS")
    playlists = client.own_playlists(force=True)
    if not playlists:
        print(f"{AVISO} no se ve ninguna.")
        print("      Si tienes playlists, falta el permiso playlist-read-private.")
        print("      Al arrancar deberia haberse reabierto el navegador para pedirlo;")
        print("      si no lo hizo, borra el token cacheado y vuelve a autorizar:")
        print("      %LOCALAPPDATA%\\asistente-local\\spotify-token.json")
    else:
        print(f"{OK} {len(playlists)} playlists visibles")
        for nombre, _ in playlists[:15]:
            print(f"       - {nombre}")
        if len(playlists) > 15:
            print(f"       ... y {len(playlists) - 15} mas")

    # --- me gusta ----------------------------------------------------------
    print("\n" + "-" * 70)
    print("TUS ME GUSTA")
    try:
        saved = raw.current_user_saved_tracks(limit=1)
        total = (saved or {}).get("total", 0)
    except Exception as exc:  # noqa: BLE001
        print(f"{MAL} no se pueden leer: {exc}")
        print("      Falta el permiso user-library-read. Borra el token y reautoriza.")
        total = 0
    if total:
        print(f"{OK} {total} canciones guardadas (se ponen en cola las {LIKED_LIMIT} ultimas)")
    elif total == 0:
        print(f"{AVISO} cero canciones guardadas: 'pon mis me gusta' no tendria que poner")

    # --- emparejamiento ----------------------------------------------------
    if args.buscar:
        print("\n" + "-" * 70)
        print(f"EMPAREJAMIENTO DE '{args.buscar}'")
        match = client.find_own_playlist(args.buscar)
        if match is None:
            print(f"{AVISO} ninguna de tus playlists pasa el umbral ({PLAYLIST_THRESHOLD:.0f}).")
            print("      Se buscaria en el catalogo publico de Spotify.")
            print("      El emparejamiento compara PALABRAS COMPLETAS: 'gym' no casa")
            print("      con 'gimnasio'. Ponle a la playlist el nombre que dices en voz alta.")
        else:
            print(f"{OK} tu playlist '{match[0]}'")

    print("\n" + "=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
