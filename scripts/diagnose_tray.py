"""Por que no aparece el icono de Apolo en la bandeja.

Los cuatro motivos posibles no se distinguen mirando la barra de tareas, que es
justo el problema: el sintoma de todos es el mismo, "no hay icono".

    1. Windows 11 lo escondio en el desplegable de la flecha (^). Es el caso
       MAS frecuente y no es un fallo: los iconos nuevos entran ahi por defecto.
    2. Falta `pystray` o `Pillow` (no se instalo el extra `[tray]`).
    3. `pystray` esta pero su backend no arranca en esta maquina.
    4. Arrancaste con `--no-tray`, o con `--text`, que no lleva icono.

Este script descarta 2 y 3 de forma concluyente: pone un icono de verdad
durante unos segundos. Si lo ves, el problema era 1 o 4.

Uso:
    python scripts/diagnose_tray.py
    python scripts/diagnose_tray.py --segundos 30
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OK = "  OK "
AVISO = "AVISO"
MAL = " MAL "


def main() -> int:
    parser = argparse.ArgumentParser(description="Por que no se ve el icono de la bandeja")
    parser.add_argument(
        "--segundos",
        type=float,
        default=15.0,
        help="cuanto se deja el icono puesto para que te de tiempo a buscarlo",
    )
    args = parser.parse_args()

    # A la consola y en DEBUG: aqui interesa ver hasta el ultimo detalle de lo
    # que pasa dentro de pystray, que es lo que en el arranque normal se queda
    # en el registro.
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)-7s %(name)s: %(message)s")

    print("=" * 70)
    print("ICONO DE BANDEJA")
    print("=" * 70)

    if not _report_dependencies():
        return 1

    from asistente.tray import STARTUP_TIMEOUT_S, Tray

    print(f"\nPoniendo el icono (hasta {STARTUP_TIMEOUT_S:.0f} s para confirmarlo)...")
    tray = Tray(on_quit=lambda: None, project_dir=ROOT)
    if not tray.start():
        print(f"{MAL} no se pudo poner el icono: {tray.failure}")
        print("      Este es el motivo 3: pystray esta, pero su backend no arranca.")
        return 1

    print(f"{OK} pystray confirma que el icono esta puesto.")
    print()
    print("      MIRA AHORA LA BARRA DE TAREAS. Si no lo ves abajo a la derecha,")
    print("      pulsa la flecha (^) que despliega los iconos ocultos: Windows 11")
    print("      mete ahi los iconos nuevos por defecto.")
    print()
    print("      Para dejarlo fijo: arrastralo del desplegable a la barra, o")
    print("      Configuracion > Personalizacion > Barra de tareas > Otros iconos.")
    print()
    print(f"      Se quita solo en {args.segundos:.0f} s.")

    with contextlib.suppress(KeyboardInterrupt):
        time.sleep(args.segundos)
    tray.stop()

    print("\n" + "=" * 70)
    print("Si lo has visto, el asistente tambien lo pone: arrancalo sin --no-tray.")
    print("Si NO lo has visto ni en el desplegable, copia el log de arriba.")
    print("=" * 70)
    return 0


def _report_dependencies() -> bool:
    """Motivo 2. Se comprueban por separado: faltan de forma independiente."""
    faltan = []
    for module, paquete in (("pystray", "pystray"), ("PIL", "pillow")):
        try:
            __import__(module)
        except ImportError:
            faltan.append(paquete)
            print(f"{MAL} falta {paquete}")
        else:
            print(f"{OK} {paquete} instalado")

    if faltan:
        print()
        print("      Instalalos con:")
        print('          pip install -e ".[tray]"')
        return False
    return True


if __name__ == "__main__":
    raise SystemExit(main())
