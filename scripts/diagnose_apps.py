"""Comprueba que cada app de la allowlist se puede ABRIR y CERRAR.

Sin esto, cada aplicacion mal configurada se descubre de una en una hablandole
al asistente y viendola fallar. Aqui salen todas de golpe, con la ruta real
donde se encontro cada una y con el proceso que se mataria al cerrarla.

Son dos preguntas distintas y hasta ahora solo se respondia la primera: una app
puede abrirse perfectamente y no cerrarse nunca, porque abrir necesita una ruta
y cerrar necesita el nombre del proceso, que no siempre coincide con nada de lo
que hay escrito en la config.

Uso:
    python scripts/diagnose_apps.py            # comprobar la allowlist
    python scripts/diagnose_apps.py --fix      # ademas, sugerir rutas encontradas
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

if TYPE_CHECKING:
    from asistente.config import Config

#: Sitios donde los instaladores de Windows suelen dejar las cosas, para
#: sugerir una ruta cuando el registro no sabe nada.
_SEARCH_HINTS = (
    r"%LOCALAPPDATA%\Microsoft\WindowsApps",
    r"%LOCALAPPDATA%\Programs",
    r"%APPDATA%",
    r"%ProgramFiles%",
    r"%ProgramFiles(x86)%",
)


def _guess_paths(name: str) -> list[Path]:
    """Busca <name>.exe en las carpetas habituales, sin recursion profunda."""
    import os

    found: list[Path] = []
    for hint in _SEARCH_HINTS:
        root = Path(os.path.expandvars(hint))
        if "%" in str(root) or not root.is_dir():
            continue
        try:
            # Dos niveles: suficiente para Programs\Spotify\Spotify.exe sin
            # recorrer medio disco.
            found.extend(root.glob(f"{name}.exe"))
            found.extend(root.glob(f"*/{name}.exe"))
            found.extend(root.glob(f"*/*/{name}.exe"))
        except OSError:
            continue
    return found[:3]


def _report_closing(config: Config) -> None:
    """Que se mataria al decir "cierra X", de lo que hay corriendo AHORA.

    Solo puede responderse con la lista real de procesos, asi que fuera de
    Windows no se imprime nada en vez de imprimir una suposicion.
    """
    from asistente.skills.apps import candidate_images, pick_running
    from asistente.skills.winproc import can_list_processes, running_images

    print("\n" + "=" * 70)
    print("CERRAR  (que proceso se mataria, de lo que corre ahora mismo)")
    print("=" * 70)

    if not can_list_processes():
        print("Solo se puede comprobar en Windows.")
        return

    vivos = running_images()
    if not vivos:
        print("No se pudo listar los procesos (`tasklist`). Se cerraria a ciegas.")
        return

    abiertas = 0
    for name, spec in sorted(config.apps.items()):
        objetivo = pick_running(candidate_images(name, spec), vivos)
        if objetivo is None:
            continue
        abiertas += 1
        print(f"  {name:<24} -> {objetivo}")

    print(f"\n{abiertas} de {len(config.apps)} estan abiertas y se pueden cerrar.")
    print("Las que no salen es que no estan corriendo, no que esten mal configuradas.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Comprueba la allowlist de aplicaciones")
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--fix", action="store_true", help="sugerir rutas para las que fallen")
    args = parser.parse_args()

    from asistente.config import Config
    from asistente.discovery.auto import augment_config
    from asistente.skills.launcher import can_launch, resolve_executable

    config = augment_config(Config.load(args.config))
    if not config.apps:
        print("La allowlist de aplicaciones esta vacia.")
        return 0

    print(f"Comprobando {len(config.apps)} aplicaciones de {args.config.name}\n")
    ok: list[str] = []
    failed: list[tuple[str, str]] = []

    for name, spec in sorted(config.apps.items()):
        resolved = resolve_executable(spec.command)
        if resolved is not None:
            ok.append(name)
            print(f"  OK    {name:<14} {spec.command:<12} -> {resolved}")
        elif can_launch(spec.command):
            # Una app de la Store o un juego de Steam no tienen ejecutable
            # localizable y se abren igual. Decir "FALLA" aqui era una alarma
            # falsa que mandaba a buscar una ruta que no existe.
            ok.append(name)
            print(f"  OK    {name:<14} {spec.command}")
        else:
            failed.append((name, spec.command))
            print(f"  FALLA {name:<14} {spec.command:<12} -> no se encuentra")

    print(f"\n{len(ok)} de {len(config.apps)} se pueden abrir.")
    _report_closing(config)

    if not failed:
        return 0

    print("\n" + "=" * 70)
    print("NO LOCALIZADAS")
    print("=" * 70)
    print("Pueden funcionar igual si son apps de la Microsoft Store: el asistente")
    print("prueba `start` como ultimo recurso. Si no, pon la ruta completa.\n")

    for name, command in failed:
        print(f"  {name} (busca '{command}')")
        if args.fix:
            for guess in _guess_paths(command):
                print(f"      posible: {guess}")

    print("\nPon las rutas en config.local.yaml (NO en config.yaml, que es del repo):\n")
    print("  apps:")
    for name, _ in failed[:2]:
        print(f"    {name}:")
        print(f"      command: 'C:\\ruta\\completa\\a\\{name}.exe'")
        print(f"      process: {name}")
    if not args.fix:
        print("\nPara que intente adivinar las rutas:  python scripts/diagnose_apps.py --fix")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
