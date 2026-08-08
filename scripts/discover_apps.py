"""Descubre las aplicaciones instaladas y las escribe en config.local.yaml.

Enumera apps de la Microsoft Store, apps de escritorio y juegos de Steam, y
genera el bloque `apps:` listo para usar. Sustituye a mantener la allowlist a
mano, que no escala.

Uso:
    python scripts/discover_apps.py                  # ver que encuentra
    python scripts/discover_apps.py --write          # anadirlo a config.local.yaml
    python scripts/discover_apps.py --filter steam   # solo lo que coincida
    python scripts/discover_apps.py --no-steam       # sin juegos

SEGURIDAD: lo que se escriba aqui pasa a ser la allowlist, o sea lo unico que
el asistente puede abrir. Revisa la lista antes de --write; con --limit y
--filter es facil quedarse solo con lo que de verdad usas.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

LOCAL_CONFIG = ROOT / "config.local.yaml"


def _yaml_quote(value: str) -> str:
    """Comillas simples con escapado YAML. Las rutas de Windows llevan barras
    invertidas, que en comillas dobles se interpretarian como escapes."""
    return "'" + value.replace("'", "''") + "'"


def main() -> int:
    parser = argparse.ArgumentParser(description="Descubre las aplicaciones instaladas")
    parser.add_argument("--write", action="store_true", help="escribir en config.local.yaml")
    parser.add_argument("--no-steam", action="store_true", help="omitir los juegos de Steam")
    parser.add_argument("--filter", default="", help="solo las que contengan este texto")
    parser.add_argument("--limit", type=int, default=0, help="quedarse con las N primeras")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)-7s %(name)s: %(message)s",
    )

    from asistente.discovery import discover_all
    from asistente.router.text import normalize

    print("Buscando aplicaciones instaladas...\n")
    apps = discover_all(include_steam=not args.no_steam)

    if args.filter:
        needle = normalize(args.filter)
        apps = [a for a in apps if needle in normalize(a.name) or needle in normalize(a.source)]
    if args.limit > 0:
        apps = apps[: args.limit]

    if not apps:
        print("No se encontro ninguna aplicacion.")
        if sys.platform != "win32":
            print("(El descubrimiento solo funciona en Windows.)")
        return 1

    por_fuente: dict[str, int] = {}
    for app in apps:
        por_fuente[app.source] = por_fuente.get(app.source, 0) + 1

    print(f"{len(apps)} aplicaciones encontradas:")
    for source, count in sorted(por_fuente.items(), key=lambda kv: -kv[1]):
        print(f"    {count:>4}  {source}")
    print()

    etiquetas = {
        "steam": "[juego]",
        "start_apps": "[store]",
        "start_menu": "[atajo]",
        "app_paths": "[exe]  ",
    }
    for app in apps:
        etiqueta = etiquetas.get(app.source, "       ")
        alias = f"  (alias: {', '.join(app.aliases)})" if app.aliases else ""
        print(f"  {etiqueta} {app.name}{alias}")

    bloque = _render_yaml(apps)

    if not args.write:
        print("\n" + "=" * 70)
        print("Para anadirlo a config.local.yaml:")
        print("  python scripts/discover_apps.py --write")
        print("\nRevisa antes la lista: esto pasa a ser la allowlist, o sea lo unico")
        print("que el asistente puede abrir. Acotala con --filter y --limit.")
        return 0

    _write_local_config(bloque, len(apps))
    return 0


def _render_yaml(apps: list) -> str:  # noqa: ANN001
    lines = ["apps:"]
    for app in apps:
        lines.append(f"  {app.slug}:")
        lines.append(f"    command: {_yaml_quote(app.command)}")
        if app.process:
            lines.append(f"    process: {_yaml_quote(app.process)}")
        alias = [app.name, *app.aliases]
        lines.append(f"    aliases: [{', '.join(_yaml_quote(a) for a in alias)}]")
    return "\n".join(lines)


def _write_local_config(bloque: str, count: int) -> None:
    """Reemplaza la seccion `apps:` de config.local.yaml, conservando el resto.

    Se conserva lo demas a proposito: ahi viven la ganancia del microfono y el
    indice del dispositivo, que costaron su rato de afinar.
    """
    encabezado = (
        "# Generado por scripts/discover_apps.py\n"
        "# Esta seccion se REEMPLAZA al volver a ejecutarlo con --write.\n"
        "# El resto del fichero se conserva.\n"
    )

    if LOCAL_CONFIG.is_file():
        existente = LOCAL_CONFIG.read_text(encoding="utf-8")
        respaldo = LOCAL_CONFIG.with_suffix(".yaml.bak")
        respaldo.write_text(existente, encoding="utf-8")
        print(f"\nCopia de seguridad en {respaldo.name}")
        resto = _strip_apps_section(existente)
    else:
        resto = ""

    contenido = f"{resto.rstrip()}\n\n{encabezado}{bloque}\n" if resto.strip() else f"{encabezado}{bloque}\n"
    LOCAL_CONFIG.write_text(contenido, encoding="utf-8")

    print(f"Escritas {count} aplicaciones en {LOCAL_CONFIG.name}")
    print("\nCompruebalo con:")
    print("  python scripts/diagnose_apps.py")


def _strip_apps_section(text: str) -> str:
    """Quita el bloque `apps:` de nivel superior y sus comentarios generados."""
    lines = text.splitlines()
    salida: list[str] = []
    dentro = False
    for line in lines:
        if line.startswith("apps:"):
            dentro = True
            continue
        if dentro:
            # La seccion termina en la primera linea de nivel superior.
            if line and not line[0].isspace():
                dentro = False
            else:
                continue
        if line.startswith("# Generado por scripts/discover_apps.py"):
            dentro = False
            continue
        if line.startswith("# Esta seccion se REEMPLAZA") or line.startswith("# El resto del fichero"):
            continue
        salida.append(line)
    return "\n".join(salida)


if __name__ == "__main__":
    raise SystemExit(main())
