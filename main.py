"""
main.py — Asesor de viajes multi-agente desde la consola.

Uso:
    python main.py "Quiero ir 7 días a Bariloche en enero..."     # una pregunta (claves reales)
    python main.py --mock "Quiero ir 7 días a Bariloche..."        # sin claves (LLM y Tavily simulados)
    python main.py --chat                                          # modo chat: preguntas una atrás de otra
    python main.py --mock --chat                                   # chat sin claves
    python main.py --mock --max-pasos 2 "..."                      # freno de pasos bajo (escenario 8)
    python main.py --mock --fallar-web "..."                       # Tavily caído (escenario 7, solo mock)
    python main.py --mermaid                                       # imprime el diagrama del grafo

Cada respuesta muestra qué agentes intervinieron (la ruta del supervisor),
cuántos datos salieron de la memoria y cuántos de la web, y dónde quedó la
traza (traces/*.json y *.log).
"""

import argparse
import asyncio
import logging
import sys

import config
from graph import construir_grafo, consultar, crear_componentes, diagrama_mermaid
from tracing import Traza

COMANDOS_SALIR: set[str] = {"salir", "exit", "quit"}


def leer_argumentos() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Asesor de viajes multi-agente (LangGraph, topología jerárquica).")
    p.add_argument("pregunta", nargs="?", default=None, help="La pregunta (si no se usa --chat).")
    p.add_argument("--mock", action="store_true", help="Modo mock: sin claves ni llamadas pagas.")
    p.add_argument("--chat", action="store_true", help="Modo interactivo: escribí 'salir' para terminar.")
    p.add_argument("--escenario", default="", help="Etiqueta para el nombre del archivo de traza.")
    p.add_argument("--max-pasos", type=int, default=None,
                   help=f"Freno del supervisor (por defecto {config.MAX_PASOS_SUPERVISOR}).")
    p.add_argument("--fallar-web", action="store_true", help="(solo mock) simula que Tavily está caído.")
    p.add_argument("--mermaid", action="store_true", help="Imprime el diagrama Mermaid del grafo y sale.")
    p.add_argument("-v", "--verbose", action="store_true", help="Muestra los logs de cada etapa.")
    return p.parse_args()


def explicar_error(e: Exception) -> str:
    """Traduce los errores más comunes de las APIs a un mensaje claro."""
    try:
        import anthropic

        if isinstance(e, anthropic.AuthenticationError):
            return "La ANTHROPIC_API_KEY del .env es inválida (la API respondió 401)."
        if isinstance(e, anthropic.NotFoundError):
            return f"El modelo '{config.ANTHROPIC_MODEL}' no existe o no tenés acceso. Revisá ANTHROPIC_MODEL en el .env."
        if isinstance(e, anthropic.RateLimitError):
            return "La API de Claude está limitando los pedidos (429). Esperá un momento y probá de nuevo."
        if isinstance(e, (anthropic.APIConnectionError, anthropic.APITimeoutError)):
            return "No se pudo conectar con la API de Claude (sin internet o timeout)."
    except ImportError:
        pass
    return f"{type(e).__name__}: {e}"


def mostrar(traza: Traza) -> None:
    """La respuesta, y abajo qué agentes intervinieron y de dónde salieron los datos."""
    m = traza.metricas
    print(f"\n{traza.respuesta_final}\n")
    print("─" * 80)
    print(f"[ruta: {traza.texto_ruta()}]")
    print(f"[decisiones del supervisor: {m['pasos_supervisor']}"
          f"{' (la última forzada por el freno)' if m['decision_forzada_por_freno'] else ''} | "
          f"refinamientos: {m['refinamientos']} | validación: {m['validacion']}]")
    print(f"[datos: {traza.texto_origenes()} | {traza.etiqueta_busquedas()}: {m['busquedas_web']} | "
          f"modo: {traza.modo} | {traza.duracion_s} s | traza: {traza.archivo}]")
    print("─" * 80)


async def main() -> int:
    args = leer_argumentos()
    # La consola de Windows usa cp1252 por defecto y no puede imprimir "→" ni emojis.
    for flujo in (sys.stdout, sys.stderr):
        if hasattr(flujo, "reconfigure"):
            flujo.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    # Si la entrada viene por tubería (no del teclado), también se lee en UTF-8: si no, "días" llega como "dÃ­as".
    if not sys.stdin.isatty() and hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    config.configurar_logs(logging.INFO if args.verbose else logging.WARNING)
    modo_mock = args.mock or config.MODO_MOCK

    if args.fallar_web and not modo_mock:
        print("❌ --fallar-web solo funciona en modo mock (agregá --mock).", file=sys.stderr)
        return 2
    try:
        config.validar_entorno(modo_mock)
    except config.ErrorDeConfiguracion as e:
        print(f"\n❌ ERROR: {e}\n", file=sys.stderr)
        return 1

    componentes = crear_componentes(modo_mock=modo_mock, fallar_web=args.fallar_web)
    grafo = construir_grafo(componentes, max_pasos=args.max_pasos)

    if args.mermaid:
        print(diagrama_mermaid(grafo))
        return 0

    async def una(pregunta: str, escenario: str) -> bool:
        try:
            mostrar(await consultar(grafo, pregunta, escenario=escenario, modo_mock=modo_mock, max_pasos=args.max_pasos))
            return True
        except Exception as e:  # la API no responde, clave inválida, etc.
            print(f"\n❌ ERROR: {explicar_error(e)}\n", file=sys.stderr)
            return False

    if not args.chat:
        if not args.pregunta:
            print("Pasá una pregunta entre comillas, o usá --chat. Ayuda: python main.py -h", file=sys.stderr)
            return 2
        print(f"\nVos: {args.pregunta}")
        return 0 if await una(args.pregunta, args.escenario) else 1

    print(f"\nAsesor de viajes multi-agente — modo {'mock (sin claves)' if modo_mock else 'real'}.")
    print("Cada pregunta se resuelve por separado; la memoria de búsquedas (ChromaDB) se comparte entre preguntas.")
    print("Escribí 'salir' para terminar.\n")
    n = 0
    while True:
        try:
            texto = (await asyncio.to_thread(input, "Vos: ")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not texto:
            continue
        if texto.lower() in COMANDOS_SALIR:
            break
        n += 1
        await una(texto, args.escenario or f"chat_{n}")
    print("Chau. ¡Buen viaje!")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
