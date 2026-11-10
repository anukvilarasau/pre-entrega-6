"""
escenarios.py — Corre los 8 escenarios de prueba y arma la tabla de resultados.

Uso:
    python escenarios.py --mock              # los 8 en modo mock (sin claves)
    python escenarios.py                     # en vivo (Claude + Tavily); el 7 se corre siempre en mock
    python escenarios.py --solo 5 2 3        # solo algunos (en el orden indicado)
    python escenarios.py --tabla             # NO corre nada: arma traces/resultados.md con las trazas guardadas

Cada corrida deja su traza en traces/<n>_<escenario>_<modo>_<fecha>.json/.log.
La tabla se arma SIEMPRE leyendo esas trazas: cada fila tiene su evidencia.

Orden por defecto: 1, 5, 2, 3, 4, 6, 7, 8. El 5 (memoria) va justo después del 1 con la
misma pregunta: la segunda vez los datos salen de ChromaDB sin llamar a Tavily.
"""

import argparse
import logging
import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import config
from graph import construir_grafo, consultar, crear_componentes

DEMO = ("Quiero ir 7 días a Bariloche en enero, somos 2 personas, salimos desde Mendoza y tenemos "
        "800.000 pesos. ¿Me armás el viaje y me decís si nos alcanza?")


@dataclass(frozen=True)
class Escenario:
    n: int
    nombre: str
    pregunta: str
    demuestra: str
    max_pasos: int | None = None
    fallar_web: bool = False   # solo mock
    solo_mock: bool = False


ESCENARIOS: dict[int, Escenario] = {e.n: e for e in [
    Escenario(1, "flujo_completo", DEMO, "Delegación completa: supervisor → investigador → analista → síntesis"),
    Escenario(2, "solo_clima", "¿Qué clima hace en Bariloche en enero?", "El supervisor NO llama al analista si no hace falta"),
    Escenario(3, "no_alcanza", DEMO.replace("800.000", "400.000"), "Presupuesto que no alcanza: el analista lo detecta y se proponen alternativas"),
    Escenario(4, "fechas_invalidas", "Quiero ir a Bariloche del 16 al 10 de enero, somos 2 personas, salimos desde "
              "Mendoza y tenemos 800.000 pesos. ¿Nos alcanza?", "validar_viaje rechaza la vuelta antes que la ida"),
    Escenario(5, "memoria", DEMO, "Misma pregunta que el 1: los datos salen de ChromaDB, sin llamar a Tavily"),
    Escenario(6, "refinamiento", "¿Cómo llegar de Mendoza a Bariloche y cuánto sale el pasaje?",
              "El supervisor rechaza una búsqueda sin precios y pide rehacerla"),
    Escenario(7, "tavily_falla", DEMO, "Tavily caído: la herramienta devuelve el error y el flujo termina igual",
              fallar_web=True, solo_mock=True),
    Escenario(8, "freno_de_pasos", DEMO, "Con MAX_PASOS_SUPERVISOR=2, el freno corta y se sintetiza igual", max_pasos=2),
]}
ORDEN: list[int] = [1, 5, 2, 3, 4, 6, 7, 8]


async def correr(numeros: list[int], modo_mock: bool) -> None:
    # Una sola memoria compartida entre escenarios (la del 5 tiene que ver lo que guardó el 1).
    comunes = crear_componentes(modo_mock=modo_mock)
    for n in numeros:
        e = ESCENARIOS[n]
        mock = modo_mock or e.solo_mock
        componentes = comunes
        if e.fallar_web:  # Tavily caído, con memoria propia y vacía (si no, la memoria evitaría la web)
            componentes = crear_componentes(modo_mock=True, fallar_web=True)
        grafo = construir_grafo(componentes, max_pasos=e.max_pasos)
        print(f"\n▶ Escenario {n} · {e.nombre} ({'mock' if mock else 'real'}) — {e.demuestra}")
        traza = await consultar(grafo, e.pregunta, escenario=f"{n}_{e.nombre}", modo_mock=mock, max_pasos=e.max_pasos)
        m = traza.metricas
        print(f"  ruta: {traza.texto_ruta()}")
        print(f"  decisiones: {m['pasos_supervisor']} | refinamientos: {m['refinamientos']} | validación: "
              f"{m['validacion']} | entra: {m['presupuesto_entra']} | datos: {traza.texto_origenes()} | "
              f"{traza.etiqueta_busquedas()}: {m['busquedas_web']} | {traza.duracion_s} s → {traza.archivo}")


def ultima_traza(n: int, modo: str) -> dict | None:
    e = ESCENARIOS[n]
    archivos = sorted(config.TRACES_DIR.glob(f"{n}_{e.nombre}_{modo}_*.json"))
    return json.loads(archivos[-1].read_text(encoding="utf-8")) | {"_archivo": archivos[-1].name} if archivos else None


def tabla(prefijo: str = "") -> str:
    """Tabla markdown con la última traza de cada escenario, en mock y en real.

    `prefijo`: carpeta a anteponer a los links ("" dentro de traces/, "traces/" en el README).
    """
    filas = ["| # | Escenario | Modo | Ruta del supervisor | Decisiones | Refin. | Validación | ¿Entra? | "
             "Datos (memoria · web) | Búsquedas web | Traza |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for n in sorted(ESCENARIOS):
        for modo in ("real", "mock"):
            t = ultima_traza(n, modo)
            if t is None:
                continue
            m = t["metricas"]
            ruta = m["ruta"].replace("supervisor", "sup").replace("investigador", "inv").replace("analista", "ana") \
                            .replace("sintesis", "síntesis")
            entra = {True: "sí", False: "no", None: "—"}[m["presupuesto_entra"]]
            freno = " (freno)" if m["decision_forzada_por_freno"] else ""
            busq = f"{m['busquedas_web']}{' (simuladas)' if modo == 'mock' else ''}"
            filas.append(f"| {n} | {ESCENARIOS[n].nombre.replace('_', ' ')} | {modo} | {ruta} | "
                         f"{m['pasos_supervisor']}{freno} | {m['refinamientos']} | {m['validacion']} | {entra} | "
                         f"{m['fuentes_memoria']} · {m['fuentes_web']} | {busq} | [{t['_archivo']}]({prefijo}{t['_archivo']}) |")
    return "\n".join(filas)


def main() -> int:
    for flujo in (sys.stdout, sys.stderr):
        if hasattr(flujo, "reconfigure"):
            flujo.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    p = argparse.ArgumentParser(description="Corre los escenarios de prueba del asesor de viajes.")
    p.add_argument("--mock", action="store_true", help="Todos en modo mock (sin claves).")
    p.add_argument("--solo", type=int, nargs="+", choices=sorted(ESCENARIOS), help="Solo estos escenarios.")
    p.add_argument("--tabla", action="store_true", help="Solo arma traces/resultados.md con las trazas existentes.")
    args = p.parse_args()
    config.configurar_logs(logging.WARNING)

    if not args.tabla:
        modo_mock = args.mock or config.MODO_MOCK
        config.validar_entorno(modo_mock)
        asyncio.run(correr(args.solo or ORDEN, modo_mock))

    salida = config.TRACES_DIR / "resultados.md"
    salida.parent.mkdir(parents=True, exist_ok=True)
    salida.write_text("# Resultados de los escenarios\n\n" + tabla() + "\n", encoding="utf-8")
    print(f"\nTabla de resultados → {salida.relative_to(config.BASE_DIR)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
