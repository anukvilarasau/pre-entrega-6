"""
research_tools.py — Las 2 herramientas del agente investigador.

1. buscar_en_memoria → busca en ChromaDB lo que ya se trajo de la web antes.
2. buscar_en_web     → busca en la web con Tavily y GUARDA lo encontrado en
                       la memoria, para la próxima vez.

Reglas de diseño (las mismas de la Pre-entrega 5):
- Esquema de entrada Pydantic (args_schema): el LLM ve qué argumentos existen
  y sus límites.
- Docstring muy descriptivo: es lo que el LLM lee para decidir CUÁNDO usarla.
- Los errores vuelven como TEXTO al LLM (para que razone); nunca rompen el programa.

Cada herramienta devuelve DOS cosas (response_format="content_and_artifact"):
- content: texto legible para el LLM;
- artifact: la lista de `Fuente` (Pydantic) de cada dato, con su origen
  ("memoria" o "web"). El nodo investigador arma el Aporte con estos
  artifacts, NO con lo que el LLM diga: así las fuentes no se pueden inventar.

El buscador web se "inyecta": en modo real es Tavily; en modo mock y en los
tests es `BuscadorMock`, que lee resultados prefijados de un JSON.
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from langchain_core.tools import BaseTool, StructuredTool, ToolException
from pydantic import BaseModel, ConfigDict, Field

import config
from agents.memory import MemoriaBusquedas, ResultadoBusquedaMemoria, ResultadoWeb, id_memoria, normalizar
from state import Fuente, Tema, cantidad

log = logging.getLogger(__name__)

LARGO_EXTRACTO: int = 700  # caracteres de cada resultado que ve el LLM


# ---------------------------------------------------------------------------
# Buscadores web (el real y el simulado)
# ---------------------------------------------------------------------------
class BuscadorWeb(Protocol):
    """Cualquier cosa que sepa buscar en la web y devolver ResultadoWeb."""

    async def buscar(self, consulta: str) -> list[ResultadoWeb]: ...


class BuscadorTavily:
    """Búsqueda web real con Tavily (paquete langchain-tavily).

    `TavilySearch` reemplaza a `TavilySearchResults` (langchain-community),
    que es el que nombra la consigna y hoy está deprecado.
    """

    def __init__(self, api_key: str = config.TAVILY_API_KEY, max_resultados: int = config.TAVILY_MAX_RESULTADOS) -> None:
        from langchain_tavily import TavilySearch

        self._tool = TavilySearch(max_results=max_resultados, tavily_api_key=api_key, search_depth="basic")

    async def buscar(self, consulta: str) -> list[ResultadoWeb]:
        try:
            crudo: dict[str, Any] = await self._tool.ainvoke({"query": consulta})
        except ToolException as e:  # langchain-tavily lanza esto cuando no hay resultados
            if "No search results" in str(e):
                return []
            raise
        if isinstance(crudo, str):  # algunos errores vuelven como texto
            raise RuntimeError(crudo)
        return [
            ResultadoWeb(url=r["url"], titulo=str(r.get("title", "")), contenido=str(r.get("content", "")))
            for r in crudo.get("results", [])
            if r.get("url")
        ]


class BuscadorMock:
    """Tavily simulado: resultados prefijados en data/mock_busquedas.json.

    Elige los resultados cuyas palabras clave aparecen en la consulta. Lleva la
    cuenta de las llamadas (para demostrar que la memoria evita ir a la web) y
    se puede configurar para fallar (escenario "Tavily falla").
    """

    def __init__(self, ruta: Path = config.BUSQUEDAS_MOCK, fallar: bool = False) -> None:
        self._entradas: list[dict[str, Any]] = json.loads(ruta.read_text(encoding="utf-8"))["resultados"]
        self.fallar = fallar
        self.llamadas: list[str] = []

    async def buscar(self, consulta: str) -> list[ResultadoWeb]:
        self.llamadas.append(consulta)
        if self.fallar:
            raise ConnectionError("Tavily no respondió (simulado): 503 Service Unavailable")
        texto = normalizar(consulta)
        puntajes: list[tuple[int, dict[str, Any]]] = []
        for e in self._entradas:
            if normalizar(e["destino"]) not in texto:
                continue
            puntaje = sum(1 for p in e["palabras"] if normalizar(p) in texto)
            if puntaje:
                puntajes.append((puntaje, e))
        puntajes.sort(key=lambda par: -par[0])
        return [
            ResultadoWeb(url=e["url"], titulo=e["titulo"], contenido=e["contenido"])
            for _, e in puntajes[: config.TAVILY_MAX_RESULTADOS]
        ]


# ---------------------------------------------------------------------------
# Esquemas de entrada (lo que ve el LLM)
# ---------------------------------------------------------------------------
_DESC_TEMA = (
    "Tema de la búsqueda: 'pasajes' (vuelos o micros), 'alojamiento', 'clima', "
    "'comida', 'actividades', 'cambio' (cotización del dólar) u 'otro'."
)


class BuscarEnMemoriaInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    consulta: str = Field(min_length=3, max_length=300, description="Qué buscás, en castellano y concreto.")
    destino: str = Field(min_length=2, max_length=80, description="Ciudad de destino, por ejemplo 'Bariloche'.")
    tema: Tema = Field(description=_DESC_TEMA)


class BuscarEnWebInput(BuscarEnMemoriaInput):
    forzar: bool = Field(
        default=False,
        description=(
            "Dejalo en false. Ponelo en true SOLO si en la memoria hay datos de este tema pero NO "
            "sirven (por ejemplo, no traen precios) y necesitás buscar de nuevo en la web."
        ),
    )


# ---------------------------------------------------------------------------
# Formateo de resultados
# ---------------------------------------------------------------------------
def _formatear_memoria(r: ResultadoBusquedaMemoria, consulta: str, ahora: datetime) -> str:
    lineas = [f"MEMORIA — {cantidad(len(r.utiles), 'dato vigente', 'datos vigentes')} para '{consulta}':"]
    for reg in r.utiles:
        extracto = reg.texto[:LARGO_EXTRACTO] + ("..." if len(reg.texto) > LARGO_EXTRACTO else "")
        lineas.append(
            f"\n[{reg.id}] origen=memoria | tema={reg.tema} | buscado hace {cantidad(reg.antiguedad_dias(ahora), 'día', 'días')} "
            f"({reg.fecha.date().isoformat()}) | url={reg.url}\n{extracto}"
        )
    return "\n".join(lineas)


def _fuentes_de_memoria(r: ResultadoBusquedaMemoria) -> list[Fuente]:
    return [
        Fuente(ref=reg.id, origen="memoria", tema=reg.tema, url=reg.url, fecha_busqueda=reg.fecha.date())  # type: ignore[arg-type]
        for reg in r.utiles
    ]


# ---------------------------------------------------------------------------
# Fábrica de herramientas
# ---------------------------------------------------------------------------
def crear_herramientas_investigador(memoria: MemoriaBusquedas, buscador: BuscadorWeb) -> list[BaseTool]:
    """Crea las 2 herramientas del investigador, conectadas a ESTA memoria y ESTE buscador."""

    async def buscar_en_memoria(consulta: str, destino: str, tema: Tema) -> tuple[str, list[Fuente]]:
        try:
            ahora = datetime.now().astimezone()
            r = await asyncio.to_thread(memoria.buscar, consulta, destino, tema, 4, ahora)
            if r.hay_utiles:
                return _formatear_memoria(r, consulta, ahora), _fuentes_de_memoria(r)
            motivos: list[str] = []
            if r.descartados_por_antiguedad:
                motivos.append(f"{cantidad(r.descartados_por_antiguedad, 'dato vencido', 'datos vencidos')} "
                               f"(más de {memoria.max_dias} días)")
            if r.descartados_por_relevancia:
                motivos.append(f"{cantidad(r.descartados_por_relevancia, 'dato reciente', 'datos recientes')} "
                               "pero poco parecido a la consulta")
            detalle = "; ".join(motivos) or "no hay nada guardado sobre este destino y tema"
            return (
                f"SIN RESULTADOS EN MEMORIA para '{consulta}' [destino={destino}, tema={tema}]: {detalle}. "
                "Buscá en la web con buscar_en_web."
            ), []
        except Exception as e:  # la base local dañada, disco lleno, etc.
            log.warning("buscar_en_memoria falló: %s", e)
            return f"ERROR al leer la memoria: {type(e).__name__}: {e}. Buscá en la web con buscar_en_web.", []

    async def buscar_en_web(consulta: str, destino: str, tema: Tema, forzar: bool = False) -> tuple[str, list[Fuente]]:
        ahora = datetime.now().astimezone()
        # Guarda: si la memoria ya tiene datos vigentes, no se gasta una búsqueda web.
        if not forzar:
            try:
                previo = await asyncio.to_thread(memoria.buscar, consulta, destino, tema, 4, ahora)
                if previo.hay_utiles:
                    log.info("buscar_en_web: evitada, datos vigentes en memoria: %d", len(previo.utiles))
                    return (
                        "NO SE BUSCÓ EN LA WEB: la memoria ya tiene datos vigentes sobre esto "
                        "(si no sirven, repetí con forzar=true).\n\n" + _formatear_memoria(previo, consulta, ahora)
                    ), _fuentes_de_memoria(previo)
            except Exception as e:  # si la memoria falla, se sigue a la web igual
                log.warning("No se pudo consultar la memoria antes de la web: %s", e)

        try:
            resultados = await buscador.buscar(f"{consulta} {destino}" if normalizar(destino) not in normalizar(consulta) else consulta)
        except Exception as e:  # Tavily caído, clave inválida, sin internet, cuota agotada...
            log.warning("buscar_en_web falló: %s", e)
            return (
                f"ERROR al buscar en la web (Tavily): {type(e).__name__}: {e}. "
                "No inventes datos: informá que la búsqueda web no está disponible."
            ), []

        if not resultados:
            return f"SIN RESULTADOS EN LA WEB para '{consulta}'. Probá con otras palabras.", []

        try:
            ids = await asyncio.to_thread(memoria.guardar, resultados, destino, tema, consulta, ahora)
        except Exception as e:  # que falle el guardado no invalida los resultados
            log.warning("No se pudo guardar en memoria: %s", e)
            ids = [id_memoria(r.url, destino, tema) for r in resultados]

        lineas = [f"WEB (Tavily) — {cantidad(len(resultados), 'resultado')} para '{consulta}' (guardados en memoria):"]
        for r, id_ in zip(resultados, ids, strict=True):
            extracto = r.contenido[:LARGO_EXTRACTO] + ("..." if len(r.contenido) > LARGO_EXTRACTO else "")
            lineas.append(f"\n[{r.url}] origen=web | tema={tema} | guardado como {id_}\n{r.titulo}\n{extracto}")
        fuentes = [Fuente(ref=r.url, origen="web", tema=tema, url=r.url, fecha_busqueda=ahora.date()) for r in resultados]
        return "\n".join(lineas), fuentes

    herramienta_memoria = StructuredTool.from_function(
        coroutine=buscar_en_memoria,
        name="buscar_en_memoria",
        args_schema=BuscarEnMemoriaInput,
        response_format="content_and_artifact",
        description=(
            "Busca en la MEMORIA LOCAL (ChromaDB) datos de viaje que ya se trajeron de la web antes. "
            "Usala SIEMPRE PRIMERO, antes de buscar_en_web: es gratis e instantánea. "
            f"Solo devuelve datos del mismo destino y tema, guardados hace {config.MEMORIA_MAX_DIAS} días o menos "
            "y parecidos a tu consulta. Cada dato trae su ID (mem-...), la URL original y la fecha de búsqueda. "
            "Si responde 'SIN RESULTADOS EN MEMORIA', usá buscar_en_web."
        ),
    )
    herramienta_web = StructuredTool.from_function(
        coroutine=buscar_en_web,
        name="buscar_en_web",
        args_schema=BuscarEnWebInput,
        response_format="content_and_artifact",
        description=(
            "Busca en la WEB con Tavily información actual de viajes: precios de pasajes (avión o micro), "
            "alojamiento, clima, actividades o cotización del dólar. Usala SOLO si buscar_en_memoria no trajo "
            "nada útil. Todo lo que encuentra queda guardado en la memoria. Cada resultado trae su URL: "
            "citala siempre. Si responde con 'ERROR', no inventes datos."
        ),
    )
    return [herramienta_memoria, herramienta_web]
