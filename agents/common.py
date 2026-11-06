"""
common.py — Piezas que comparten el investigador y el analista.

- correr_react: corre un agente ReAct (create_react_agent) con UN solo mensaje
  de entrada y su propio techo de pasos, sin perder lo hecho si se corta.
- extraer_llamadas: lee los mensajes del ciclo ReAct y arma el registro de
  llamadas a herramientas y las fuentes que esas herramientas devolvieron.
"""

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.errors import GraphRecursionError
from langgraph.graph.state import CompiledStateGraph

import config
from state import Fuente, LlamadaHerramienta, OrquestadorState

log = logging.getLogger(__name__)

LARGO_RESULTADO_TRAZA: int = 600  # caracteres de cada resultado que quedan en la traza

NodoAsync = Callable[[OrquestadorState], Awaitable[dict[str, Any]]]


async def correr_react(agente: CompiledStateGraph, entrada_texto: str) -> tuple[list[BaseMessage], bool]:
    """Corre un agente ReAct con UN solo mensaje de entrada y devuelve (mensajes, se_cortó).

    Se usa astream(stream_mode="values"): después de cada paso llega el estado
    completo del subagente. Si el ciclo supera RECURSION_LIMIT_ESPECIALISTA,
    LangGraph lanza GraphRecursionError y nos quedamos con la última foto,
    así no se pierde lo que el agente alcanzó a hacer.
    """
    entrada = {"messages": [HumanMessage(content=entrada_texto)]}
    mensajes: list[BaseMessage] = []
    try:
        async for foto in agente.astream(
            entrada, {"recursion_limit": config.RECURSION_LIMIT_ESPECIALISTA}, stream_mode="values"
        ):
            mensajes = list(foto["messages"])
        return mensajes, False
    except GraphRecursionError:
        log.warning("[%s] cortado por RECURSION_LIMIT_ESPECIALISTA=%d", agente.name, config.RECURSION_LIMIT_ESPECIALISTA)
        return mensajes, True


def extraer_llamadas(mensajes: list[BaseMessage]) -> tuple[list[LlamadaHerramienta], list[Fuente], dict[str, dict[str, Any]]]:
    """Recorre los mensajes del ciclo ReAct.

    Devuelve: las llamadas (en orden), las fuentes que devolvieron las
    herramientas (artifacts) y, para cada tool_call_id, los argumentos usados.
    """
    pedidos: dict[str, dict[str, Any]] = {}
    for m in mensajes:
        if isinstance(m, AIMessage):
            for tc in m.tool_calls:
                pedidos[tc["id"] or ""] = tc

    llamadas: list[LlamadaHerramienta] = []
    fuentes: list[Fuente] = []
    argumentos_de: dict[str, dict[str, Any]] = {}
    for m in mensajes:
        if not isinstance(m, ToolMessage):
            continue
        pedido = pedidos.get(m.tool_call_id, {})
        argumentos = dict(pedido.get("args", {}))
        texto = str(m.content)
        llamadas.append(LlamadaHerramienta(
            herramienta=m.name or str(pedido.get("name", "?")),
            argumentos=argumentos,
            resultado=texto[:LARGO_RESULTADO_TRAZA] + ("..." if len(texto) > LARGO_RESULTADO_TRAZA else ""),
        ))
        if isinstance(m.artifact, list):
            fuentes.extend(f for f in m.artifact if isinstance(f, Fuente))
        argumentos_de[m.tool_call_id] = argumentos
    return llamadas, fuentes, argumentos_de


def respuesta_final(mensajes: list[BaseMessage], defecto: str) -> str:
    """El último mensaje del LLM sin pedidos de herramientas (su respuesta final)."""
    finales = [m for m in mensajes if isinstance(m, AIMessage) and not m.tool_calls and m.text.strip()]
    return finales[-1].text.strip() if finales else defecto


def resultados(mensajes: list[BaseMessage], herramienta: str | None = None) -> list[ToolMessage]:
    """Los resultados de herramientas (opcionalmente, solo los de una herramienta)."""
    return [m for m in mensajes if isinstance(m, ToolMessage) and (herramienta is None or m.name == herramienta)]
