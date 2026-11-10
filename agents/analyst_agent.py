"""
analyst_agent.py — El agente ANALISTA.

Es un agente ReAct (`create_react_agent`) con SOLO dos herramientas propias,
sin LLM adentro: validar_viaje (Pydantic) y calcular_presupuesto (cuentas y
conversión de moneda). No busca datos: trabaja con lo que trajo el
investigador.

Cómo se conecta con el grafo principal (nodo "analista"):
1. Recibe SOLO: la instrucción del supervisor + DATOS DEL PEDIDO + los
   HALLAZGOS DEL INVESTIGADOR (solo el texto final de cada aporte, no sus
   llamadas internas ni el historial). Contexto acotado.
2. Corre su ciclo ReAct: valida el viaje, calcula el presupuesto y, si no
   alcanza, calcula alternativas.
3. Arma su Aporte con el resultado, las fuentes de los precios y
   cotizaciones que usó, y cada llamada a herramienta.
4. El chequeo por código anota problemas objetivos (no validó, no calculó,
   una herramienta dio error...). El supervisor los lee.
5. Escribe SOLO su parte del estado.
"""

import logging
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent

import config
from agents.common import NodoAsync, correr_react, extraer_llamadas, respuesta_final, resultados
from state import (
    Aporte,
    OrquestadorState,
    PedidoViaje,
    aportes_de,
    contexto_especialista,
    validacion_del_pedido,
    validar_actualizacion,
)

log = logging.getLogger(__name__)

PROMPT_ANALISTA: str = """Sos el agente ANALISTA de un asesor de viajes. No buscás datos: trabajás SOLO con los \
DATOS DEL PEDIDO y los HALLAZGOS DEL INVESTIGADOR que te pasan, y usás tus herramientas para validar y calcular. \
Nunca hagas cuentas de cabeza: los números salen de las herramientas.

Pasos:
1. validar_viaje con los DATOS DEL PEDIDO, tal cual (dias_pedidos = Días, si está). Si responde INVÁLIDO, informá los \
errores y TERMINÁ: no calcules el presupuesto y NO inventes ni supongas otras fechas, aunque la instrucción lo pida \
(eso lo corrige el usuario).
2. calcular_presupuesto, opción "económica", con los precios más baratos razonables de los hallazgos. Usá las \
noches que devolvió validar_viaje. Respetá las unidades:
   - pasaje: precio POR PERSONA; tramo="ida_y_vuelta" si el precio incluye los dos viajes, "por_tramo" si es solo ida;
   - alojamiento: unidad="por_noche_por_habitacion" o "por_noche_por_persona", como diga el hallazgo (si no lo dice, \
usá por persona, que es lo más caro, y aclaralo);
   - si un precio es un rango, usá el promedio (o el punto medio); si es un valor "desde", usalo pero aclará que es optimista;
   - comida y actividades: por persona por día;
   - extras (excursiones, etc.) en "extras".
   En cada precio poné su "fuente" (la URL o el ID mem-... del hallazgo). Si algún precio está en otra moneda que el \
presupuesto, pasá los tipos_de_cambio que encontró el investigador, cada uno con su fuente; medio_pago="tarjeta" \
salvo que el hallazgo diga efectivo.
   Llamá a calcular_presupuesto de a UNA opción por vez (nunca varias llamadas en paralelo en el mismo mensaje).
3. Si el resultado dice NO ENTRA, calculá al menos UNA alternativa más barata (menos días, micro en vez de avión, \
alojamiento más simple, sin extras) y decí si esa entra.
4. Si falta un precio imprescindible (pasajes, alojamiento o comida), no lo inventes: decilo en "Datos que faltan".

Formato de tu respuesta final (en castellano):
## Validación
<VÁLIDO o INVÁLIDO, con el detalle>
## Presupuesto
<copiá el resultado de cada opción calculada: renglones, total, si entra, cuánto sobra o falta, costo por día y las \
conversiones de moneda con el dólar usado y su fuente>
## Conclusión
<una o dos frases: si alcanza o no y qué conviene>
## Datos que faltan
<lo que faltó, o "ninguno">"""


def crear_agente_analista(llm: BaseChatModel, herramientas: list[BaseTool]) -> CompiledStateGraph:
    """El agente ReAct del analista (subgrafo con su propio ciclo)."""
    return create_react_agent(llm, herramientas, prompt=PROMPT_ANALISTA, name="analista")


def hallazgos_para_analista(state: OrquestadorState) -> str:
    """Solo el texto final de cada aporte del investigador (no sus llamadas internas)."""
    aportes = aportes_de(state, "investigador")
    if not aportes:
        return "HALLAZGOS DEL INVESTIGADOR:\n(todavía no hay hallazgos)"
    bloques = [f"--- ronda {a.ronda} ---\n{a.contenido}" for a in aportes]
    return "HALLAZGOS DEL INVESTIGADOR:\n" + "\n".join(bloques)


def chequear_analisis(pedido: PedidoViaje | None, contenido: str, mensajes: list[BaseMessage]) -> list[str]:
    """Chequeo POR CÓDIGO del trabajo del analista. Devuelve la lista de problemas."""
    observaciones: list[str] = []
    hechos = resultados(mensajes)
    if not hechos:
        return ["El analista no usó ninguna herramienta: sus números no están calculados."]

    calculos = resultados(mensajes, "calcular_presupuesto")
    llamadas, _, _ = extraer_llamadas(mensajes)
    estado_validacion = validacion_del_pedido(llamadas, pedido)  # el pedido REAL, no fechas inventadas
    if estado_validacion is None:
        observaciones.append("El analista no validó el viaje con validar_viaje.")
    viaje_valido = estado_validacion == "valido"
    calculo_ok = any(str(m.content).startswith("PRESUPUESTO") for m in calculos)

    if viaje_valido and pedido is not None and pedido.presupuesto is not None and not calculo_ok:
        observaciones.append("El viaje es válido y hay presupuesto, pero no se calculó el presupuesto con éxito.")
    for m in hechos:
        texto = str(m.content)
        if texto.startswith(("ERROR", "Error")):
            observaciones.append(f"Una herramienta del analista dio error ({m.name}): {texto[:160]}")
    if viaje_valido and calculo_ok and not any(s in contenido for s in ("ENTRA", "entra", "alcanza")):
        observaciones.append("La respuesta del analista no dice si el viaje entra o no en el presupuesto.")
    return observaciones


def crear_nodo_analista(agente: CompiledStateGraph) -> NodoAsync:
    """Envuelve al agente ReAct como nodo del grafo principal."""

    async def analista(state: OrquestadorState) -> dict[str, Any]:
        instruccion = state.get("instruccion_actual", "").strip() or "Validá el viaje y calculá el presupuesto."
        ronda = max(1, state.get("pasos_supervisor", 1))
        pedido = state.get("pedido")
        log.info("[analista] ronda %d — instrucción: %s", ronda, instruccion[:150].replace("\n", " "))

        # Contexto acotado: instrucción + pedido + hallazgos del investigador. Nada más.
        entrada = contexto_especialista(instruccion, pedido, extra=hallazgos_para_analista(state))
        mensajes, cortado = await correr_react(agente, entrada)

        llamadas, fuentes, _ = extraer_llamadas(mensajes)
        contenido = respuesta_final(mensajes, "(el analista no llegó a dar una respuesta final)")
        observaciones = chequear_analisis(pedido, contenido, mensajes)
        if cortado:
            observaciones.insert(0, f"El ciclo ReAct se cortó por el límite de {config.RECURSION_LIMIT_ESPECIALISTA} pasos.")

        aporte = Aporte(
            agente="analista",
            ronda=ronda,
            instruccion=instruccion,
            contenido=contenido,
            fuentes=fuentes,
            herramientas=list(dict.fromkeys(ll.herramienta for ll in llamadas)),
            llamadas=llamadas,
            observaciones=observaciones,
        )
        log.info("[analista] llamadas: %d, observaciones: %d", len(llamadas), len(observaciones))
        resumen = AIMessage(
            content=f"[analista] herramientas: {', '.join(aporte.herramientas) or 'ninguna'}. "
                    f"Observaciones: {'; '.join(observaciones) or 'ninguna'}",
            name="analista",
        )
        return validar_actualizacion("analista", {"aportes": [aporte], "messages": [resumen]})

    return analista
