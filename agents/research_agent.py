"""
research_agent.py — El agente INVESTIGADOR.

Es un agente ReAct armado con `create_react_agent` (LangGraph) con SOLO dos
herramientas: buscar_en_memoria y buscar_en_web. Consigue datos actuales con
fuente (pasajes, alojamiento, clima, actividades, tipo de cambio). No arma
itinerarios ni hace cuentas: eso es trabajo de otros.

Cómo se conecta con el grafo principal (nodo "investigador"):
1. Recibe SOLO `instruccion_actual` (lo que le pidió el supervisor), la lista
   estructurada TEMAS A BUSCAR (`temas_pedidos`) y el bloque "DATOS DEL PEDIDO" (origen, destino, fechas, personas,
   presupuesto), armado por código desde `state["pedido"]`. No recibe la
   pregunta original, ni el historial, ni los aportes de otros: así se evita
   la "contaminación de contexto" y no depende de que el supervisor se
   acuerde de copiar todos los datos en la instrucción.
2. Corre su propio ciclo ReAct (pensar → herramienta → observar → ...), con
   su propio techo de pasos (RECURSION_LIMIT_ESPECIALISTA).
3. Arma un `Aporte` con: su respuesta, las fuentes que DEVOLVIERON LAS
   HERRAMIENTAS (no las que diga el LLM) y cada llamada que hizo.
4. Un chequeo por código (`chequear_investigacion`) anota problemas
   objetivos en `observaciones` (sin fuentes, sin precios, web caída...).
   El supervisor los lee para decidir si pide rehacer el trabajo.
5. Escribe SOLO su parte del estado (validar_actualizacion lo garantiza).
"""

import logging
import re
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent

import config
from agents.common import NodoAsync, correr_react, extraer_llamadas, respuesta_final, resultados
from state import PATRON_PRECIO, Aporte, CoberturaTema, Fuente, OrquestadorState, cantidad, contexto_especialista, validar_actualizacion

log = logging.getLogger(__name__)

PROMPT_INVESTIGADOR: str = f"""Sos el agente INVESTIGADOR de un asesor de viajes. Tu ÚNICO trabajo es conseguir \
datos actuales y con fuente. No armás itinerarios ni hacés cuentas: eso lo hacen otros agentes.

Herramientas:
- buscar_en_memoria: datos ya buscados antes (de {config.MEMORIA_MAX_DIAS} días o menos). Usala SIEMPRE PRIMERO, \
una vez por cada tema que te pidan (podés pedir varias en paralelo).
- buscar_en_web: búsqueda web con Tavily. Usala SOLO para los temas en los que la memoria respondió \
"SIN RESULTADOS EN MEMORIA". Usá forzar=true únicamente si lo que hay en memoria no sirve (por ejemplo, no trae precios).

Temas posibles: pasajes, alojamiento, comida, clima, actividades, cambio (cotización del dólar).

Presupuesto de búsquedas: como máximo UNA búsqueda web por tema (dos solo si la primera no trajo ningún precio). Los precios de la web son de referencia: rangos, valores "desde" o de fechas cercanas sirven; no busques precios exactos para las fechas del viaje.

Reglas:
1. No inventes nada. Cada dato va con su fuente entre corchetes: la URL [https://...] si salió de la web, o el ID \
[mem-...] si salió de la memoria, y aclarando (web) o (memoria).
2. Precios: copiá el monto con su moneda y SU UNIDAD tal como dice la fuente:
   - pasajes: por persona, ida y vuelta o por tramo; avión o micro;
   - alojamiento: "por noche por habitación (para N personas)" o "por persona por noche";
   - si la fuente no dice la unidad, escribí "unidad no especificada".
3. Tipo de cambio: buscá el dólar oficial, tarjeta y MEP (no el blue); informá cada valor y su fuente.
4. Si una búsqueda da ERROR o no encuentra nada, decilo en "Datos que faltan". No lo tapes.

Formato de tu respuesta final (en castellano, breve):
## Hallazgos
### <Tema>
- <dato con monto, moneda y unidad> — [fuente] (web|memoria)
## Datos que faltan
- <lo que no se encontró, o "ninguno">"""

TEMAS_CON_PRECIO: tuple[str, ...] = ("pasajes", "alojamiento")


def crear_agente_investigador(llm: BaseChatModel, herramientas: list[BaseTool]) -> CompiledStateGraph:
    """El agente ReAct del investigador (subgrafo con su propio ciclo)."""
    return create_react_agent(llm, herramientas, prompt=PROMPT_INVESTIGADOR, name="investigador")


def calcular_cobertura(mensajes: list[BaseMessage]) -> list[CoberturaTema]:
    """Resumen por tema armado POR CÓDIGO: fuentes, de dónde salieron y si hay precios.

    El supervisor lo ve junto al texto del investigador, así no depende de leer un
    informe largo (en la corrida real del escenario 5 el texto le llegaba cortado y
    volvió a pedir temas que ya estaban cubiertos).
    """
    _, _, argumentos_de = extraer_llamadas(mensajes)
    por_tema: dict[str, dict[str, Any]] = {}
    for m in resultados(mensajes):
        tema = str(argumentos_de.get(m.tool_call_id, {}).get("tema", "otro"))
        datos = por_tema.setdefault(tema, {"refs": {}, "precios": False})
        texto = str(m.content)
        if not texto.startswith("ERROR") and PATRON_PRECIO.search(texto):
            datos["precios"] = True
        for f in (m.artifact if isinstance(m.artifact, list) else []):
            if isinstance(f, Fuente):
                datos["refs"].setdefault(f.url or f.ref, f.origen)
    return [
        CoberturaTema(
            tema=tema,  # type: ignore[arg-type]
            fuentes=len(d["refs"]),
            memoria=sum(1 for o in d["refs"].values() if o == "memoria"),
            web=sum(1 for o in d["refs"].values() if o == "web"),
            con_precios=d["precios"],
        )
        for tema, d in por_tema.items()
    ]


def chequear_investigacion(
    temas_pedidos: list[str], contenido: str, mensajes: list[BaseMessage], fuentes: list[Fuente]
) -> list[str]:
    """Chequeo POR CÓDIGO (no depende del LLM). Devuelve la lista de problemas encontrados.

    Los temas a controlar salen del campo ESTRUCTURADO `temas_pedidos` que escribió el
    supervisor (no de buscar palabras en la instrucción: "no repitas pasajes" no es pedir pasajes).
    """
    observaciones: list[str] = []
    hechos = resultados(mensajes)
    if not hechos:
        observaciones.append("El investigador no usó ninguna herramienta: sus datos no tienen respaldo.")
    if hechos and not fuentes:
        observaciones.append("Las búsquedas no devolvieron ninguna fuente utilizable.")

    errores = [str(m.content) for m in hechos if str(m.content).startswith("ERROR")]
    for e in errores:
        observaciones.append(f"Falló una búsqueda: {e[:160]}")

    # ¿Se buscó cada tema pedido? ¿Hay precios en pasajes y alojamiento?
    _, _, argumentos_de = extraer_llamadas(mensajes)
    for tema in dict.fromkeys(temas_pedidos):
        del_tema = [m for m in hechos
                    if argumentos_de.get(m.tool_call_id, {}).get("tema") == tema and not str(m.content).startswith("ERROR")]
        if not del_tema:
            if not errores:
                observaciones.append(f"Se pidió el tema '{tema}' pero no se hizo ninguna búsqueda sobre él.")
        elif tema in TEMAS_CON_PRECIO and not any(PATRON_PRECIO.search(str(m.content)) for m in del_tema):
            observaciones.append(f"Se pidieron precios de {tema} pero los resultados encontrados no traen ningún precio.")

    # ¿La respuesta cita al menos una de las fuentes reales?
    if fuentes and not any(f.ref in contenido or (f.url and f.url in contenido) for f in fuentes):
        observaciones.append("La respuesta del investigador no cita ninguna de las fuentes que encontró.")
    return observaciones


def crear_nodo_investigador(agente: CompiledStateGraph) -> NodoAsync:
    """Envuelve al agente ReAct como nodo del grafo principal."""

    async def investigador(state: OrquestadorState) -> dict[str, Any]:
        instruccion = state.get("instruccion_actual", "").strip() or state["pregunta"]
        ronda = max(1, state.get("pasos_supervisor", 1))
        log.info("[investigador] ronda %d — instrucción: %s", ronda, instruccion[:150].replace("\n", " "))

        # Contexto acotado: la instrucción del supervisor + los datos estructurados del
        # pedido. Nada del historial general ni la pregunta original.
        temas = list(state.get("temas_pedidos", []))
        extra = f"TEMAS A BUSCAR: {', '.join(temas)}" if temas else ""
        mensajes, cortado = await correr_react(agente, contexto_especialista(instruccion, state.get("pedido"), extra))

        llamadas, fuentes, _ = extraer_llamadas(mensajes)
        contenido = respuesta_final(mensajes, "(el investigador no llegó a dar una respuesta final)")
        observaciones = chequear_investigacion(temas, contenido, mensajes, fuentes)
        if cortado:
            observaciones.insert(0, f"El ciclo ReAct se cortó por el límite de {config.RECURSION_LIMIT_ESPECIALISTA} pasos.")

        aporte = Aporte(
            agente="investigador",
            ronda=ronda,
            instruccion=instruccion,
            contenido=contenido,
            fuentes=fuentes,
            herramientas=list(dict.fromkeys(ll.herramienta for ll in llamadas)),
            llamadas=llamadas,
            cobertura=calcular_cobertura(mensajes),
            observaciones=observaciones,
        )
        origenes = {o: sum(1 for f in aporte.fuentes if f.origen == o) for o in ("memoria", "web")}
        log.info(
            "[investigador] llamadas: %d, fuentes: %d de memoria / %d de la web, observaciones: %d",
            len(llamadas), origenes["memoria"], origenes["web"], len(observaciones),
        )
        resumen = AIMessage(
            content=f"[investigador] {cantidad(len(aporte.fuentes), 'fuente')} "
                    f"({origenes['memoria']} de memoria, {origenes['web']} de la web). "
                    f"Observaciones: {'; '.join(observaciones) or 'ninguna'}",
            name="investigador",
        )
        return validar_actualizacion("investigador", {"aportes": [aporte], "messages": [resumen]})

    return investigador
