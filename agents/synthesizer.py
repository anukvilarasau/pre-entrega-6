"""
synthesizer.py — La fase de SÍNTESIS: la respuesta final para el usuario.

Recibe la pregunta, los datos del pedido y los aportes VALIDADOS (el texto
final de cada especialista, con sus observaciones). El LLM redacta:
itinerario día por día, presupuesto desglosado (copiado del analista, no
recalculado) y alternativas si no alcanza.

Hay partes que NO se dejan en manos del LLM, las agrega el código:
- la lista de FUENTES, armada con las fuentes que devolvieron las
  herramientas (con su origen: memoria o web, y la fecha de búsqueda), así
  no se puede inventar una URL;
- el aviso de que los precios son aproximados y salen de la web;
- un aviso si la respuesta se armó con información parcial porque actuó el
  freno de pasos.
"""

import logging
import re
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agents.common import NodoAsync
from state import Fuente, OrquestadorState, validar_actualizacion

log = logging.getLogger(__name__)

AVISO_PRECIOS: str = (
    "_Los precios son aproximados: salen de búsquedas web (o de la memoria local de búsquedas de los últimos días) "
    "y pueden cambiar. Confirmalos antes de reservar._"
)

PROMPT_SINTESIS: str = """Sos el agente de SÍNTESIS de un asesor de viajes. Escribís la respuesta FINAL para el \
usuario, en español rioplatense, clara y concreta, usando SOLO la información de los aportes de los especialistas. \
No inventes datos ni hagas cuentas nuevas: los números salen del analista.

Según el caso:
- Si el analista dijo que el viaje es INVÁLIDO: explicá qué datos están mal y cómo corregirlos. No armes itinerario.
- Si hay presupuesto calculado: incluí
  ## Itinerario (día por día, del Día 1 al último, con las actividades encontradas y el clima esperado). \
NO repitas una actividad en dos días distintos: si hay más días que actividades, proponé días libres o de descanso.
  ## Presupuesto (el desglose del analista: cada renglón, total, si alcanza, cuánto sobra o falta, costo por día y \
qué dólar se usó y de dónde salió)
  Si NO alcanza, agregá ## Alternativas con las opciones que calculó el analista y otras ideas concretas para bajar \
costos (menos días, micro en vez de avión, alojamiento compartido, cocinar, actividades gratuitas).
- Si la pregunta no pide presupuesto (por ejemplo, solo el clima): respondé eso directamente, sin itinerario.
- Si faltaron datos (búsqueda caída, sin precios), decilo con claridad en ## Lo que no pudimos averiguar.
Citá las fuentes junto a cada dato con [URL] o [mem-...] tal como aparecen en los aportes.
Sé breve: como máximo unas 600 palabras. No copies todos los hallazgos: elegí lo que sirve para decidir.
No agregues una sección de fuentes al final ni avisos sobre precios: eso lo agrega el sistema."""


def mensaje_para_sintesis(state: OrquestadorState) -> str:
    pedido = state.get("pedido")
    bloques = [f"PREGUNTA DEL USUARIO:\n{state['pregunta']}"]
    if pedido is not None:
        bloques.append(pedido.como_texto())
    for a in state.get("aportes", []):
        obs = " | ".join(a.observaciones) if a.observaciones else "ninguna"
        bloques.append(f"APORTE DEL {a.agente.upper()} (ronda {a.ronda}) — observaciones: {obs}\n{a.contenido}")
    if not state.get("aportes"):
        bloques.append("(No hay aportes de los especialistas.)")
    if any(d.forzada for d in state.get("decisiones", [])):
        bloques.append("ATENCIÓN: el trabajo se cortó por el límite de pasos; la información puede estar incompleta. "
                       "Decilo y respondé con lo que haya.")
    return "\n\n".join(bloques)


MAX_FUENTES_SIN_CITA: int = 8  # si la respuesta no cita ninguna, se listan como mucho estas


def fuentes_a_listar(fuentes: list[Fuente], cuerpo: str, usadas_por_analista: list[Fuente]) -> list[Fuente]:
    """Qué fuentes van en la lista final (decidido por código).

    1. Las CITADAS en la respuesta (su ID mem-... o su URL aparece en el texto)
       + las que usó el analista para calcular.
    2. Si no hay ninguna citada, las primeras MAX_FUENTES_SIN_CITA.
    Sin repetir la misma página: si una URL aparece como 'web' y como 'memoria', queda una sola.
    """
    citadas = [f for f in fuentes if f.ref in cuerpo or (f.url and f.url in cuerpo)]
    elegidas = citadas + [f for f in usadas_por_analista if f not in citadas]
    if not elegidas:
        elegidas = fuentes[:MAX_FUENTES_SIN_CITA]
    por_pagina: dict[str, Fuente] = {}
    for f in elegidas:
        por_pagina.setdefault(f.url or f.ref, f)
    return list(por_pagina.values())


def seccion_fuentes(fuentes: list[Fuente]) -> str:
    """Lista de fuentes armada por código (no por el LLM)."""
    if not fuentes:
        return "## Fuentes\n- No se usaron fuentes externas."
    lineas = ["## Fuentes"]
    for f in fuentes:
        if f.origen == "web":
            lineas.append(f"- [web] {f.ref} — {f.tema}, buscado el {f.fecha_busqueda.isoformat()}")
        else:
            url = f" → {f.url}" if f.url else ""
            lineas.append(f"- [memoria] {f.ref}{url} — {f.tema}, buscado el {f.fecha_busqueda.isoformat()}")
    return "\n".join(lineas)


# Un lugar con nombre propio de 2+ palabras: "Cerro Catedral", "Isla Victoria", "Camino de los 7 Lagos".
PATRON_LUGAR: re.Pattern[str] = re.compile(
    r"\b[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:(?:\s+(?:de|del|la|las|los|el))*\s+(?:[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+|\d+))+"
)
NO_SON_LUGARES: frozenset[str] = frozenset({"Día", "Días", "Clima", "Regreso", "Llegada"})
PATRON_DIA: re.Pattern[str] = re.compile(r"^\s*[-*]\s*\**\s*Día\s+(\d+)\b(.*)$", re.MULTILINE)


def actividades_repetidas(cuerpo: str) -> dict[str, list[int]]:
    """Chequeo POR CÓDIGO: lugares con nombre propio que aparecen en más de un día del itinerario.

    Devuelve {lugar: [días]} solo para los repetidos. Ignora el primer y el último
    día (viaje de ida y de vuelta) y las palabras del propio formato ("Día 3").
    """
    dias = [(int(n), texto) for n, texto in PATRON_DIA.findall(cuerpo)]
    if len(dias) < 3:
        return {}
    extremos = {dias[0][0], dias[-1][0]}
    vistos: dict[str, list[int]] = {}
    for n, texto in dias:
        if n in extremos:
            continue
        for lugar in {m.group(0) for m in PATRON_LUGAR.finditer(texto)}:
            if lugar.split()[0] in NO_SON_LUGARES:
                continue
            vistos.setdefault(lugar, []).append(n)
    return {lugar: ns for lugar, ns in vistos.items() if len(ns) > 1}


def crear_nodo_sintesis(llm: BaseChatModel) -> NodoAsync:
    async def redactar(mensajes: list[Any]) -> str:
        respuesta = await llm.ainvoke(mensajes)
        cuerpo = (respuesta.text if isinstance(respuesta, AIMessage) else str(respuesta)).strip()
        if isinstance(respuesta, AIMessage) and respuesta.response_metadata.get("stop_reason") == "max_tokens":
            log.warning("[sintesis] la respuesta llegó al tope de tokens y quedó cortada")
            cuerpo += "\n\n_(La respuesta quedó cortada por el límite de longitud.)_"
        return cuerpo

    async def sintesis(state: OrquestadorState) -> dict[str, Any]:
        mensajes: list[Any] = [SystemMessage(content=PROMPT_SINTESIS), HumanMessage(content=mensaje_para_sintesis(state))]
        cuerpo = await redactar(mensajes)

        # Validación por código del itinerario: si se repite una actividad, UNA corrección.
        repetidas = actividades_repetidas(cuerpo)
        if repetidas:
            detalle = "; ".join(f"'{lugar}' en los días {', '.join(map(str, ns))}" for lugar, ns in repetidas.items())
            log.warning("[sintesis] actividades repetidas: %s → se pide una corrección", detalle)
            cuerpo = await redactar([*mensajes, AIMessage(content=cuerpo), HumanMessage(content=(
                f"En el itinerario se repite: {detalle}. Reescribí la respuesta COMPLETA sin repetir ninguna "
                "actividad entre días (usá días libres o de descanso si faltan actividades). El resto, igual."))])
            if (siguen := actividades_repetidas(cuerpo)):
                cuerpo += ("\n\n_Nota: el itinerario repite " + ", ".join(siguen) + "; tomalo como sugerencia y "
                           "reemplazá un día por un día libre._")

        partes: list[str] = []
        if any(d.forzada for d in state.get("decisiones", [])):
            partes.append("> ⚠️ Esta respuesta se armó con información parcial: el sistema llegó a su límite de "
                          "pasos antes de completar todas las verificaciones.")
        partes.append(cuerpo)
        todas = [f for a in state.get("aportes", []) for f in a.fuentes]
        del_analista = [f for a in state.get("aportes", []) if a.agente == "analista" for f in a.fuentes]
        fuentes = fuentes_a_listar(todas, cuerpo, del_analista)
        partes.append(seccion_fuentes(fuentes))
        if fuentes:
            partes.append(AVISO_PRECIOS)
        final = "\n\n".join(partes)

        log.info("[sintesis] respuesta final: %d caracteres, fuentes: %d", len(final), len(fuentes))
        return validar_actualizacion("sintesis", {
            "respuesta_final": final,
            "messages": [AIMessage(content=final, name="sintesis")],
        })

    return sintesis
