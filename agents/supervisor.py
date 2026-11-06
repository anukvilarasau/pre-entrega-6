"""
supervisor.py — El SUPERVISOR: decide quién trabaja y cuándo terminar.

En cada turno:
1. (Solo la primera vez) ENTIENDE EL PEDIDO: con salida estructurada
   `PedidoViaje` extrae origen, destino, fechas, personas, presupuesto y
   moneda de la pregunta. Queda fijo para toda la corrida.
2. FRENO DE PASOS: si ya tomó MAX_PASOS_SUPERVISOR − 1 decisiones, la
   siguiente es FINALIZAR forzado, por código, sin preguntarle al LLM.
   Así nunca hay "supervisor infinito".
3. DECIDE con salida estructurada `DecisionSupervisor`
   (siguiente: Literal["investigador", "analista", "FINALIZAR"]), guiado por
   una RÚBRICA en su prompt para validar los aportes antes de terminar. Ve
   un resumen de cada aporte (incluidas las observaciones del chequeo por
   código), no las llamadas internas de las herramientas.
4. REGLAS DURAS (por código) que corrigen decisiones imposibles: mandar al
   analista sin datos, o finalizar sin ningún aporte.
5. Si el LLM no devuelve una decisión válida dos veces, decide por reglas.

`ruta_supervisor` es la función de las aristas condicionales del grafo:
lee la decisión y devuelve el NOMBRE DEL NODO siguiente (con Literal).
"""

import logging
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import ValidationError

import config
from agents.common import NodoAsync
from state import (
    Aporte,
    DecisionRegistrada,
    DecisionSupervisor,
    OrquestadorState,
    PedidoViaje,
    aportes_de,
    validar_actualizacion,
)

log = logging.getLogger(__name__)

LARGO_CONTENIDO_RESUMEN: int = 2500  # caracteres del contenido de cada aporte que ve el supervisor
MAX_RONDAS_INVESTIGADOR: int = 2      # la búsqueda original + UNA corrección

PROMPT_PEDIDO: str = """Extraé los datos del pedido de viaje del usuario. Hoy es {hoy}.
Reglas:
- "pesos" → moneda "ARS"; "dólares" → "USD". El presupuesto es el TOTAL del viaje.
- Si dice un mes sin día exacto, asumí que la ida es el día 10 de ese mes (del próximo año en que ese mes \
todavía no pasó) y anotalo en "supuestos".
- Si dice la cantidad de días y la ida, calculá la vuelta como ida + (días − 1). Ej.: 7 días desde el 10/01 → vuelta 16/01.
- NO corrijas los errores del usuario: si da fechas imposibles (vuelta antes que la ida), copialas tal cual.
- Lo que el usuario no dijo queda en null."""

PROMPT_SUPERVISOR: str = """Sos el SUPERVISOR de un asesor de viajes multi-agente. Coordinás a dos especialistas y \
decidís quién trabaja ahora, o si ya se puede FINALIZAR (pasar a la síntesis de la respuesta).

Especialistas:
- investigador: busca datos actuales con fuente (memoria local o web): pasajes, alojamiento, clima, actividades, \
comida y cotización del dólar. No hace cuentas.
- analista: valida el viaje (fechas, personas, presupuesto) y calcula el presupuesto con los precios que trajo el \
investigador. Sin precios del investigador no puede trabajar.

RÚBRICA: para FINALIZAR se tienen que cumplir TODOS estos puntos.
R1 Cobertura: cada parte de la pregunta tiene datos. Si pregunta si alcanza la plata o pide armar el viaje, hacen \
falta pasajes, alojamiento, comida/actividades y, si hay precios en dólares, la cotización.
R2 Fuentes: cada aporte del investigador tiene fuentes (URL o mem-...).
R3 Precios: los hallazgos de pasajes y alojamiento traen al menos un monto cada uno. Los precios de la web son DE \
REFERENCIA: rangos, valores "desde", promedios o precios de fechas cercanas ALCANZAN para estimar. NO exijas precios \
exactos para las fechas del viaje (casi nunca están publicados): con un precio de referencia de pasajes y otro de \
alojamiento, el analista ya puede calcular (y aclara las unidades dudosas).
R4 Cálculo: si hay presupuesto, el analista ejecutó validar_viaje y calcular_presupuesto y dijo si entra. Si \
validar_viaje dio INVÁLIDO, alcanza con eso: no hay que calcular.
R5 Observaciones: ningún aporte tiene observaciones del chequeo por código sin resolver. Si las tiene, devolvé el \
trabajo al MISMO especialista con una instrucción que diga exactamente qué corregir (por ejemplo: "rehacer la \
búsqueda de pasajes con forzar=true porque no trajo precios").
R6 Coherencia: no hay contradicciones entre aportes (noches, personas, monedas). Si las hay, pedile al analista \
que recalcule con los datos correctos.

Reglas:
- Si la pregunta NO tiene presupuesto ni pide cuentas (por ejemplo, solo el clima), no llames al analista.
- No repitas un pedido que ya salió bien. Como máximo UNA ronda de corrección por especialista: después, seguí con lo \
que hay (el sistema no te va a dejar mandar al investigador una tercera vez).
- Si hay presupuesto, NO se puede finalizar sin que el analista haya calculado (el sistema lo impide).
- Pedí la cotización del dólar oficial, tarjeta y MEP (no el blue).
- Si un problema no se puede resolver (por ejemplo, la búsqueda web está caída y ya se reintentó), FINALIZÁ igual: \
la síntesis le va a avisar al usuario qué faltó.
- Tenés un máximo de {maximo} decisiones; esta es la número {paso}. Si estás cerca del límite, priorizá terminar.
- La instrucción tiene que ser concreta y autocontenida: el especialista NO ve la pregunta del usuario (sí ve los \
DATOS DEL PEDIDO). Al investigador enumerale los temas a buscar (pasajes, alojamiento, comida, actividades, clima, \
cambio) y pedile precios de referencia, no exactos.
- En "evaluacion", repasá la rúbrica punto por punto en una o dos líneas."""


# ---------------------------------------------------------------------------
# Lo que ve el supervisor
# ---------------------------------------------------------------------------
def resumen_aporte(n: int, a: Aporte) -> str:
    """Resumen de UN aporte para el supervisor (no incluye las llamadas internas)."""
    origenes = {o: sum(1 for f in a.fuentes if f.origen == o) for o in ("memoria", "web")}
    contenido = a.contenido[:LARGO_CONTENIDO_RESUMEN] + ("..." if len(a.contenido) > LARGO_CONTENIDO_RESUMEN else "")
    return (
        f"APORTE #{n} | agente={a.agente} | ronda={a.ronda} | herramientas={','.join(a.herramientas) or 'ninguna'} | "
        f"fuentes={len(a.fuentes)} (memoria={origenes['memoria']}, web={origenes['web']})\n"
        f"Instrucción que recibió: {a.instruccion}\n"
        f"Observaciones del chequeo por código: {' | '.join(a.observaciones) if a.observaciones else 'ninguna'}\n"
        f"Contenido:\n{contenido}"
    )


def mensaje_para_supervisor(state: OrquestadorState, pedido: PedidoViaje) -> str:
    aportes = state.get("aportes", [])
    bloques = [f"PREGUNTA DEL USUARIO:\n{state['pregunta']}", pedido.como_texto()]
    if aportes:
        bloques.append("APORTES HASTA AHORA:\n\n" + "\n\n".join(resumen_aporte(i, a) for i, a in enumerate(aportes, 1)))
    else:
        bloques.append("APORTES HASTA AHORA: ninguno.")
    bloques.append("¿Quién trabaja ahora, o FINALIZAR?")
    return "\n\n".join(bloques)


# ---------------------------------------------------------------------------
# Decisiones por código
# ---------------------------------------------------------------------------
def decision_por_reglas(state: OrquestadorState, pedido: PedidoViaje) -> DecisionSupervisor:
    """Respaldo determinístico si el LLM no devuelve una decisión válida."""
    if not aportes_de(state, "investigador"):
        return DecisionSupervisor(
            evaluacion="(por reglas) Todavía no hay datos.", siguiente="investigador",
            instruccion="Buscá pasajes, alojamiento, comida y actividades, clima y la cotización del dólar.",
        )
    if pedido.presupuesto is not None and not aportes_de(state, "analista"):
        return DecisionSupervisor(
            evaluacion="(por reglas) Hay precios y presupuesto, falta el cálculo.", siguiente="analista",
            instruccion="Validá el viaje y calculá el presupuesto con los precios encontrados.",
        )
    return DecisionSupervisor(evaluacion="(por reglas) Ya hay datos suficientes.", siguiente="FINALIZAR")


def aplicar_reglas_duras(
    decision: DecisionSupervisor, state: OrquestadorState, pedido: PedidoViaje | None = None
) -> tuple[DecisionSupervisor, str]:
    """Corrige por código las decisiones que violan la rúbrica. Devuelve (decisión, nota o '').

    1. Analista sin datos del investigador → investigador.
    2. FINALIZAR sin ningún aporte → investigador.
    3. FINALIZAR con presupuesto, datos con fuente y sin cálculo → analista (R4 por código).
    4. Investigador por tercera vez → analista (si hay presupuesto) o FINALIZAR (anti-bucle).
    """
    sin_investigacion = not aportes_de(state, "investigador")
    if decision.siguiente == "analista" and sin_investigacion:
        return DecisionSupervisor(
            evaluacion=decision.evaluacion, siguiente="investigador",
            instruccion="Buscá los precios de pasajes, alojamiento, comida y actividades, y la cotización del dólar.",
        ), "regla dura: el analista no puede trabajar sin datos del investigador → investigador"
    if decision.siguiente == "FINALIZAR" and not state.get("aportes"):
        return DecisionSupervisor(
            evaluacion=decision.evaluacion, siguiente="investigador",
            instruccion=decision.instruccion or "Buscá los datos necesarios para responder la pregunta del pedido.",
        ), "regla dura: no se puede finalizar sin ningún aporte → investigador"

    hay_presupuesto = pedido is not None and pedido.presupuesto is not None
    sin_analisis = not aportes_de(state, "analista")
    con_fuentes = any(a.fuentes for a in aportes_de(state, "investigador"))
    instruccion_analista = ("Validá el viaje con los DATOS DEL PEDIDO y calculá el presupuesto con los precios de "
                            "referencia de los hallazgos (opción económica; si no alcanza, una alternativa). Si un "
                            "precio es un rango o un valor 'desde', usá un valor representativo y aclaralo.")

    # R4 por código: con presupuesto y datos con fuente, no se termina sin el cálculo del analista.
    if decision.siguiente == "FINALIZAR" and hay_presupuesto and sin_analisis and con_fuentes:
        return DecisionSupervisor(evaluacion=decision.evaluacion, siguiente="analista",
                                  instruccion=instruccion_analista), \
            "regla dura: hay presupuesto y precios, no se puede finalizar sin el cálculo del analista → analista"

    # Anti-bucle: el investigador trabaja como máximo 2 veces (la original + 1 corrección).
    if decision.siguiente == "investigador" and len(aportes_de(state, "investigador")) >= MAX_RONDAS_INVESTIGADOR:
        if hay_presupuesto and sin_analisis and con_fuentes:
            return DecisionSupervisor(evaluacion=decision.evaluacion, siguiente="analista",
                                      instruccion=instruccion_analista), \
                f"regla dura: el investigador ya trabajó {MAX_RONDAS_INVESTIGADOR} veces; se sigue con lo que hay → analista"
        return DecisionSupervisor(evaluacion=decision.evaluacion, siguiente="FINALIZAR"), \
            f"regla dura: el investigador ya trabajó {MAX_RONDAS_INVESTIGADOR} veces; se finaliza con lo que hay"
    return decision, ""


# ---------------------------------------------------------------------------
# El nodo
# ---------------------------------------------------------------------------
def crear_nodo_supervisor(llm: BaseChatModel, max_pasos: int | None = None) -> NodoAsync:
    """Crea el nodo supervisor. `max_pasos` permite bajar el freno (escenario 8)."""
    extractor = llm.with_structured_output(PedidoViaje)
    decisor = llm.with_structured_output(DecisionSupervisor)

    async def extraer_pedido(pregunta: str) -> PedidoViaje:
        mensajes = [SystemMessage(content=PROMPT_PEDIDO.format(hoy=config.FECHA_REFERENCIA.isoformat())),
                    HumanMessage(content=pregunta)]
        for intento in (1, 2):
            try:
                pedido = await extractor.ainvoke(mensajes)
                return pedido if isinstance(pedido, PedidoViaje) else PedidoViaje.model_validate(pedido)
            except (ValidationError, ValueError) as e:
                log.warning("[supervisor] extracción del pedido inválida (intento %d): %s", intento, e)
        return PedidoViaje(supuestos=["No se pudieron extraer los datos del pedido automáticamente."])

    async def supervisor(state: OrquestadorState) -> dict[str, Any]:
        maximo = max_pasos or config.MAX_PASOS_SUPERVISOR
        paso = state.get("pasos_supervisor", 0) + 1
        actualizacion: dict[str, Any] = {"pasos_supervisor": paso}

        pedido = state.get("pedido")
        if pedido is None:
            pedido = await extraer_pedido(state["pregunta"])
            actualizacion["pedido"] = pedido
            log.info("[supervisor] pedido: %s", pedido.model_dump(exclude_none=True))

        forzada = False
        nota = ""
        if paso >= maximo:
            # FRENO: la última decisión posible es siempre FINALIZAR, sin consultar al LLM.
            forzada = True
            decision = DecisionSupervisor(
                evaluacion=f"FRENO DE PASOS: se llegó a la decisión {paso} de {maximo}. Se finaliza con lo que hay.",
                siguiente="FINALIZAR",
            )
        else:
            mensajes = [SystemMessage(content=PROMPT_SUPERVISOR.format(maximo=maximo, paso=paso)),
                        HumanMessage(content=mensaje_para_supervisor(state, pedido))]
            decision = None
            for intento in (1, 2):
                try:
                    crudo = await decisor.ainvoke(mensajes)
                    decision = crudo if isinstance(crudo, DecisionSupervisor) else DecisionSupervisor.model_validate(crudo)
                    break
                except (ValidationError, ValueError) as e:
                    log.warning("[supervisor] decisión inválida del LLM (intento %d): %s", intento, e)
            if decision is None:
                decision = decision_por_reglas(state, pedido)
                nota = "el LLM no devolvió una decisión válida: se decidió por reglas"
            else:
                decision, nota = aplicar_reglas_duras(decision, state, pedido)

        evaluacion = decision.evaluacion + (f" [{nota}]" if nota else "")
        log.info("[supervisor] decisión %d/%d → %s%s", paso, maximo, decision.siguiente, " (FORZADA)" if forzada else "")
        actualizacion.update(
            siguiente=decision.siguiente,
            instruccion_actual=decision.instruccion,
            decisiones=[DecisionRegistrada(paso=paso, siguiente=decision.siguiente, instruccion=decision.instruccion,
                                           evaluacion=evaluacion, forzada=forzada)],
            messages=[AIMessage(
                content=f"[supervisor] decisión {paso}: {decision.siguiente}"
                        f"{' (forzada por el freno de pasos)' if forzada else ''}. {evaluacion}",
                name="supervisor",
            )],
        )
        return validar_actualizacion("supervisor", actualizacion)

    return supervisor


def ruta_supervisor(state: OrquestadorState) -> Literal["investigador", "analista", "sintesis"]:
    """Arista condicional: traduce la decisión del supervisor al nombre del nodo siguiente."""
    siguiente = state.get("siguiente", "FINALIZAR")
    if siguiente == "investigador":
        return "investigador"
    if siguiente == "analista":
        return "analista"
    return "sintesis"  # FINALIZAR → fase de síntesis
