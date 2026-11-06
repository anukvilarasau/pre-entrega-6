"""Tests del supervisor: extracción del pedido, ruteo con Literal, rúbrica, reglas duras y freno de pasos."""

import typing
from datetime import date
from typing import Any, Literal

import pytest
from langchain_core.messages import BaseMessage
from pydantic import BaseModel

import config
from agents.mock import politica_supervisor_estructurada
from agents.supervisor import PROMPT_SUPERVISOR, crear_nodo_supervisor, ruta_supervisor
from llm import LLMMock
from state import Aporte, DecisionSupervisor, Fuente, LlamadaHerramienta, PedidoViaje, estado_inicial

DEMO = ("Quiero ir 7 días a Bariloche en enero, somos 2 personas, salimos desde Mendoza y tenemos "
        "800.000 pesos. ¿Me armás el viaje y me decís si nos alcanza?")
PEDIDO = PedidoViaje(origen="Mendoza", destino="Bariloche", fecha_ida=date(2027, 1, 10),
                     fecha_vuelta=date(2027, 1, 16), dias=7, personas=2, presupuesto=800_000, moneda="ARS")
URL = "https://micros.mock-viajes.example/mendoza-bariloche"


@pytest.fixture(autouse=True)
def hoy_fijo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "FECHA_REFERENCIA", date(2026, 11, 5))


def estado(**cambios: Any) -> Any:
    s: Any = estado_inicial(DEMO)
    s.update(cambios)
    return s


def aporte_inv(observaciones: list[str] | None = None) -> Aporte:
    return Aporte(agente="investigador", ronda=1, instruccion="Buscá precios", contenido=f"- micro $140.000 — [{URL}] (web)",
                  fuentes=[Fuente(ref=URL, origen="web", tema="pasajes")], herramientas=["buscar_en_memoria", "buscar_en_web"],
                  llamadas=[LlamadaHerramienta(herramienta="buscar_en_memoria", resultado="SIN RESULTADOS EN MEMORIA ...")],
                  observaciones=observaciones or [])


def llm_que_decide(decision: DecisionSupervisor | Exception) -> LLMMock:
    """LLM cuyo decisor devuelve siempre `decision` (o lanza el error); el pedido lo extrae el mock."""
    def politica(mensajes: list[BaseMessage], esquema: type[BaseModel]) -> BaseModel:
        if esquema is PedidoViaje:
            return politica_supervisor_estructurada(mensajes, esquema)
        if isinstance(decision, Exception):
            raise decision
        return decision
    return LLMMock(politica_estructurada=politica)


# ------------------------------------------------------------------- ruteo
def test_ruta_devuelve_literal_con_los_nombres_de_los_nodos() -> None:
    retorno = typing.get_type_hints(ruta_supervisor)["return"]
    assert retorno == Literal["investigador", "analista", "sintesis"]
    assert ruta_supervisor(estado(siguiente="investigador")) == "investigador"
    assert ruta_supervisor(estado(siguiente="analista")) == "analista"
    assert ruta_supervisor(estado(siguiente="FINALIZAR")) == "sintesis"


def test_el_prompt_tiene_la_rubrica() -> None:
    for punto in ("R1 Cobertura", "R2 Fuentes", "R3 Precios", "R4 Cálculo", "R5 Observaciones", "R6 Coherencia"):
        assert punto in PROMPT_SUPERVISOR


# -------------------------------------------------------- extracción del pedido
async def test_primera_decision_extrae_el_pedido_y_delega_al_investigador() -> None:
    nodo = crear_nodo_supervisor(LLMMock(politica_estructurada=politica_supervisor_estructurada))
    salida = await nodo(estado())
    p: PedidoViaje = salida["pedido"]
    assert (p.origen, p.destino, p.dias, p.personas, p.presupuesto, p.moneda) == ("Mendoza", "Bariloche", 7, 2, 800_000, "ARS")
    assert (p.fecha_ida, p.fecha_vuelta) == (date(2027, 1, 10), date(2027, 1, 16))
    assert p.supuestos and "enero sin día exacto" in p.supuestos[0]
    assert salida["siguiente"] == "investigador" and salida["pasos_supervisor"] == 1
    assert salida["decisiones"][0].paso == 1 and not salida["decisiones"][0].forzada
    assert set(salida) <= {"messages", "pedido", "pasos_supervisor", "siguiente", "instruccion_actual", "decisiones"}


async def test_el_pedido_se_extrae_una_sola_vez() -> None:
    llm = LLMMock(politica_estructurada=politica_supervisor_estructurada)
    salida = await crear_nodo_supervisor(llm)(estado(pedido=PEDIDO, pasos_supervisor=1, aportes=[aporte_inv()]))
    assert "pedido" not in salida  # ya estaba: no se vuelve a escribir
    assert len(llm.vistos) == 1     # solo la decisión, sin extracción


# ---------------------------------------------------------------- decisiones
async def test_con_precios_y_presupuesto_delega_al_analista() -> None:
    nodo = crear_nodo_supervisor(LLMMock(politica_estructurada=politica_supervisor_estructurada))
    salida = await nodo(estado(pedido=PEDIDO, pasos_supervisor=1, aportes=[aporte_inv()]))
    assert salida["siguiente"] == "analista" and "calculá el presupuesto" in salida["instruccion_actual"]


async def test_busqueda_sin_precios_se_devuelve_al_investigador() -> None:
    """Escenario 6: la observación del chequeo por código hace que el supervisor pida rehacer."""
    nodo = crear_nodo_supervisor(LLMMock(politica_estructurada=politica_supervisor_estructurada))
    obs = ["Se pidieron precios de pasajes pero los resultados encontrados no traen ningún precio."]
    salida = await nodo(estado(pedido=PEDIDO, pasos_supervisor=1, aportes=[aporte_inv(obs)]))
    assert salida["siguiente"] == "investigador"
    assert "forzar=true" in salida["instruccion_actual"]
    assert "R3/R5 NO se cumplen" in salida["decisiones"][0].evaluacion


async def test_el_supervisor_ve_resumenes_no_llamadas_internas() -> None:
    llm = LLMMock(politica_estructurada=politica_supervisor_estructurada)
    await crear_nodo_supervisor(llm)(estado(pedido=PEDIDO, pasos_supervisor=1, aportes=[aporte_inv(["obs X"])]))
    visto = " ".join(str(m.content) for m in llm.vistos[-1])
    assert "APORTE #1 | agente=investigador" in visto and "fuentes=1 (memoria=0, web=1)" in visto
    assert "Observaciones del chequeo por código: obs X" in visto
    assert "SIN RESULTADOS EN MEMORIA" not in visto  # el resultado crudo de la herramienta no llega
    assert "R1 Cobertura" in visto and "esta es la número 2" in visto


# ------------------------------------------------------------- reglas duras
async def test_analista_sin_datos_se_corrige_a_investigador() -> None:
    nodo = crear_nodo_supervisor(llm_que_decide(DecisionSupervisor(evaluacion="x", siguiente="analista", instruccion="calculá")))
    salida = await nodo(estado(pedido=PEDIDO))
    assert salida["siguiente"] == "investigador"
    assert "regla dura" in salida["decisiones"][0].evaluacion


async def test_finalizar_sin_aportes_se_corrige_a_investigador() -> None:
    nodo = crear_nodo_supervisor(llm_que_decide(DecisionSupervisor(evaluacion="listo", siguiente="FINALIZAR")))
    salida = await nodo(estado(pedido=PEDIDO))
    assert salida["siguiente"] == "investigador" and salida["instruccion_actual"]


async def test_si_el_llm_falla_decide_por_reglas() -> None:
    nodo = crear_nodo_supervisor(llm_que_decide(ValueError("salida mal formada")))
    salida = await nodo(estado(pedido=PEDIDO, pasos_supervisor=1, aportes=[aporte_inv()]))
    assert salida["siguiente"] == "analista"
    assert "se decidió por reglas" in salida["decisiones"][0].evaluacion


# ---------------------------------------------------------- freno de pasos
async def test_freno_fuerza_finalizar_sin_consultar_al_llm() -> None:
    """Escenario 8: en la última decisión permitida se finaliza por código."""
    nodo = crear_nodo_supervisor(llm_que_decide(RuntimeError("no deberían consultarme")), max_pasos=2)
    salida = await nodo(estado(pedido=PEDIDO, pasos_supervisor=1, aportes=[aporte_inv()]))
    d = salida["decisiones"][0]
    assert salida["siguiente"] == "FINALIZAR" and d.forzada and d.paso == 2
    assert "FRENO DE PASOS" in d.evaluacion


async def test_freno_por_defecto_es_max_pasos_supervisor() -> None:
    nodo = crear_nodo_supervisor(llm_que_decide(RuntimeError("no")))
    salida = await nodo(estado(pedido=PEDIDO, pasos_supervisor=config.MAX_PASOS_SUPERVISOR - 1, aportes=[aporte_inv()]))
    assert salida["decisiones"][0].forzada and salida["pasos_supervisor"] == config.MAX_PASOS_SUPERVISOR


# ------------------------------------------- reglas duras surgidas de la corrida real
async def test_no_finaliza_con_presupuesto_sin_calculo() -> None:
    """Corrida real 1: el LLM quiso finalizar sin pasar por el analista → el código lo impide."""
    nodo = crear_nodo_supervisor(llm_que_decide(DecisionSupervisor(evaluacion="faltan precios exactos", siguiente="FINALIZAR")))
    salida = await nodo(estado(pedido=PEDIDO, pasos_supervisor=1, aportes=[aporte_inv()]))
    assert salida["siguiente"] == "analista" and "regla dura: hay presupuesto" in salida["decisiones"][0].evaluacion


async def test_sin_presupuesto_si_puede_finalizar_sin_analista() -> None:
    nodo = crear_nodo_supervisor(llm_que_decide(DecisionSupervisor(evaluacion="ok", siguiente="FINALIZAR")))
    salida = await nodo(estado(pedido=PEDIDO.model_copy(update={"presupuesto": None}), pasos_supervisor=1,
                               aportes=[aporte_inv()]))
    assert salida["siguiente"] == "FINALIZAR"


async def test_investigador_no_trabaja_una_tercera_vez() -> None:
    """Corrida real 1: el LLM pidió 3 rondas de búsqueda → la tercera se redirige al analista."""
    pedir_mas = DecisionSupervisor(evaluacion="faltan precios exactos", siguiente="investigador", instruccion="buscá más")
    dos_rondas = [aporte_inv(), aporte_inv()]
    salida = await crear_nodo_supervisor(llm_que_decide(pedir_mas))(estado(pedido=PEDIDO, pasos_supervisor=2, aportes=dos_rondas))
    assert salida["siguiente"] == "analista" and "ya trabajó 2 veces" in salida["decisiones"][0].evaluacion
    sin_presupuesto = PEDIDO.model_copy(update={"presupuesto": None})
    salida = await crear_nodo_supervisor(llm_que_decide(pedir_mas))(estado(pedido=sin_presupuesto, pasos_supervisor=2,
                                                                           aportes=dos_rondas))
    assert salida["siguiente"] == "FINALIZAR"


def test_rubrica_acepta_precios_de_referencia() -> None:
    assert "NO exijas precios exactos" in PROMPT_SUPERVISOR and "UNA ronda de corrección" in PROMPT_SUPERVISOR
