"""Tests del agente analista: investigador (mock) → analista (mock), con las herramientas reales."""

from datetime import date
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

import config
from agents.analyst_agent import crear_agente_analista, crear_nodo_analista
from agents.analyst_tools import crear_herramientas_analista
from agents.memory import MemoriaBusquedas
from agents.mock import politica_analista, politica_investigador
from agents.research_agent import crear_agente_investigador, crear_nodo_investigador
from agents.research_tools import BuscadorMock, crear_herramientas_investigador
from llm import LLMMock, guion
from state import Aporte, OrquestadorState, PedidoViaje, estado_inicial

PREGUNTA = "PREGUNTA ORIGINAL: Quiero ir 7 días a Bariloche en enero..."
PEDIDO = PedidoViaje(origen="Mendoza", destino="Bariloche", fecha_ida=date(2027, 1, 10),
                     fecha_vuelta=date(2027, 1, 16), dias=7, personas=2, presupuesto=800_000, moneda="ARS")
BUSCAR_TODO = "Buscá pasajes, alojamiento, clima, actividades y comida, y la cotización del dólar."
ANALIZAR = "Validá el viaje y calculá el presupuesto con los precios encontrados."


@pytest.fixture(autouse=True)
def hoy_fijo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "FECHA_REFERENCIA", date(2026, 11, 5))


async def investigar(memoria: MemoriaBusquedas, buscador: BuscadorMock, pedido: PedidoViaje) -> OrquestadorState:
    """Corre el investigador mock y devuelve un estado con su aporte, listo para el analista."""
    agente = crear_agente_investigador(LLMMock(politica=politica_investigador),
                                       crear_herramientas_investigador(memoria, buscador))
    s: Any = estado_inicial(PREGUNTA)
    s.update(pedido=pedido, instruccion_actual=BUSCAR_TODO, pasos_supervisor=1)
    s["aportes"] = (await crear_nodo_investigador(agente)(s))["aportes"]
    s.update(instruccion_actual=ANALIZAR, pasos_supervisor=2)
    return s


def nodo_analista(llm: LLMMock | None = None) -> tuple[Any, LLMMock]:
    llm = llm or LLMMock(politica=politica_analista)
    return crear_nodo_analista(crear_agente_analista(llm, crear_herramientas_analista())), llm


async def test_flujo_completo_entra_en_el_presupuesto(memoria: MemoriaBusquedas, buscador: BuscadorMock) -> None:
    """Escenario 1: valida, calcula y concluye que entra."""
    s = await investigar(memoria, buscador, PEDIDO)
    nodo, _ = nodo_analista()
    salida = await nodo(s)
    aporte: Aporte = salida["aportes"][0]

    assert set(salida) == {"aportes", "messages"} and aporte.agente == "analista" and aporte.ronda == 2
    assert aporte.herramientas == ["validar_viaje", "calcular_presupuesto"]
    assert aporte.observaciones == []
    assert "VÁLIDO" in aporte.contenido and "TOTAL: $ 785.100" in aporte.contenido
    assert "ENTRA en el presupuesto" in aporte.contenido and "dólar tarjeta" in aporte.contenido
    # Las fuentes de los precios y de la cotización quedan registradas.
    assert {f.tema for f in aporte.fuentes} >= {"pasajes", "alojamiento", "cambio"}


async def test_presupuesto_que_no_alcanza_calcula_alternativa(memoria: MemoriaBusquedas, buscador: BuscadorMock) -> None:
    """Escenario 3: con $400.000 no entra; calcula una alternativa con menos días."""
    s = await investigar(memoria, buscador, PEDIDO.model_copy(update={"presupuesto": 400_000}))
    aporte: Aporte = (await nodo_analista()[0](s))["aportes"][0]
    calculos = [ll for ll in aporte.llamadas if ll.herramienta == "calcular_presupuesto"]
    assert len(calculos) == 2
    assert "NO ENTRA" in aporte.contenido and "alternativa" in aporte.contenido
    assert calculos[1].argumentos["noches"] < 6 and "extras" not in calculos[1].argumentos


async def test_fechas_invalidas_no_calcula(memoria: MemoriaBusquedas, buscador: BuscadorMock) -> None:
    """Escenario 4: vuelta antes que la ida → validar_viaje lo rechaza y no se calcula nada."""
    malo = PedidoViaje(**{**PEDIDO.model_dump(), "fecha_ida": date(2027, 1, 16), "fecha_vuelta": date(2027, 1, 10)})
    s = await investigar(memoria, buscador, malo)
    aporte: Aporte = (await nodo_analista()[0](s))["aportes"][0]
    assert aporte.herramientas == ["validar_viaje"]
    assert "INVÁLIDO" in aporte.contenido and "DESPUÉS de la ida" in aporte.contenido
    assert aporte.observaciones == []  # rechazar un viaje imposible es hacer bien el trabajo


async def test_contexto_acotado(memoria: MemoriaBusquedas, buscador: BuscadorMock) -> None:
    """El analista ve: instrucción + pedido + hallazgos. NO ve la pregunta, el historial ni las llamadas internas."""
    s = await investigar(memoria, buscador, PEDIDO)
    s["messages"].append(AIMessage(content="SECRETO del historial", name="supervisor"))
    nodo, llm = nodo_analista()
    await nodo(s)
    primera = llm.vistos[0]
    assert [type(m) for m in primera] == [SystemMessage, HumanMessage]
    texto = str(primera[1].content)
    assert texto.startswith("INSTRUCCIÓN DEL SUPERVISOR:\n" + ANALIZAR)
    assert "DATOS DEL PEDIDO" in texto and "HALLAZGOS DEL INVESTIGADOR" in texto and "## Hallazgos" in texto
    todo = " ".join(str(m.content) for vistos in llm.vistos for m in vistos)
    assert "SECRETO" not in todo and "PREGUNTA ORIGINAL" not in todo
    assert "SIN RESULTADOS EN MEMORIA" not in todo  # las llamadas internas del investigador no llegan


async def test_analista_que_no_calcula_queda_observado(memoria: MemoriaBusquedas, buscador: BuscadorMock) -> None:
    s = await investigar(memoria, buscador, PEDIDO)
    perezoso = LLMMock(politica=guion([AIMessage(content="Calculo que sale unos $700.000, entra.")]))
    aporte: Aporte = (await nodo_analista(perezoso)[0](s))["aportes"][0]
    assert aporte.observaciones == ["El analista no usó ninguna herramienta: sus números no están calculados."]


async def test_sin_precios_no_inventa(memoria: MemoriaBusquedas) -> None:
    """Si el investigador no consiguió precios (Tavily caído), el analista no inventa números."""
    s = await investigar(memoria, BuscadorMock(fallar=True), PEDIDO)
    aporte: Aporte = (await nodo_analista()[0](s))["aportes"][0]
    assert aporte.herramientas == ["validar_viaje"]
    assert "Datos que faltan" in aporte.contenido and "pasaje" in aporte.contenido
    assert any("no se calculó el presupuesto" in o for o in aporte.observaciones)
