"""Tests del agente investigador (create_react_agent + memoria + Tavily simulado, sin claves)."""

from datetime import date
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

import config
from agents.memory import MemoriaBusquedas
from agents.mock import politica_investigador
from agents.research_agent import crear_agente_investigador, crear_nodo_investigador
from agents.research_tools import BuscadorMock, crear_herramientas_investigador
from llm import LLMMock, guion, nueva_llamada
from state import (
    Aporte,
    ErrorDeEstado,
    OrquestadorState,
    PedidoViaje,
    contexto_especialista,
    estado_inicial,
    validar_actualizacion,
)

INSTRUCCION = "Buscá precios de pasajes y alojamiento, y el clima en las fechas del viaje."
PREGUNTA = "PREGUNTA ORIGINAL: quiero ir 7 días a Bariloche, ¿nos alcanza?"
PEDIDO = PedidoViaje(origen="Mendoza", destino="Bariloche", fecha_ida=date(2027, 1, 10),
                     fecha_vuelta=date(2027, 1, 16), dias=7, personas=2, presupuesto=800_000, moneda="ARS")
SIN_PRECIOS = "Buscá pasajes.\n- pasajes: como llegar rutas aerolineas"


def armar_nodo(memoria: MemoriaBusquedas, buscador: BuscadorMock, llm: LLMMock | None = None) -> tuple[Any, LLMMock]:
    llm = llm or LLMMock(politica=politica_investigador)
    agente = crear_agente_investigador(llm, crear_herramientas_investigador(memoria, buscador))
    return crear_nodo_investigador(agente), llm


TEMAS = ["pasajes", "alojamiento", "clima"]


def estado(instruccion: str = INSTRUCCION, temas: list[str] | None = None) -> OrquestadorState:
    s = estado_inicial(PREGUNTA)
    s.update(instruccion_actual=instruccion, pasos_supervisor=1, pedido=PEDIDO, temas_pedidos=temas or TEMAS)
    return s  # type: ignore[return-value]


async def test_primera_vez_va_a_la_web_y_arma_un_aporte_valido(memoria: MemoriaBusquedas, buscador: BuscadorMock) -> None:
    nodo, _ = armar_nodo(memoria, buscador)
    salida = await nodo(estado())
    aporte: Aporte = salida["aportes"][0]

    assert set(salida) == {"aportes", "messages"}            # solo escribe su parte
    assert aporte.agente == "investigador" and aporte.ronda == 1
    assert aporte.instruccion == INSTRUCCION
    assert aporte.herramientas == ["buscar_en_memoria", "buscar_en_web"]
    assert {ll.herramienta for ll in aporte.llamadas} == {"buscar_en_memoria", "buscar_en_web"}
    assert aporte.fuentes and all(f.origen == "web" for f in aporte.fuentes)
    assert {f.tema for f in aporte.fuentes} == {"pasajes", "alojamiento", "clima"}
    assert aporte.observaciones == []                         # pasó el chequeo por código
    assert "## Hallazgos" in aporte.contenido and "$" in aporte.contenido
    assert salida["messages"][0].name == "investigador"
    assert len(buscador.llamadas) == 3


async def test_usa_origen_y_destino_del_pedido(memoria: MemoriaBusquedas, buscador: BuscadorMock) -> None:
    """La instrucción no dice origen ni destino: los toma de DATOS DEL PEDIDO."""
    nodo, _ = armar_nodo(memoria, buscador)
    aporte: Aporte = (await nodo(estado("Buscá precios de pasajes.")))["aportes"][0]
    args = aporte.llamadas[0].argumentos
    assert args["destino"] == "Bariloche" and "Mendoza" in args["consulta"]


async def test_segunda_vez_sale_todo_de_memoria(memoria: MemoriaBusquedas, buscador: BuscadorMock) -> None:
    """Escenario 5: la misma instrucción dos veces → la segunda no llama a Tavily."""
    nodo, _ = armar_nodo(memoria, buscador)
    await nodo(estado())
    llamadas_web = len(buscador.llamadas)

    segunda: Aporte = (await nodo(estado()))["aportes"][0]
    assert len(buscador.llamadas) == llamadas_web               # Tavily no se volvió a usar
    assert segunda.herramientas == ["buscar_en_memoria"]
    assert segunda.fuentes and all(f.origen == "memoria" and f.ref.startswith("mem-") for f in segunda.fuentes)
    assert all(f.url and f.url.startswith("https://") for f in segunda.fuentes)  # conserva la URL original
    assert segunda.observaciones == []


async def test_contexto_acotado_solo_ve_su_instruccion(memoria: MemoriaBusquedas, buscador: BuscadorMock) -> None:
    """Contaminación de contexto: el LLM del investigador ve su instrucción + DATOS DEL PEDIDO,
    pero NO la pregunta original ni el historial general."""
    nodo, llm = armar_nodo(memoria, buscador)
    s = estado()
    s["messages"].append(AIMessage(content="SECRETO del supervisor que no debe filtrarse", name="supervisor"))
    await nodo(s)
    primera = llm.vistos[0]
    assert isinstance(primera[0], SystemMessage) and "INVESTIGADOR" in str(primera[0].content)
    assert [type(m) for m in primera] == [SystemMessage, HumanMessage]
    assert primera[1].content == contexto_especialista(INSTRUCCION, PEDIDO, "TEMAS A BUSCAR: pasajes, alojamiento, clima")
    texto = str(primera[1].content)
    assert INSTRUCCION in texto and "DATOS DEL PEDIDO" in texto
    assert "- Origen: Mendoza" in texto and "- Fecha de ida: 2027-01-10" in texto and "- Personas: 2" in texto
    todo = " ".join(str(m.content) for vistos in llm.vistos for m in vistos)
    assert "SECRETO" not in todo
    assert "PREGUNTA ORIGINAL" not in todo


async def test_sin_herramientas_queda_observado(memoria: MemoriaBusquedas, buscador: BuscadorMock) -> None:
    llm = LLMMock(politica=guion([AIMessage(content="Un pasaje sale $100 (lo sé de memoria).")]))
    nodo, _ = armar_nodo(memoria, buscador, llm)
    aporte: Aporte = (await nodo(estado()))["aportes"][0]
    assert any("no usó ninguna herramienta" in o for o in aporte.observaciones)
    assert aporte.fuentes == []


async def test_busqueda_sin_precios_queda_observada(memoria: MemoriaBusquedas, buscador: BuscadorMock) -> None:
    """Base del escenario 6: la búsqueda de pasajes no trae precios → el chequeo lo marca."""
    nodo, _ = armar_nodo(memoria, buscador)
    aporte: Aporte = (await nodo(estado(SIN_PRECIOS, ["pasajes"])))["aportes"][0]
    assert any("precios de pasajes" in o for o in aporte.observaciones)


async def test_refinamiento_con_forzar_trae_precios(memoria: MemoriaBusquedas, buscador: BuscadorMock) -> None:
    nodo, _ = armar_nodo(memoria, buscador)
    await nodo(estado(SIN_PRECIOS, ["pasajes"]))
    rehecho: Aporte = (await nodo(estado(
        "Rehacer con forzar: faltan precios.\n- pasajes: precio pasaje vuelo micro Mendoza Bariloche", ["pasajes"]
    )))["aportes"][0]
    assert rehecho.herramientas == ["buscar_en_web"]
    assert rehecho.observaciones == [] and "$" in rehecho.contenido


async def test_tavily_caido_no_rompe_y_queda_observado(memoria: MemoriaBusquedas) -> None:
    """Escenario 7: la web falla; el nodo termina igual y deja constancia."""
    nodo, _ = armar_nodo(memoria, BuscadorMock(fallar=True))
    aporte: Aporte = (await nodo(estado()))["aportes"][0]
    assert any(o.startswith("Falló una búsqueda") for o in aporte.observaciones)
    assert "## Datos que faltan" in aporte.contenido and "ERROR" in aporte.contenido
    assert aporte.fuentes == []


async def test_corte_por_limite_de_pasos_no_rompe(memoria: MemoriaBusquedas, buscador: BuscadorMock,
                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    """Freno 3: un investigador que nunca termina se corta y se rescata lo que hizo."""
    monkeypatch.setattr(config, "RECURSION_LIMIT_ESPECIALISTA", 5)
    infinito = LLMMock(politica=lambda _m, _h: AIMessage(content="otra vuelta", tool_calls=[
        nueva_llamada("buscar_en_memoria", {"consulta": "clima en enero", "destino": "Bariloche", "tema": "clima"})]))
    nodo, _ = armar_nodo(memoria, buscador, infinito)
    aporte: Aporte = (await nodo(estado()))["aportes"][0]
    assert aporte.observaciones[0].startswith("El ciclo ReAct se cortó")
    assert aporte.llamadas  # lo que alcanzó a hacer quedó registrado


async def test_firma_falsa_es_imposible(memoria: MemoriaBusquedas, buscador: BuscadorMock) -> None:
    """La salida del investigador no puede hacerse pasar por la de otro nodo."""
    nodo, _ = armar_nodo(memoria, buscador)
    salida = await nodo(estado())
    with pytest.raises(ErrorDeEstado):
        validar_actualizacion("analista", salida)


# --------------------------------------------- arreglos 2 y 3 (escenario 5 real)
async def test_cobertura_por_tema_la_calcula_el_codigo(memoria: MemoriaBusquedas, buscador: BuscadorMock) -> None:
    nodo, _ = armar_nodo(memoria, buscador)
    aporte: Aporte = (await nodo(estado()))["aportes"][0]
    cob = {c.tema: c for c in aporte.cobertura}
    assert set(cob) == {"pasajes", "alojamiento", "clima"}
    assert cob["pasajes"].fuentes == 2 and cob["pasajes"].web == 2 and cob["pasajes"].con_precios
    assert cob["clima"].con_precios is False  # el clima no trae montos
    segunda: Aporte = (await nodo(estado()))["aportes"][0]
    assert all(c.memoria == c.fuentes and c.web == 0 for c in segunda.cobertura)


async def test_el_chequeo_usa_los_temas_estructurados_no_las_palabras(memoria: MemoriaBusquedas,
                                                                    buscador: BuscadorMock) -> None:
    """Corrida real 5: 'No repitas búsquedas de pasajes ni alojamiento' generaba una observación falsa."""
    nodo, _ = armar_nodo(memoria, buscador)
    instr = "Buscá el clima. No repitas búsquedas de pasajes ni alojamiento."
    aporte: Aporte = (await nodo(estado(instr, ["clima"])))["aportes"][0]
    assert aporte.observaciones == []
    assert {ll.argumentos["tema"] for ll in aporte.llamadas} == {"clima"}


def test_tema_pedido_y_no_buscado_queda_observado() -> None:
    from agents.research_agent import chequear_investigacion

    assert chequear_investigacion(["cambio"], "x", [], []) == [
        "El investigador no usó ninguna herramienta: sus datos no tienen respaldo.",
        "Se pidió el tema 'cambio' pero no se hizo ninguna búsqueda sobre él."]
