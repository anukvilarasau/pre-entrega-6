"""Tests de state.py: herencia, validación Pydantic y "cada nodo escribe solo su parte"."""

import typing

import pytest
from langgraph.graph import END, START, MessagesState, StateGraph
from pydantic import ValidationError

from state import (
    Aporte,
    Fuente,
    DecisionSupervisor,
    ErrorDeEstado,
    OrquestadorState,
    estado_inicial,
    validar_actualizacion,
)


URL = "https://www.ejemplo.com/vuelos-mendoza-bariloche"


def web(ref: str = URL, tema: str = "pasajes") -> Fuente:
    return Fuente(ref=ref, origen="web", tema=tema)


def aporte(**cambios: object) -> Aporte:
    datos = {"agente": "investigador", "ronda": 1, "instruccion": "buscar pasajes",
             "contenido": "Mendoza-Bariloche ida y vuelta ~ $250.000 por persona",
             "fuentes": [web()], "herramientas": ["buscar_en_web"]}
    datos.update(cambios)
    return Aporte(**datos)


def test_estado_hereda_de_messages_state() -> None:
    # Los TypedDict no admiten issubclass(): se mira la clase base declarada.
    assert MessagesState in OrquestadorState.__orig_bases__
    campos = typing.get_type_hints(OrquestadorState)
    assert "messages" in campos and "aportes" in campos


def test_aporte_valido_y_fuentes_sin_repetidos() -> None:
    mem = Fuente(ref="mem-0123456789ab", origen="memoria", tema="clima", url="https://clima.ejemplo.com/bariloche")
    a = aporte(fuentes=[web(), web(), mem], herramientas=["buscar_en_memoria", "buscar_en_web"])
    assert [f.ref for f in a.fuentes] == [URL, "mem-0123456789ab"]
    assert [f.origen for f in a.fuentes] == ["web", "memoria"]


@pytest.mark.parametrize("ref", ["esto no es una url", "ftp://archivos.com/x", "mem-0123456789ab"])
def test_fuente_web_exige_url_http(ref: str) -> None:
    with pytest.raises(ValidationError, match="URL inválida"):
        Fuente(ref=ref, origen="web")


@pytest.mark.parametrize("ref", [URL, "mem-123", "mem-ZZZZZZZZZZZZ"])
def test_fuente_memoria_exige_id_mem(ref: str) -> None:
    with pytest.raises(ValidationError, match="ID de memoria inválido"):
        Fuente(ref=ref, origen="memoria")


def test_fuente_rechaza_tema_u_origen_inventado() -> None:
    with pytest.raises(ValidationError):
        Fuente(ref=URL, origen="web", tema="gastronomia")
    with pytest.raises(ValidationError):
        Fuente(ref=URL, origen="diario")


def test_aporte_rechaza_herramienta_de_otro_rol() -> None:
    with pytest.raises(ValidationError, match="no puede usar"):
        aporte(herramientas=["calcular_presupuesto"])  # es del analista
    with pytest.raises(ValidationError, match="no puede usar"):
        aporte(agente="analista", fuentes=[], herramientas=["buscar_en_web"])  # es del investigador


def test_aporte_rechaza_campos_extra() -> None:
    with pytest.raises(ValidationError):
        aporte(inventado="x")


def test_decision_exige_instruccion_si_delega() -> None:
    with pytest.raises(ValidationError, match="instrucción"):
        DecisionSupervisor(evaluacion="falta todo", siguiente="investigador", instruccion="  ")
    assert DecisionSupervisor(evaluacion="ok", siguiente="FINALIZAR").instruccion == ""


def test_decision_rechaza_destino_inexistente() -> None:
    with pytest.raises(ValidationError):
        DecisionSupervisor(evaluacion="x", siguiente="programador", instruccion="y")


def test_nodo_no_puede_escribir_campos_ajenos() -> None:
    with pytest.raises(ErrorDeEstado, match="campos ajenos"):
        validar_actualizacion("investigador", {"siguiente": "FINALIZAR"})
    with pytest.raises(ErrorDeEstado, match="campos ajenos"):
        validar_actualizacion("sintesis", {"aportes": []})


def test_nodo_no_puede_firmar_como_otro() -> None:
    with pytest.raises(ErrorDeEstado, match="firmar"):
        validar_actualizacion("analista", {"aportes": [aporte()]})


def test_dict_invalido_se_convierte_en_error_de_estado() -> None:
    with pytest.raises(ErrorDeEstado, match="inválidos"):
        validar_actualizacion("investigador", {"aportes": [{"agente": "investigador"}]})


def test_dict_valido_se_convierte_en_modelo() -> None:
    out = validar_actualizacion("investigador", {"aportes": [aporte().model_dump()]})
    assert isinstance(out["aportes"][0], Aporte)


def test_estado_inicial_rechaza_pregunta_vacia() -> None:
    with pytest.raises(ValueError):
        estado_inicial("   ")


async def test_reductor_suma_aportes_en_un_grafo_real() -> None:
    """Dos nodos agregan aportes: el reductor operator.add los acumula, no los pisa."""

    async def inv(_: OrquestadorState) -> dict:
        return validar_actualizacion("investigador", {"aportes": [aporte()]})

    async def ana(_: OrquestadorState) -> dict:
        return validar_actualizacion("analista", {"aportes": [aporte(
            agente="analista", fuentes=[], herramientas=["validar_viaje", "calcular_presupuesto"], contenido="El viaje entra en el presupuesto")]})

    g = StateGraph(OrquestadorState)
    g.add_node("investigador", inv)
    g.add_node("analista", ana)
    g.add_edge(START, "investigador")
    g.add_edge("investigador", "analista")
    g.add_edge("analista", END)
    final = await g.compile().ainvoke(estado_inicial("¿Me alcanza para Bariloche?"))
    assert [a.agente for a in final["aportes"]] == ["investigador", "analista"]
    assert final["messages"][0].content == "¿Me alcanza para Bariloche?"


def test_pedido_acepta_fechas_imposibles_para_que_las_valide_el_analista() -> None:
    from datetime import date

    from state import PedidoViaje

    p = PedidoViaje(destino="Bariloche", fecha_ida=date(2027, 1, 16), fecha_vuelta=date(2027, 1, 10))
    assert p.fecha_vuelta < p.fecha_ida  # se registra tal cual lo dijo el usuario
    with pytest.raises(ValidationError):
        PedidoViaje(destino="Bariloche", moneda="EUR")  # pero los tipos sí se controlan
    with pytest.raises(ValidationError):
        PedidoViaje(inventado=1)


def test_contexto_especialista_no_incluye_pregunta() -> None:
    from datetime import date

    from state import PedidoViaje, contexto_especialista

    texto = contexto_especialista("Buscá clima", PedidoViaje(destino="Bariloche", fecha_ida=date(2027, 1, 10),
                                                            presupuesto=800000, moneda="ARS"))
    assert texto.startswith("INSTRUCCIÓN DEL SUPERVISOR:\nBuscá clima")
    assert "- Destino: Bariloche" in texto and "- Mes: enero" in texto
    assert "- Presupuesto total: 800.000 ARS" in texto and "- Origen: no informado" in texto


def test_solo_el_supervisor_escribe_el_pedido() -> None:
    with pytest.raises(ErrorDeEstado, match="campos ajenos"):
        validar_actualizacion("investigador", {"pedido": {"destino": "Bariloche"}})
    out = validar_actualizacion("supervisor", {"pedido": {"destino": "Bariloche"}})
    assert out["pedido"].destino == "Bariloche"
