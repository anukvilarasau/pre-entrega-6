"""Tests de agents/research_tools.py: memoria primero, web después, guardado y errores."""

from typing import Any

import pytest
from langchain_core.messages import ToolMessage
from pydantic import ValidationError

from agents.memory import MemoriaBusquedas
from agents.research_tools import BuscadorMock, crear_herramientas_investigador
from state import Fuente

ARGS = {"consulta": "precio pasaje vuelo Mendoza Bariloche ida y vuelta", "destino": "Bariloche", "tema": "pasajes"}


async def llamar(tool: Any, args: dict[str, Any]) -> ToolMessage:
    """Invoca la herramienta como lo hace el agente (tool_call) y devuelve el ToolMessage."""
    return await tool.ainvoke({"type": "tool_call", "id": "llamada-1", "name": tool.name, "args": args})


def herramientas(memoria: MemoriaBusquedas, buscador: BuscadorMock) -> tuple[Any, Any]:
    mem, web = crear_herramientas_investigador(memoria, buscador)
    assert (mem.name, web.name) == ("buscar_en_memoria", "buscar_en_web")
    return mem, web


async def test_memoria_vacia_pide_ir_a_la_web(memoria: MemoriaBusquedas, buscador: BuscadorMock) -> None:
    mem, _ = herramientas(memoria, buscador)
    msg = await llamar(mem, ARGS)
    assert str(msg.content).startswith("SIN RESULTADOS EN MEMORIA")
    assert "no hay nada guardado" in str(msg.content)
    assert msg.artifact == []


async def test_web_devuelve_urls_y_guarda_en_memoria(memoria: MemoriaBusquedas, buscador: BuscadorMock) -> None:
    _, web = herramientas(memoria, buscador)
    msg = await llamar(web, ARGS)
    assert str(msg.content).startswith("WEB (Tavily)")
    assert buscador.llamadas == ["precio pasaje vuelo Mendoza Bariloche ida y vuelta"]
    assert msg.artifact and all(isinstance(f, Fuente) and f.origen == "web" for f in msg.artifact)
    assert all(f.ref.startswith("https://") and f.tema == "pasajes" for f in msg.artifact)
    assert memoria.cantidad() == len(msg.artifact)


async def test_segunda_vez_sale_de_memoria_sin_llamar_a_tavily(memoria: MemoriaBusquedas, buscador: BuscadorMock) -> None:
    """Escenario 5: la misma búsqueda dos veces; la segunda no toca la web."""
    mem, web = herramientas(memoria, buscador)
    await llamar(web, ARGS)
    msg = await llamar(mem, ARGS)
    assert str(msg.content).startswith("MEMORIA")
    assert msg.artifact and all(f.origen == "memoria" and f.ref.startswith("mem-") for f in msg.artifact)
    # Aunque el LLM insista con la web, la guarda evita la llamada a Tavily.
    msg_web = await llamar(web, ARGS)
    assert str(msg_web.content).startswith("NO SE BUSCÓ EN LA WEB")
    assert all(f.origen == "memoria" for f in msg_web.artifact)
    assert len(buscador.llamadas) == 1


async def test_forzar_se_bloquea_si_la_memoria_tiene_precios(memoria: MemoriaBusquedas, buscador: BuscadorMock) -> None:
    """Arreglo del escenario 5 real: el LLM forzaba la web teniendo precios vigentes en memoria."""
    _, web = herramientas(memoria, buscador)
    await llamar(web, ARGS)  # guarda resultados CON precios ($480.000, $70.000...)
    msg = await llamar(web, {**ARGS, "forzar": True})
    assert str(msg.content).startswith("NO SE BUSCÓ EN LA WEB: la memoria ya tiene datos vigentes CON PRECIOS")
    assert all(f.origen == "memoria" for f in msg.artifact)
    assert len(buscador.llamadas) == 1  # Tavily no se volvió a llamar


async def test_forzar_va_a_la_web_si_la_memoria_no_tiene_precios(memoria: MemoriaBusquedas, buscador: BuscadorMock) -> None:
    """Escenario 6: lo guardado ('cómo llegar') no trae precios → forzar sí busca en la web."""
    _, web = herramientas(memoria, buscador)
    sin_precio = {"consulta": "como llegar rutas aerolineas Mendoza Bariloche", "destino": "Bariloche", "tema": "pasajes"}
    await llamar(web, sin_precio)
    msg = await llamar(web, {**sin_precio, "forzar": True})
    assert str(msg.content).startswith("WEB (Tavily)")
    assert len(buscador.llamadas) == 2
    assert memoria.cantidad() == 1  # la misma página se actualizó, no se duplicó


async def test_tavily_caido_devuelve_error_sin_romper(memoria: MemoriaBusquedas) -> None:
    """Escenario 7: la herramienta devuelve el error como texto."""
    _, web = herramientas(memoria, BuscadorMock(fallar=True))
    msg = await llamar(web, ARGS)
    assert str(msg.content).startswith("ERROR al buscar en la web")
    assert "No inventes datos" in str(msg.content)
    assert msg.artifact == [] and memoria.cantidad() == 0


async def test_web_sin_resultados(memoria: MemoriaBusquedas, buscador: BuscadorMock) -> None:
    _, web = herramientas(memoria, buscador)
    msg = await llamar(web, {"consulta": "submarinismo en glaciares", "destino": "Bariloche", "tema": "otro"})
    assert str(msg.content).startswith("SIN RESULTADOS EN LA WEB")


@pytest.mark.parametrize("malos", [
    {**ARGS, "tema": "gastronomia"},  # tema inexistente
    {**ARGS, "consulta": "x"},        # consulta demasiado corta
    {k: v for k, v in ARGS.items() if k != "destino"},  # falta el destino
])
async def test_argumentos_invalidos_se_rechazan(memoria: MemoriaBusquedas, buscador: BuscadorMock, malos: dict[str, Any]) -> None:
    mem, _ = herramientas(memoria, buscador)
    with pytest.raises(ValidationError):
        await mem.ainvoke(malos)


async def test_cada_tema_del_mock_tiene_datos(memoria: MemoriaBusquedas, buscador: BuscadorMock) -> None:
    _, web = herramientas(memoria, buscador)
    consultas = {
        "pasajes": "precio micro Mendoza Bariloche", "alojamiento": "precio hostel habitacion doble por noche",
        "clima": "clima en enero", "actividades": "que hacer excursiones", "cambio": "cotizacion dolar tarjeta MEP",
    }
    for tema, consulta in consultas.items():
        msg = await llamar(web, {"consulta": consulta, "destino": "Bariloche", "tema": tema})
        assert str(msg.content).startswith("WEB"), tema
