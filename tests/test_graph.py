"""Tests de punta a punta: el grafo completo en modo mock, un test por escenario (sin claves)."""

from datetime import date
from pathlib import Path

import pytest

import config
from graph import Componentes, construir_grafo, consultar, crear_componentes, diagrama_mermaid
from tracing import Traza

DEMO = ("Quiero ir 7 días a Bariloche en enero, somos 2 personas, salimos desde Mendoza y tenemos "
        "800.000 pesos. ¿Me armás el viaje y me decís si nos alcanza?")
RUTA_COMPLETA = "supervisor → investigador → supervisor → analista → supervisor → sintesis"


@pytest.fixture(autouse=True)
def hoy_fijo(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(config, "FECHA_REFERENCIA", date(2026, 11, 5))
    monkeypatch.setattr(config, "TRACES_DIR", tmp_path / "traces")  # los tests no ensucian traces/


def mock(tmp_path: Path, fallar_web: bool = False) -> Componentes:
    return crear_componentes(modo_mock=True, fallar_web=fallar_web, ruta_memoria=tmp_path / "chroma")


async def correr(c: Componentes, pregunta: str, max_pasos: int | None = None) -> Traza:
    grafo = construir_grafo(c, max_pasos=max_pasos)
    return await consultar(grafo, pregunta, escenario="test", modo_mock=True, max_pasos=max_pasos, guardar=False)


def test_grafo_tiene_los_nodos_y_aristas_de_la_topologia(tmp_path: Path) -> None:
    g = construir_grafo(mock(tmp_path)).get_graph()
    assert set(g.nodes) == {"__start__", "supervisor", "investigador", "analista", "sintesis", "__end__"}
    aristas = {(a.source, a.target, a.conditional) for a in g.edges}
    assert ("__start__", "supervisor", False) in aristas
    assert {("supervisor", n, True) for n in ("investigador", "analista", "sintesis")} <= aristas
    assert {("investigador", "supervisor", False), ("analista", "supervisor", False), ("sintesis", "__end__", False)} <= aristas
    texto = diagrama_mermaid(construir_grafo(mock(tmp_path)))
    assert texto.startswith("---") and "supervisor -.-> analista" in texto


async def test_escenario_1_flujo_completo(tmp_path: Path) -> None:
    t = await correr(mock(tmp_path), DEMO)
    assert t.estado == "completo" and t.texto_ruta() == RUTA_COMPLETA
    m = t.metricas
    assert m["pasos_supervisor"] == 3 and m["validacion"] == "aprobada" and m["presupuesto_entra"] is True
    assert "## Itinerario" in t.respuesta_final and "- Día 7 (16/01)" in t.respuesta_final
    assert "TOTAL: $ 785.100" in t.respuesta_final and "## Fuentes" in t.respuesta_final
    planes = [ln.split(": ", 1)[1] for ln in t.respuesta_final.splitlines() if ln.startswith("- Día ")]
    assert len(planes) == 7 and len(set(planes)) == 7  # ninguna actividad se repite entre días
    assert "(s)" not in t.respuesta_final and "2 personas" in t.respuesta_final


async def test_escenario_2_solo_clima_no_pasa_por_el_analista(tmp_path: Path) -> None:
    t = await correr(mock(tmp_path), "¿Qué clima hace en Bariloche en enero?")
    assert t.texto_ruta() == "supervisor → investigador → supervisor → sintesis"
    assert "analista" not in t.metricas["delegaciones"] and "22 °C" in t.respuesta_final


async def test_escenario_3_presupuesto_que_no_alcanza(tmp_path: Path) -> None:
    t = await correr(mock(tmp_path), DEMO.replace("800.000", "400.000"))
    assert t.texto_ruta() == RUTA_COMPLETA
    assert t.metricas["presupuesto_entra"] is False and t.metricas["alternativa_calculada"]
    assert "NO ENTRA" in t.respuesta_final and "## Alternativas" in t.respuesta_final


async def test_escenario_4_fechas_invalidas(tmp_path: Path) -> None:
    pregunta = "Quiero ir a Bariloche del 16 al 10 de enero, somos 2 personas, desde Mendoza, con 800.000 pesos. ¿Nos alcanza?"
    t = await correr(mock(tmp_path), pregunta)
    assert t.metricas["validacion"] == "rechazada" and t.metricas["presupuesto_entra"] is None
    assert "no se puede armar" in t.respuesta_final and "## Itinerario" not in t.respuesta_final


async def test_escenario_5_la_segunda_vez_sale_de_memoria(tmp_path: Path) -> None:
    c = mock(tmp_path)
    primera = await correr(c, DEMO)
    llamadas_tavily = len(c.buscador.llamadas)  # type: ignore[attr-defined]
    segunda = await correr(c, DEMO)
    assert primera.metricas["busquedas_web"] > 0 and primera.metricas["fuentes_memoria"] == 0
    assert segunda.metricas["busquedas_web"] == 0 and segunda.metricas["fuentes_web"] == 0
    assert segunda.metricas["fuentes_memoria"] > 0
    assert len(c.buscador.llamadas) == llamadas_tavily  # type: ignore[attr-defined]
    assert "[memoria] mem-" in segunda.respuesta_final and "TOTAL: $ 785.100" in segunda.respuesta_final


async def test_escenario_6_refinamiento(tmp_path: Path) -> None:
    t = await correr(mock(tmp_path), "¿Cómo llegar de Mendoza a Bariloche y cuánto sale el pasaje?")
    assert t.texto_ruta() == "supervisor → investigador → supervisor → investigador → supervisor → sintesis"
    assert t.metricas["refinamientos"] == 1
    assert any("no traen ningún precio" in o for o in t.metricas["observaciones"])
    rechazo = [e for e in t.eventos if e.nodo == "supervisor"][1].detalle
    assert "R3/R5 NO se cumplen" in rechazo["evaluacion"] and "forzar=true" in rechazo["instruccion"]
    assert "$140.000" in t.respuesta_final or "$ 140.000" in t.respuesta_final or "140.000" in t.respuesta_final


async def test_escenario_7_tavily_caido(tmp_path: Path) -> None:
    t = await correr(mock(tmp_path, fallar_web=True), DEMO)
    assert t.estado == "completo"
    assert any(o.startswith("Falló una búsqueda") for o in t.metricas["observaciones"])
    assert "## Lo que no pudimos averiguar" in t.respuesta_final
    assert t.metricas["presupuesto_entra"] is None  # no inventó un presupuesto sin precios


async def test_escenario_8_freno_de_pasos(tmp_path: Path) -> None:
    t = await correr(mock(tmp_path), DEMO, max_pasos=2)
    assert t.texto_ruta() == "supervisor → investigador → supervisor → sintesis"
    assert t.metricas["decision_forzada_por_freno"] and t.metricas["pasos_supervisor"] == 2
    assert t.respuesta_final.startswith("> ⚠️ Esta respuesta se armó con información parcial")


async def test_recursion_limit_corta_de_forma_controlada(tmp_path: Path) -> None:
    grafo = construir_grafo(mock(tmp_path))
    t = await consultar(grafo, DEMO, modo_mock=True, recursion_limit=3, guardar=False)
    assert t.estado == "cortado_por_recursion_limit" and "RECURSION_LIMIT" in t.respuesta_final


async def test_la_traza_se_guarda_en_json_y_log(tmp_path: Path) -> None:
    grafo = construir_grafo(mock(tmp_path))
    t = await consultar(grafo, DEMO, escenario="guardado", modo_mock=True)
    json_ = config.TRACES_DIR / Path(t.archivo).name
    assert json_.exists() and json_.with_suffix(".log").exists()
    assert "RUTA: " + RUTA_COMPLETA in json_.with_suffix(".log").read_text(encoding="utf-8")
