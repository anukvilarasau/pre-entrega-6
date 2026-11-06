"""Tests de la síntesis: fuentes armadas por código, avisos, itinerario y casos límite."""

from datetime import date
from typing import Any

import pytest
from langchain_core.messages import AIMessage

from agents.mock import politica_sintesis
from agents.synthesizer import AVISO_PRECIOS, crear_nodo_sintesis
from llm import LLMMock, guion
from state import Aporte, DecisionRegistrada, ErrorDeEstado, Fuente, PedidoViaje, estado_inicial, validar_actualizacion

PEDIDO = PedidoViaje(origen="Mendoza", destino="Bariloche", fecha_ida=date(2027, 1, 10),
                     fecha_vuelta=date(2027, 1, 16), dias=7, personas=2, presupuesto=800_000, moneda="ARS")
URL_MICRO = "https://micros.mock-viajes.example/mendoza-bariloche"
URL_CLIMA = "https://clima.mock-viajes.example/bariloche-enero"

INV = Aporte(
    agente="investigador", ronda=1, instruccion="Buscá todo",
    contenido=(f"## Hallazgos\n### Pasajes\n- Ida y vuelta en semicama: $140.000 por persona — [{URL_MICRO}] (web)\n"
               "### Clima\n- Máximas de 22 °C y mínimas de 7 °C — [mem-0123456789ab] (memoria)\n"
               "### Actividades\n- Circuito Chico, Refugio Frey, lago Gutiérrez — [mem-0123456789ab] (memoria)\n"
               "## Datos que faltan\n- ninguno"),
    fuentes=[Fuente(ref=URL_MICRO, origen="web", tema="pasajes", fecha_busqueda=date(2026, 11, 5)),
             Fuente(ref="mem-0123456789ab", origen="memoria", tema="clima", url=URL_CLIMA, fecha_busqueda=date(2026, 11, 3)),
             Fuente(ref=URL_MICRO, origen="web", tema="otro")],  # misma URL, otro tema: se lista una vez
    herramientas=["buscar_en_memoria", "buscar_en_web"],
)


def ana(contenido: str) -> Aporte:
    return Aporte(agente="analista", ronda=2, instruccion="Calculá", contenido=contenido,
                  herramientas=["validar_viaje", "calcular_presupuesto"])


ANA_ENTRA = ana("## Validación\nVÁLIDO: 7 días, 6 noches.\n## Presupuesto\nPRESUPUESTO — opción 'económica':\n"
                "TOTAL: $ 785.100 | PRESUPUESTO: $ 800.000\nRESULTADO: ENTRA en el presupuesto. Sobran $ 14.900.\n"
                "## Conclusión\nEntra.\n## Datos que faltan\n- ninguno")
ANA_NO_ENTRA = ana(ANA_ENTRA.contenido.replace("RESULTADO: ENTRA en el presupuesto. Sobran $ 14.900.",
                                                "RESULTADO: NO ENTRA en el presupuesto. Faltan $ 385.100."))
ANA_INVALIDO = ana("## Validación\nINVÁLIDO: el viaje no cumple las reglas.\n- viaje: la vuelta (2027-01-10) tiene que "
                   "ser DESPUÉS de la ida (2027-01-16)\n## Presupuesto\nNo se calculó.\n## Datos que faltan\n- fechas válidas")


def estado(aportes: list[Aporte], forzada: bool = False) -> Any:
    s: Any = estado_inicial("¿Me armás el viaje a Bariloche?")
    s.update(pedido=PEDIDO, aportes=aportes,
             decisiones=[DecisionRegistrada(paso=1, siguiente="FINALIZAR", forzada=forzada)])
    return s


async def sintetizar(aportes: list[Aporte], forzada: bool = False, llm: LLMMock | None = None) -> dict[str, Any]:
    return await crear_nodo_sintesis(llm or LLMMock(politica=politica_sintesis))(estado(aportes, forzada))


async def test_fuentes_y_aviso_los_agrega_el_codigo() -> None:
    llm = LLMMock(politica=guion([AIMessage(content="Respuesta del LLM.")]))
    salida = await sintetizar([INV, ANA_ENTRA], llm=llm)
    r = salida["respuesta_final"]
    assert r.startswith("Respuesta del LLM.")
    assert f"- [web] {URL_MICRO} — pasajes, buscado el 2026-11-05" in r
    assert f"- [memoria] mem-0123456789ab → {URL_CLIMA} — clima, buscado el 2026-11-03" in r
    assert r.count(URL_MICRO) == 1  # sin repetidos
    assert r.endswith(AVISO_PRECIOS)
    assert set(salida) == {"respuesta_final", "messages"} and salida["messages"][0].name == "sintesis"


async def test_flujo_completo_tiene_itinerario_dia_por_dia_y_presupuesto() -> None:
    r = (await sintetizar([INV, ANA_ENTRA]))["respuesta_final"]
    assert "## Itinerario" in r and "- Día 1 (10/01): viaje desde Mendoza" in r and "- Día 7 (16/01): regreso" in r
    assert "Clima esperado" in r
    assert "## Presupuesto" in r and "TOTAL: $ 785.100" in r and "ENTRA en el presupuesto" in r
    assert "## Alternativas" not in r


async def test_si_no_alcanza_propone_alternativas() -> None:
    r = (await sintetizar([INV, ANA_NO_ENTRA]))["respuesta_final"]
    assert "NO ENTRA" in r and "## Alternativas" in r


async def test_viaje_invalido_no_arma_itinerario() -> None:
    r = (await sintetizar([INV, ANA_INVALIDO]))["respuesta_final"]
    assert "no se puede armar" in r and "DESPUÉS de la ida" in r
    assert "## Itinerario" not in r


async def test_pregunta_sin_presupuesto_responde_directo() -> None:
    r = (await sintetizar([INV]))["respuesta_final"]
    assert "## Lo que encontramos" in r and "22 °C" in r and "## Itinerario" not in r


async def test_freno_agrega_aviso_de_informacion_parcial() -> None:
    r = (await sintetizar([INV], forzada=True))["respuesta_final"]
    assert r.startswith("> ⚠️ Esta respuesta se armó con información parcial")


async def test_sin_fuentes_no_promete_precios_de_la_web() -> None:
    r = (await sintetizar([]))["respuesta_final"]
    assert "No se usaron fuentes externas" in r and AVISO_PRECIOS not in r


async def test_sintesis_no_puede_escribir_aportes() -> None:
    with pytest.raises(ErrorDeEstado):
        validar_actualizacion("sintesis", {"aportes": [INV]})


def test_solo_se_listan_las_fuentes_citadas_y_sin_repetir_pagina() -> None:
    from agents.synthesizer import fuentes_a_listar

    web = Fuente(ref=URL_CLIMA, origen="web", tema="clima")
    mem = Fuente(ref="mem-0123456789ab", origen="memoria", tema="clima", url=URL_CLIMA)  # misma página
    otra = Fuente(ref="https://ruido.example/x", origen="web", tema="otro")
    elegidas = fuentes_a_listar([web, mem, otra], f"Hace 22 °C [{URL_CLIMA}]", [])
    assert elegidas == [web]
    assert fuentes_a_listar([web, otra], "sin citas", []) == [web, otra]  # sin citas: las primeras


# ------------------------------------------- actividades repetidas (corrida real 1)
ITINERARIO_REAL = """## Itinerario
- **Día 1 (sáb. 10/01/2027):** llegada desde Mendoza.
- **Día 2:** Cerro Catedral, excursión de 4 horas, ARS 26.000 [https://bariloche.org/x].
- **Día 3:** Circuito Chico y Cerro Catedral (día completo), desde ARS 29.400.
- **Día 4:** Camino de los 7 Lagos (día completo).
- **Día 7 (vie. 16/01/2027):** regreso a Mendoza."""


def test_detecta_la_repeticion_de_la_corrida_real() -> None:
    from agents.synthesizer import actividades_repetidas

    assert actividades_repetidas(ITINERARIO_REAL) == {"Cerro Catedral": [2, 3]}
    sin_repetir = ITINERARIO_REAL.replace("Circuito Chico y Cerro Catedral", "Circuito Chico")
    assert actividades_repetidas(sin_repetir) == {}  # Mendoza en los días 1 y 7 no cuenta (ida y vuelta)


async def test_si_repite_pide_una_correccion() -> None:
    corregido = ITINERARIO_REAL.replace("Circuito Chico y Cerro Catedral", "Circuito Chico")
    llm = LLMMock(politica=guion([AIMessage(content=ITINERARIO_REAL), AIMessage(content=corregido)]))
    r = (await sintetizar([INV, ANA_ENTRA], llm=llm))["respuesta_final"]
    assert "Circuito Chico (día completo)" in r and "Nota: el itinerario repite" not in r
    assert "se repite: 'Cerro Catedral' en los días 2, 3" in str(llm.vistos[1][-1].content)


async def test_si_sigue_repitiendo_avisa() -> None:
    llm = LLMMock(politica=guion([AIMessage(content=ITINERARIO_REAL), AIMessage(content=ITINERARIO_REAL)]))
    r = (await sintetizar([INV, ANA_ENTRA], llm=llm))["respuesta_final"]
    assert "Nota: el itinerario repite Cerro Catedral" in r
