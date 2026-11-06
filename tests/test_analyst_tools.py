"""Tests de agents/analyst_tools.py: validar_viaje y calcular_presupuesto (determinísticas, sin LLM)."""

from datetime import date
from typing import Any

import pytest
from langchain_core.messages import ToolMessage
from pydantic import ValidationError

import config
from agents.analyst_tools import crear_herramientas_analista

URL_CAMBIO = "https://cotizaciones.mock-viajes.example/dolar-hoy"
VIAJE = {"origen": "Mendoza", "destino": "Bariloche", "fecha_ida": "2027-01-10", "fecha_vuelta": "2027-01-16",
         "personas": 2, "presupuesto": 800_000, "moneda": "ARS", "dias_pedidos": 7}
CAMBIOS = [{"tipo": "oficial", "valor_ars": 1450, "fuente": URL_CAMBIO},
           {"tipo": "tarjeta", "valor_ars": 1885, "fuente": URL_CAMBIO},
           {"tipo": "MEP", "valor_ars": 1500, "fuente": "mem-0123456789ab"}]


def base(**cambios: Any) -> dict[str, Any]:
    """El presupuesto económico del mock: 2 personas, 6 noches, 7 días."""
    datos: dict[str, Any] = {
        "opcion": "económica", "personas": 2, "noches": 6, "presupuesto": 800_000, "moneda_presupuesto": "ARS",
        "pasaje": {"monto": 140_000, "moneda": "ARS", "tramo": "ida_y_vuelta", "medio": "micro",
                   "fuente": "https://micros.mock-viajes.example/mendoza-bariloche"},
        "alojamiento": {"monto": 35_000, "moneda": "ARS", "unidad": "por_noche_por_habitacion", "personas_por_habitacion": 2},
        "comida_por_persona_por_dia": {"monto": 9_000, "moneda": "ARS"},
        "actividades_por_persona_por_dia": {"monto": 4_000, "moneda": "ARS"},
        "extras": [{"descripcion": "excursión Isla Victoria", "monto": 30, "moneda": "USD", "medio_pago": "tarjeta"}],
        "tipos_de_cambio": CAMBIOS,
    }
    datos.update(cambios)
    return datos


@pytest.fixture(autouse=True)
def hoy_fijo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "FECHA_REFERENCIA", date(2026, 11, 5))


async def usar(nombre: str, args: dict[str, Any]) -> ToolMessage:
    herramienta = {t.name: t for t in crear_herramientas_analista()}[nombre]
    return await herramienta.ainvoke({"type": "tool_call", "id": "t1", "name": nombre, "args": args})


# ---------------------------------------------------------------- validar_viaje
async def test_viaje_valido_informa_dias_y_noches() -> None:
    msg = await usar("validar_viaje", VIAJE)
    assert str(msg.content).startswith("VÁLIDO") and "7 días, 6 noches" in str(msg.content)
    assert msg.artifact == {"valido": True, "dias": 7, "noches": 6, "personas": 2}


async def test_vuelta_antes_que_la_ida_es_invalido() -> None:
    """Escenario 4."""
    msg = await usar("validar_viaje", {**VIAJE, "fecha_ida": "2027-01-16", "fecha_vuelta": "2027-01-10"})
    texto = str(msg.content)
    assert texto.startswith("INVÁLIDO") and "tiene que ser DESPUÉS de la ida" in texto
    assert "No calcules el presupuesto" in texto and msg.artifact["valido"] is False


@pytest.mark.parametrize(("cambios", "esperado"), [
    ({"fecha_ida": "2026-10-01", "fecha_vuelta": "2026-10-07"}, "ya pasó"),
    ({"fecha_vuelta": "2027-01-20"}, "las fechas dan 11 días"),
    ({"destino": "mendoza"}, "el mismo lugar"),
    ({"fecha_vuelta": "2027-04-10"}, "el máximo es 60"),
])
async def test_reglas_de_negocio(cambios: dict[str, Any], esperado: str) -> None:
    msg = await usar("validar_viaje", {**VIAJE, **cambios})
    assert str(msg.content).startswith("INVÁLIDO") and esperado in str(msg.content)


async def test_junta_todos_los_errores_juntos() -> None:
    msg = await usar("validar_viaje", {**VIAJE, "personas": 0, "presupuesto": -5, "moneda": "EUR"})
    errores = msg.artifact["errores"]
    assert len(errores) == 3
    assert {e.split(":")[0] for e in errores} == {"personas", "presupuesto", "moneda"}


# ---------------------------------------------------------- calcular_presupuesto
async def test_presupuesto_economico_entra_y_usa_dolar_tarjeta() -> None:
    """Escenario 1 (números del mock): entra justo, con la excursión convertida al dólar tarjeta."""
    msg = await usar("calcular_presupuesto", base())
    t = str(msg.content)
    assert "- Pasajes micro: $ 280.000" in t                # 140.000 × 2 personas (ida y vuelta)
    assert "- Alojamiento: $ 210.000" in t                  # 35.000 × 6 noches × 1 habitación doble
    assert "- Comida: $ 126.000" in t                       # 9.000 × 7 días × 2 personas
    assert "- Actividades: $ 56.000" in t                   # 4.000 × 7 × 2
    assert "- excursión Isla Victoria: $ 113.100" in t      # US$ 30 × 1.885 × 2
    assert "TOTAL: $ 785.100" in t and "ENTRA en el presupuesto. Sobran $ 14.900" in t
    assert "Costo por día: $ 112.157 | por persona por día: $ 56.079" in t
    assert "US$ 30,00 → $ 56.550 con dólar tarjeta $1.885 (pago con tarjeta; fuente: " + URL_CAMBIO in t
    assert {f.tema for f in msg.artifact} == {"pasajes", "cambio"}


async def test_presupuesto_que_no_alcanza() -> None:
    """Escenario 3: mismo viaje con $400.000."""
    t = str((await usar("calcular_presupuesto", base(presupuesto=400_000))).content)
    assert "NO ENTRA en el presupuesto. Faltan $ 385.100" in t


async def test_alojamiento_por_persona_no_se_duplica_ni_se_divide() -> None:
    por_hab = str((await usar("calcular_presupuesto", base())).content)
    por_persona = str((await usar("calcular_presupuesto", base(
        alojamiento={"monto": 35_000, "moneda": "ARS", "unidad": "por_noche_por_persona"}))).content)
    assert "- Alojamiento: $ 210.000" in por_hab and "× 1 habitación (de 2 personas)" in por_hab
    assert "- Alojamiento: $ 420.000" in por_persona and "× 6 noches × 2 personas" in por_persona


async def test_habitaciones_se_redondean_para_arriba() -> None:
    t = str((await usar("calcular_presupuesto", base(personas=3))).content)
    assert "× 2 habitaciones" in t and "- Alojamiento: $ 420.000" in t  # 35.000 × 6 × 2


async def test_pasaje_por_tramo_se_multiplica_por_dos() -> None:
    t = str((await usar("calcular_presupuesto", base(pasaje={"monto": 70_000, "moneda": "ARS", "tramo": "por_tramo"}))).content)
    assert "- Pasajes: $ 280.000" in t and "por tramo × 2 tramos" in t


async def test_efectivo_usa_mep_y_si_no_hay_mep_usa_oficial() -> None:
    extra_efectivo = [{"descripcion": "kayak", "monto": 10, "moneda": "USD", "medio_pago": "efectivo"}]
    con_mep = str((await usar("calcular_presupuesto", base(extras=extra_efectivo))).content)
    assert "- kayak: $ 30.000" in con_mep and "dólar MEP $1.500" in con_mep and "mem-0123456789ab" in con_mep
    sin_mep = str((await usar("calcular_presupuesto", base(extras=extra_efectivo, tipos_de_cambio=CAMBIOS[:2]))).content)
    assert "- kayak: $ 29.000" in sin_mep and "dólar oficial $1.450" in sin_mep


async def test_falta_dolar_tarjeta_devuelve_error_claro() -> None:
    msg = await usar("calcular_presupuesto", base(tipos_de_cambio=[CAMBIOS[2]]))  # solo MEP
    assert str(msg.content).startswith("ERROR: falta la cotización del dólar tarjeta")


async def test_presupuesto_en_dolares_convierte_al_reves() -> None:
    t = str((await usar("calcular_presupuesto", base(presupuesto=500, moneda_presupuesto="USD", extras=[]))).content)
    # 280.000 ARS pagados con tarjeta / 1.885 = US$ 148,54
    assert "- Pasajes micro: US$ 148,54" in t and "dólar tarjeta" in t


async def test_sin_conversiones_lo_dice() -> None:
    t = str((await usar("calcular_presupuesto", base(extras=[]))).content)
    assert "Conversiones de moneda: ninguna" in t


async def test_unidad_de_alojamiento_es_obligatoria() -> None:
    herramienta = {t.name: t for t in crear_herramientas_analista()}["calcular_presupuesto"]
    with pytest.raises(ValidationError, match="unidad"):
        await herramienta.ainvoke(base(alojamiento={"monto": 35_000, "moneda": "ARS"}))


async def test_es_deterministica() -> None:
    a = (await usar("calcular_presupuesto", base())).content
    b = (await usar("calcular_presupuesto", base())).content
    assert a == b


def test_fuente_mal_escrita_por_el_llm_se_limpia() -> None:
    from agents.analyst_tools import _fuente_de

    f = _fuente_de("https://www.google.com/travel/flights?gl=AR (mem-9594a81499af)", "pasajes")
    assert f is not None and f.ref == "https://www.google.com/travel/flights?gl=AR" and f.origen == "web"
    g = _fuente_de("[mem-9594a81499af]", "pasajes")
    assert g is not None and g.ref == "mem-9594a81499af" and g.origen == "memoria"
    assert _fuente_de("una revista", "pasajes") is None
