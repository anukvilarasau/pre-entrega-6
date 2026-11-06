"""
analyst_tools.py — Las 2 herramientas del agente ANALISTA (sin LLM adentro).

A. validar_viaje: valida con Pydantic el pedido/plan de viaje. Reglas:
   origen y destino distintos, vuelta DESPUÉS de la ida, viaje en el futuro,
   duración razonable, días que coincidan con las fechas, personas entre 1
   y 20, presupuesto positivo y moneda ARS o USD. Junta TODOS los errores
   (no solo el primero) para que el usuario sepa todo lo que tiene que corregir.

B. calcular_presupuesto: hace la cuenta del viaje, renglón por renglón:
     pasajes      = precio por persona × personas (× 2 si el precio es por tramo)
     alojamiento  = por habitación: precio × noches × habitaciones
                    por persona:    precio × noches × personas
                    (nunca las dos: la unidad es explícita y obligatoria)
     comida       = precio por persona por día × días × personas
     actividades  = precio por persona por día × días × personas
     extras       = precio × personas (o una sola vez)
   Si algún precio está en otra moneda, lo convierte e INFORMA qué dólar usó
   y de qué fuente salió:
     - pagado con tarjeta (crédito o débito) → dólar TARJETA;
     - pagado en efectivo/transferencia     → dólar MEP (si no está, oficial).
   Devuelve el total, si entra en el presupuesto, cuánto sobra o falta, el
   costo por día y por persona por día.

Las dos son determinísticas: el mismo input da siempre el mismo resultado.
Igual que las del investigador, devuelven texto (para el LLM) + artifact.
"""

import math
import re
from datetime import date
from typing import Literal

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

import config
from state import PATRON_ID_MEMORIA, Fuente, Moneda, cantidad

MAX_DIAS_VIAJE: int = 60
MedioPago = Literal["tarjeta", "efectivo"]
TipoDolar = Literal["oficial", "tarjeta", "MEP"]


# ===========================================================================
# A. validar_viaje
# ===========================================================================
class ViajeValidado(BaseModel):
    """Un viaje que cumple TODAS las reglas. Si no las cumple, Pydantic lanza ValidationError."""

    model_config = ConfigDict(extra="forbid")

    origen: str = Field(min_length=2, max_length=80)
    destino: str = Field(min_length=2, max_length=80)
    fecha_ida: date
    fecha_vuelta: date
    personas: int = Field(ge=1, le=20, description="Entre 1 y 20 personas.")
    presupuesto: float = Field(gt=0, description="Tiene que ser positivo.")
    moneda: Moneda
    dias_pedidos: int | None = Field(default=None, ge=1, le=MAX_DIAS_VIAJE)

    @field_validator("fecha_ida")
    @classmethod
    def _ida_en_el_futuro(cls, v: date) -> date:
        if v < config.FECHA_REFERENCIA:
            raise ValueError(f"la fecha de ida ({v}) ya pasó (hoy es {config.FECHA_REFERENCIA})")
        return v

    @model_validator(mode="after")
    def _reglas_entre_campos(self) -> "ViajeValidado":
        errores: list[str] = []
        if self.origen.strip().lower() == self.destino.strip().lower():
            errores.append("el origen y el destino son el mismo lugar")
        noches = (self.fecha_vuelta - self.fecha_ida).days
        if noches <= 0:
            errores.append(f"la vuelta ({self.fecha_vuelta}) tiene que ser DESPUÉS de la ida ({self.fecha_ida})")
        elif noches + 1 > MAX_DIAS_VIAJE:
            errores.append(f"el viaje dura {noches + 1} días: el máximo es {MAX_DIAS_VIAJE}")
        elif self.dias_pedidos is not None and noches + 1 != self.dias_pedidos:
            errores.append(
                f"las fechas dan {noches + 1} días ({noches} noches) pero se pidieron {self.dias_pedidos} días"
            )
        if errores:
            raise ValueError("; ".join(errores))
        return self

    @property
    def noches(self) -> int:
        return (self.fecha_vuelta - self.fecha_ida).days

    @property
    def dias(self) -> int:
        return self.noches + 1


class ValidarViajeInput(BaseModel):
    """Lo que recibe la herramienta: tipos básicos (las REGLAS se aplican adentro)."""

    model_config = ConfigDict(extra="forbid")

    origen: str = Field(description="Ciudad de salida.")
    destino: str = Field(description="Ciudad de destino.")
    fecha_ida: date = Field(description="Fecha de ida, formato AAAA-MM-DD.")
    fecha_vuelta: date = Field(description="Fecha de vuelta, formato AAAA-MM-DD.")
    personas: int = Field(description="Cantidad de personas.")
    presupuesto: float = Field(description="Presupuesto TOTAL del viaje.")
    moneda: str = Field(description="Moneda del presupuesto: 'ARS' o 'USD'.")
    dias_pedidos: int | None = Field(default=None, description="Días que pidió el usuario (si los dijo).")


def _errores_legibles(e: ValidationError) -> list[str]:
    salida: list[str] = []
    for err in e.errors():
        campo = ".".join(str(x) for x in err["loc"]) or "viaje"
        mensaje = str(err["msg"]).removeprefix("Value error, ")
        salida.append(f"{campo}: {mensaje}")
    return salida


def validar_viaje(**datos: object) -> tuple[str, dict[str, object]]:
    """Aplica ViajeValidado. Devuelve (texto, artifact con el resultado)."""
    try:
        v = ViajeValidado.model_validate(datos)
    except ValidationError as e:
        errores = _errores_legibles(e)
        return (
            "INVÁLIDO: el viaje no cumple las reglas.\n" + "\n".join(f"- {x}" for x in errores)
            + "\nNo calcules el presupuesto hasta que se corrijan estos datos."
        ), {"valido": False, "errores": errores}
    texto = (
        f"VÁLIDO: {v.origen} → {v.destino}, del {v.fecha_ida} al {v.fecha_vuelta} "
        f"({cantidad(v.dias, 'día', 'días')}, {cantidad(v.noches, 'noche')}), {cantidad(v.personas, 'persona')}, "
        f"presupuesto {_fmt(v.presupuesto, v.moneda)}."
    )
    return texto, {"valido": True, "dias": v.dias, "noches": v.noches, "personas": v.personas}


# ===========================================================================
# B. calcular_presupuesto
# ===========================================================================
class Monto(BaseModel):
    model_config = ConfigDict(extra="forbid")

    monto: float = Field(ge=0, description="Importe (sin símbolos, ej. 35000).")
    moneda: Moneda = Field(description="'ARS' o 'USD'.")
    medio_pago: MedioPago = Field(
        default="tarjeta",
        description="'tarjeta' (crédito o débito) o 'efectivo' (efectivo o transferencia). Define qué dólar se usa.",
    )
    fuente: str | None = Field(default=None, description="URL o ID de memoria (mem-...) de donde salió el precio.")


class Pasaje(Monto):
    tramo: Literal["ida_y_vuelta", "por_tramo"] = Field(
        description="'ida_y_vuelta' si el precio ya incluye los 2 viajes; 'por_tramo' si es solo ida (se multiplica × 2)."
    )
    medio: str = Field(default="", description="Ej. 'micro semicama' o 'avión'.")


class Alojamiento(Monto):
    unidad: Literal["por_noche_por_habitacion", "por_noche_por_persona"] = Field(
        description="OBLIGATORIO: si el precio es por noche POR HABITACIÓN o por noche POR PERSONA."
    )
    personas_por_habitacion: int = Field(default=2, ge=1, le=10, description="Solo para precios por habitación.")
    tipo: str = Field(default="", description="Ej. 'hostel habitación doble'.")


class GastoExtra(Monto):
    descripcion: str = Field(min_length=2, description="Ej. 'excursión Isla Victoria'.")
    por_persona: bool = Field(default=True, description="True: se paga por cada persona. False: una sola vez.")


class TipoCambio(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tipo: TipoDolar = Field(description="'oficial', 'tarjeta' o 'MEP'.")
    valor_ars: float = Field(gt=0, description="Pesos por cada dólar (ej. 1885).")
    fuente: str = Field(min_length=3, description="URL o ID de memoria (mem-...) de donde salió la cotización.")


class CalcularPresupuestoInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    opcion: str = Field(default="principal", description="Nombre de la opción (ej. 'económica', 'con 5 días').")
    personas: int = Field(ge=1, le=20)
    noches: int = Field(ge=1, le=MAX_DIAS_VIAJE)
    dias: int | None = Field(default=None, ge=1, le=MAX_DIAS_VIAJE + 1, description="Por defecto noches + 1.")
    presupuesto: float = Field(gt=0, description="Presupuesto TOTAL del viaje.")
    moneda_presupuesto: Moneda
    pasaje: Pasaje = Field(description="Precio del pasaje POR PERSONA.")
    alojamiento: Alojamiento
    comida_por_persona_por_dia: Monto
    actividades_por_persona_por_dia: Monto | None = None
    extras: list[GastoExtra] = Field(default_factory=list)
    tipos_de_cambio: list[TipoCambio] = Field(default_factory=list)


class Renglon(BaseModel):
    concepto: str
    formula: str
    subtotal: float


class ResultadoPresupuesto(BaseModel):
    opcion: str
    moneda: Moneda
    renglones: list[Renglon]
    total: float
    presupuesto: float
    entra: bool
    diferencia: float = Field(description="Positiva: sobra. Negativa: falta.")
    costo_por_dia: float
    costo_por_persona_por_dia: float
    conversiones: list[str]
    dias: int
    noches: int
    personas: int


class ErrorDeCambio(ValueError):
    """Falta una cotización para convertir un monto."""


def _fmt(valor: float, moneda: str) -> str:
    """$ 1.234.567 (ARS) o US$ 1.234,50 (USD), a la argentina."""
    if moneda == "USD":
        entero, dec = f"{valor:,.2f}".split(".")
        return f"US$ {entero.replace(',', '.')},{dec}"
    return f"$ {valor:,.0f}".replace(",", ".")


def elegir_dolar(medio_pago: MedioPago, tipos: dict[str, TipoCambio]) -> TipoCambio:
    """Regla del dólar: tarjeta → TARJETA; efectivo → MEP (o, si no está, OFICIAL)."""
    preferencias: list[str] = ["tarjeta"] if medio_pago == "tarjeta" else ["MEP", "oficial"]
    for tipo in preferencias:
        if tipo in tipos:
            return tipos[tipo]
    raise ErrorDeCambio(
        f"falta la cotización del dólar {' o '.join(preferencias)} (pago con {medio_pago}) para convertir"
    )


def convertir(m: Monto, destino: Moneda, tipos: dict[str, TipoCambio], conversiones: list[str], concepto: str) -> float:
    """Pasa un monto a la moneda del presupuesto y anota qué dólar usó y de dónde salió."""
    if m.moneda == destino:
        return m.monto
    tc = elegir_dolar(m.medio_pago, tipos)
    convertido = m.monto * tc.valor_ars if destino == "ARS" else m.monto / tc.valor_ars
    cotizacion = f"{tc.valor_ars:,.0f}".replace(",", ".")
    nota = (f"{concepto}: {_fmt(m.monto, m.moneda)} → {_fmt(convertido, destino)} con dólar {tc.tipo} "
            f"${cotizacion} (pago con {m.medio_pago}; fuente: {tc.fuente})")
    if nota not in conversiones:
        conversiones.append(nota)
    return convertido


def calcular(p: CalcularPresupuestoInput) -> ResultadoPresupuesto:
    """La cuenta del viaje. Lanza ErrorDeCambio si falta una cotización necesaria."""
    dias = p.dias or p.noches + 1
    mon = p.moneda_presupuesto
    tipos = {t.tipo: t for t in p.tipos_de_cambio}
    conv: list[str] = []
    renglones: list[Renglon] = []

    # Pasajes (siempre por persona)
    tramos = 1 if p.pasaje.tramo == "ida_y_vuelta" else 2
    pasaje_u = convertir(p.pasaje, mon, tipos, conv, "pasaje")
    renglones.append(Renglon(
        concepto=f"Pasajes {p.pasaje.medio}".strip(),
        formula=(f"{_fmt(pasaje_u, mon)} por persona{' por tramo × 2 tramos' if tramos == 2 else ' (ida y vuelta)'} "
                 f"× {cantidad(p.personas, 'persona')}"),
        subtotal=pasaje_u * tramos * p.personas,
    ))

    # Alojamiento (la unidad decide la fórmula: nunca se multiplica dos veces)
    aloj_u = convertir(p.alojamiento, mon, tipos, conv, "alojamiento")
    if p.alojamiento.unidad == "por_noche_por_habitacion":
        habitaciones = math.ceil(p.personas / p.alojamiento.personas_por_habitacion)
        formula = (f"{_fmt(aloj_u, mon)} por noche por habitación × {cantidad(p.noches, 'noche')} × "
                   f"{cantidad(habitaciones, 'habitación', 'habitaciones')} "
                   f"(de {cantidad(p.alojamiento.personas_por_habitacion, 'persona')})")
        subtotal = aloj_u * p.noches * habitaciones
    else:
        formula = (f"{_fmt(aloj_u, mon)} por persona por noche × {cantidad(p.noches, 'noche')} × "
                   f"{cantidad(p.personas, 'persona')}")
        subtotal = aloj_u * p.noches * p.personas
    renglones.append(Renglon(concepto=f"Alojamiento {p.alojamiento.tipo}".strip(), formula=formula, subtotal=subtotal))

    # Gastos diarios por persona
    diarios = [("Comida", p.comida_por_persona_por_dia)]
    if p.actividades_por_persona_por_dia is not None:
        diarios.append(("Actividades", p.actividades_por_persona_por_dia))
    for concepto, monto in diarios:
        u = convertir(monto, mon, tipos, conv, concepto.lower())
        renglones.append(Renglon(
            concepto=concepto,
            formula=(f"{_fmt(u, mon)} por persona por día × {cantidad(dias, 'día', 'días')} × "
                     f"{cantidad(p.personas, 'persona')}"),
            subtotal=u * dias * p.personas,
        ))

    # Extras
    for x in p.extras:
        u = convertir(x, mon, tipos, conv, x.descripcion)
        veces = p.personas if x.por_persona else 1
        renglones.append(Renglon(
            concepto=x.descripcion,
            formula=f"{_fmt(u, mon)} × {cantidad(veces, 'persona')}" if x.por_persona else f"{_fmt(u, mon)} (una vez)",
            subtotal=u * veces,
        ))

    decimales = 2 if mon == "USD" else 0
    for r in renglones:
        r.subtotal = round(r.subtotal, decimales)
    total = round(sum(r.subtotal for r in renglones), decimales)
    return ResultadoPresupuesto(
        opcion=p.opcion, moneda=mon, renglones=renglones, total=total, presupuesto=p.presupuesto,
        entra=total <= p.presupuesto, diferencia=round(p.presupuesto - total, decimales),
        costo_por_dia=round(total / dias, decimales),
        costo_por_persona_por_dia=round(total / dias / p.personas, decimales),
        conversiones=conv, dias=dias, noches=p.noches, personas=p.personas,
    )


def _texto_presupuesto(r: ResultadoPresupuesto) -> str:
    m = r.moneda
    lineas = [f"PRESUPUESTO — opción '{r.opcion}' ({cantidad(r.dias, 'día', 'días')}, {cantidad(r.noches, 'noche')}, "
              f"{cantidad(r.personas, 'persona')}):"]
    lineas += [f"- {x.concepto}: {_fmt(x.subtotal, m)}  [{x.formula}]" for x in r.renglones]
    lineas.append(f"TOTAL: {_fmt(r.total, m)} | PRESUPUESTO: {_fmt(r.presupuesto, m)}")
    if r.entra:
        lineas.append(f"RESULTADO: ENTRA en el presupuesto. Sobran {_fmt(r.diferencia, m)}.")
    else:
        lineas.append(f"RESULTADO: NO ENTRA en el presupuesto. Faltan {_fmt(-r.diferencia, m)}.")
    lineas.append(f"Costo por día: {_fmt(r.costo_por_dia, m)} | por persona por día: {_fmt(r.costo_por_persona_por_dia, m)}")
    if r.conversiones:
        lineas.append("Conversiones de moneda:")
        lineas += [f"- {c}" for c in r.conversiones]
    else:
        lineas.append("Conversiones de moneda: ninguna (todo en la misma moneda).")
    return "\n".join(lineas)


def _fuente_de(ref: str | None, tema: str) -> Fuente | None:
    """Convierte la referencia de un precio o cotización en una Fuente (si tiene formato válido)."""
    if not ref:
        return None
    # El LLM a veces escribe "https://... (mem-...)" o "[mem-...]": se extrae la referencia limpia.
    url = re.search(r"https?://[^\s\])>,]+", ref)
    mem = re.search(r"mem-[0-9a-f]{12}", ref)
    ref = url.group(0) if url else (mem.group(0) if mem else ref.strip().strip("[]"))
    try:
        if PATRON_ID_MEMORIA.match(ref):
            return Fuente(ref=ref, origen="memoria", tema=tema)  # type: ignore[arg-type]
        return Fuente(ref=ref, origen="web", tema=tema, url=ref)  # type: ignore[arg-type]
    except ValidationError:
        return None


def calcular_presupuesto(**datos: object) -> tuple[str, list[Fuente]]:
    """Herramienta: valida la entrada, calcula y devuelve (texto, fuentes usadas)."""
    p = CalcularPresupuestoInput.model_validate(datos)
    try:
        r = calcular(p)
    except ErrorDeCambio as e:
        return (f"ERROR: {e}. Pedile al investigador la cotización que falta o usá precios en la misma "
                "moneda del presupuesto."), []
    refs = [(p.pasaje.fuente, "pasajes"), (p.alojamiento.fuente, "alojamiento"),
            (p.comida_por_persona_por_dia.fuente, "comida")]
    if p.actividades_por_persona_por_dia:
        refs.append((p.actividades_por_persona_por_dia.fuente, "actividades"))
    refs += [(x.fuente, "actividades") for x in p.extras]
    refs += [(t.fuente, "cambio") for t in p.tipos_de_cambio]
    fuentes = [f for ref, tema in refs if (f := _fuente_de(ref, tema)) is not None]
    return _texto_presupuesto(r), fuentes


# ===========================================================================
# Fábrica
# ===========================================================================
def crear_herramientas_analista() -> list[BaseTool]:
    """Las 2 herramientas del analista (no dependen de ningún servicio externo)."""
    return [
        StructuredTool.from_function(
            func=validar_viaje,
            name="validar_viaje",
            args_schema=ValidarViajeInput,
            response_format="content_and_artifact",
            description=(
                "Valida con reglas estrictas los datos del viaje: origen y destino distintos, vuelta DESPUÉS de la "
                "ida, ida en el futuro, días que coincidan con las fechas, 1 a 20 personas, presupuesto positivo, "
                "moneda ARS o USD. Usala SIEMPRE PRIMERO. Responde 'VÁLIDO: ...' (con días y noches) o "
                "'INVÁLIDO: ...' con TODOS los errores; si es inválido, no calcules el presupuesto."
            ),
        ),
        StructuredTool.from_function(
            func=calcular_presupuesto,
            name="calcular_presupuesto",
            args_schema=CalcularPresupuestoInput,
            response_format="content_and_artifact",
            description=(
                "Calcula el costo total del viaje con los precios que encontró el investigador: pasajes por persona "
                "(ida y vuelta o por tramo) + alojamiento (indicá SIEMPRE si es por noche por habitación o por "
                "persona) + comida y actividades por persona por día + extras. Si hay precios en otra moneda, "
                "pasá los tipos de cambio con su fuente: con tarjeta se usa el dólar tarjeta; en efectivo, el MEP. "
                "Dice si entra en el presupuesto, cuánto sobra o falta y el costo por día. Podés llamarla varias "
                "veces con opciones distintas (ej. 'económica' y 'con menos días')."
            ),
        ),
    ]
