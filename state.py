"""
state.py — El estado compartido del orquestador: la "ficha" que viaja por el grafo.

Tres ideas:

1. `OrquestadorState` HEREDA de `MessagesState` (de LangGraph), como pide la
   consigna. De ahí trae `messages` (el historial, con el reductor
   `add_messages`) y le suma los campos propios del multi-agente.

2. Trazabilidad: cada especialista deja su trabajo como un `Aporte` (modelo
   Pydantic) en la lista `aportes`. Cada Aporte dice QUIÉN lo hizo, con QUÉ
   instrucción, QUÉ encontró, de QUÉ fuentes y con QUÉ herramientas. Así
   nunca se pierde de dónde salió cada dato.

3. Cada nodo escribe SOLO su parte. `validar_actualizacion()` revisa lo que
   devuelve cada nodo antes de que entre al estado: si el investigador
   intentara tocar la decisión del supervisor, o firmar un aporte como
   "analista", falla con un error claro.
"""

import operator
import re
from datetime import date, datetime
from typing import Annotated, Any, Literal

from langgraph.graph import MessagesState
from pydantic import (
    AnyHttpUrl,
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)

# ---------------------------------------------------------------------------
# Nombres (tipos Literal: el código no acepta ningún otro valor)
# ---------------------------------------------------------------------------
Especialista = Literal["investigador", "analista"]
Nodo = Literal["supervisor", "investigador", "analista", "sintesis"]
# Lo que puede decidir el supervisor (salida estructurada).
Destino = Literal["investigador", "analista", "FINALIZAR"]

# Herramientas que puede usar cada especialista: "herramientas acotadas".
HERRAMIENTAS_POR_AGENTE: dict[str, frozenset[str]] = {
    "investigador": frozenset({"buscar_en_memoria", "buscar_en_web"}),
    "analista": frozenset({"validar_viaje", "calcular_presupuesto"}),
}

# De dónde salió un dato y sobre qué tema es.
Origen = Literal["memoria", "web"]
Tema = Literal["pasajes", "alojamiento", "comida", "clima", "actividades", "cambio", "otro"]

# Un precio: "$480.000", "US$ 120", "USD 95", "ARS 60.000"...
PATRON_PRECIO: re.Pattern[str] = re.compile(r"(\$|US\$|USD|U\$S|ARS)\s?\d", re.IGNORECASE)

# IDs de la memoria ChromaDB: "mem-" + 12 caracteres hexadecimales.
PATRON_ID_MEMORIA: re.Pattern[str] = re.compile(r"^mem-[0-9a-f]{12}$")
_VALIDADOR_URL: TypeAdapter[AnyHttpUrl] = TypeAdapter(AnyHttpUrl)


def cantidad(n: int | float, singular: str, plural: str | None = None) -> str:
    """'1 persona', '2 personas', '1 habitación', '3 habitaciones' (plural bien escrito)."""
    return f"{n} {singular if n == 1 else (plural or singular + 's')}"


def ahora() -> str:
    """Timestamp ISO con milisegundos y zona horaria local."""
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


# ---------------------------------------------------------------------------
# Modelos Pydantic
# ---------------------------------------------------------------------------
class Fuente(BaseModel):
    """De dónde salió un dato: una URL de la web o un registro de la memoria."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ref: str = Field(min_length=1, description="URL http(s) si origen='web'; ID 'mem-…' si origen='memoria'.")
    origen: Origen
    tema: Tema = "otro"
    url: str | None = Field(default=None, description="URL original (en memoria: la de cuando se buscó).")
    fecha_busqueda: date = Field(default_factory=date.today)

    @model_validator(mode="after")
    def _ref_segun_origen(self) -> "Fuente":
        if self.origen == "memoria":
            if not PATRON_ID_MEMORIA.match(self.ref):
                raise ValueError(f"ID de memoria inválido: {self.ref!r} (formato 'mem-' + 12 hex)")
        else:
            try:
                _VALIDADOR_URL.validate_python(self.ref)
            except ValidationError as e:
                raise ValueError(f"URL inválida: {self.ref!r}") from e
        if self.url is not None:
            try:
                _VALIDADOR_URL.validate_python(self.url)
            except ValidationError as e:
                raise ValueError(f"URL original inválida: {self.url!r}") from e
        return self


class LlamadaHerramienta(BaseModel):
    """Una llamada a herramienta dentro del ciclo ReAct de un especialista."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    herramienta: str = Field(min_length=1)
    argumentos: dict[str, Any] = Field(default_factory=dict)
    resultado: str = Field(default="", description="Resultado resumido (primeros caracteres).")
    timestamp: str = Field(default_factory=ahora)


class CoberturaTema(BaseModel):
    """Qué se consiguió de UN tema en una investigación (lo calcula el CÓDIGO, no el LLM)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tema: Tema
    fuentes: int = Field(ge=0, description="Fuentes distintas de este tema.")
    memoria: int = Field(ge=0, description="Cuántas salieron de la memoria.")
    web: int = Field(ge=0, description="Cuántas salieron de la web.")
    con_precios: bool = Field(description="Si algún resultado de este tema trae un monto ($, US$, ARS...).")


class Aporte(BaseModel):
    """Lo que aportó UN especialista en UNA intervención."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    agente: Especialista
    ronda: int = Field(ge=1, description="Número de decisión del supervisor que pidió este trabajo.")
    instruccion: str = Field(min_length=1, description="Lo que le pidió el supervisor.")
    contenido: str = Field(min_length=1, description="El resultado del especialista.")
    fuentes: list[Fuente] = Field(default_factory=list, description="De dónde salió cada dato.")
    herramientas: list[str] = Field(default_factory=list, description="Herramientas ejecutadas.")
    llamadas: list[LlamadaHerramienta] = Field(
        default_factory=list, description="Cada llamada a herramienta, en orden (para la traza)."
    )
    cobertura: list[CoberturaTema] = Field(
        default_factory=list, description="(investigador) Resumen por tema armado por código."
    )
    observaciones: list[str] = Field(
        default_factory=list,
        description="Problemas que detectó el chequeo por código (vacío = sin problemas).",
    )
    timestamp: str = Field(default_factory=ahora)

    @field_validator("fuentes")
    @classmethod
    def _fuentes_sin_repetidos(cls, fuentes: list[Fuente]) -> list[Fuente]:
        # Repetida = misma referencia Y mismo tema (una página puede servir para 2 temas).
        vistas: dict[tuple[str, str], Fuente] = {}
        for f in fuentes:
            vistas.setdefault((f.ref, f.tema), f)
        return list(vistas.values())

    @model_validator(mode="after")
    def _herramientas_del_rol(self) -> "Aporte":
        """Un especialista solo puede haber usado SUS herramientas."""
        usadas = set(self.herramientas) | {ll.herramienta for ll in self.llamadas}
        ajenas = usadas - HERRAMIENTAS_POR_AGENTE[self.agente]
        if ajenas:
            raise ValueError(f"El {self.agente} no puede usar estas herramientas: {sorted(ajenas)}")
        return self


class DecisionSupervisor(BaseModel):
    """Salida estructurada del supervisor: el LLM está OBLIGADO a responder con esto."""

    model_config = ConfigDict(extra="forbid")

    evaluacion: str = Field(
        description=(
            "Evaluación breve de los aportes existentes contra la rúbrica "
            "(qué está bien, qué falta o qué hay que corregir)."
        ),
        max_length=1500,
    )
    siguiente: Destino = Field(
        description="Quién trabaja ahora: 'investigador', 'analista' o 'FINALIZAR' (pasar a la síntesis)."
    )
    instruccion: str = Field(
        default="",
        description=(
            "Instrucción concreta y autocontenida para el especialista elegido (qué tiene que "
            "hacer y con qué datos). Vacía si siguiente == 'FINALIZAR'."
        ),
        max_length=1500,
    )
    temas: list[Tema] = Field(
        default_factory=list,
        description=(
            "SOLO si siguiente == 'investigador': los temas que tiene que buscar ('pasajes', 'alojamiento', "
            "'comida', 'clima', 'actividades', 'cambio', 'otro'). Vacía en cualquier otro caso."
        ),
    )

    @model_validator(mode="after")
    def _instruccion_si_delega(self) -> "DecisionSupervisor":
        if self.siguiente != "FINALIZAR" and not self.instruccion.strip():
            raise ValueError("Si delega a un especialista, la instrucción no puede estar vacía.")
        if self.siguiente == "investigador" and not self.temas:
            raise ValueError("Si delega al investigador, 'temas' tiene que tener al menos un tema.")
        if self.siguiente != "investigador":
            self.temas = []  # los temas solo tienen sentido para el investigador
        self.temas = list(dict.fromkeys(self.temas))  # sin repetidos
        return self


Moneda = Literal["ARS", "USD"]

MESES: tuple[str, ...] = ("enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
                          "septiembre", "octubre", "noviembre", "diciembre")


class PedidoViaje(BaseModel):
    """Los datos del pedido TAL COMO LOS DIJO EL USUARIO (los extrae el supervisor).

    A propósito NO valida reglas de negocio (vuelta después de la ida, fecha
    futura, días que coincidan...): eso lo hace la herramienta validar_viaje
    del analista. Si este modelo las validara, un pedido con fechas imposibles
    ni siquiera se podría registrar y nadie le avisaría al usuario qué está mal.
    Solo controla tipos y que no haya campos inventados.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    origen: str | None = Field(default=None, description="Ciudad de salida.")
    destino: str | None = Field(default=None, description="Ciudad de destino.")
    fecha_ida: date | None = Field(default=None, description="Fecha de ida (AAAA-MM-DD).")
    fecha_vuelta: date | None = Field(default=None, description="Fecha de vuelta (AAAA-MM-DD).")
    dias: int | None = Field(default=None, description="Cantidad de días del viaje, si la dijo.")
    personas: int | None = Field(default=None, description="Cantidad de personas que viajan.")
    presupuesto: float | None = Field(default=None, description="Presupuesto TOTAL del viaje.")
    moneda: Moneda | None = Field(default=None, description="Moneda del presupuesto: 'ARS' (pesos) o 'USD'.")
    supuestos: list[str] = Field(
        default_factory=list,
        description="Supuestos hechos para completar datos faltantes (ej.: 'enero sin día exacto: del 10 al 16/01/2027').",
    )

    def como_texto(self) -> str:
        """El bloque 'DATOS DEL PEDIDO' que reciben los especialistas."""
        def v(x: object) -> str:
            return "no informado" if x is None else str(x)

        mes = MESES[self.fecha_ida.month - 1] if self.fecha_ida else None
        presupuesto = None if self.presupuesto is None else f"{self.presupuesto:,.0f}".replace(",", ".")
        lineas = [
            "DATOS DEL PEDIDO:",
            f"- Origen: {v(self.origen)}",
            f"- Destino: {v(self.destino)}",
            f"- Fecha de ida: {v(self.fecha_ida)}",
            f"- Fecha de vuelta: {v(self.fecha_vuelta)}",
            f"- Mes: {v(mes)}",
            f"- Días: {v(self.dias)}",
            f"- Personas: {v(self.personas)}",
            f"- Presupuesto total: {v(presupuesto)} {self.moneda or ''}".rstrip(),
        ]
        lineas += [f"- Supuesto: {s}" for s in self.supuestos]
        return "\n".join(lineas)


def contexto_especialista(instruccion: str, pedido: PedidoViaje | None, extra: str = "") -> str:
    """Lo ÚNICO que recibe un especialista: su instrucción + los datos del pedido (+ extra).

    Nunca incluye la pregunta original ni el historial de mensajes.
    """
    partes = [f"INSTRUCCIÓN DEL SUPERVISOR:\n{instruccion.strip()}"]
    if pedido is not None:
        partes.append(pedido.como_texto())
    if extra.strip():
        partes.append(extra.strip())
    return "\n\n".join(partes)


class DecisionRegistrada(BaseModel):
    """Una decisión del supervisor tal como quedó en el estado (para trazas y README)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    paso: int = Field(ge=1)
    siguiente: Destino
    instruccion: str = ""
    temas: list[Tema] = Field(default_factory=list)
    evaluacion: str = ""
    forzada: bool = Field(default=False, description="True si la impuso el freno de pasos, no el LLM.")
    timestamp: str = Field(default_factory=ahora)


# ---------------------------------------------------------------------------
# El estado del grafo
# ---------------------------------------------------------------------------
class OrquestadorState(MessagesState):
    """Estado compartido. Hereda `messages` de MessagesState.

    Campos propios (y quién los escribe):
    - pregunta            → se carga al inicio (la pregunta original del usuario).
    - pedido              → supervisor, UNA vez al principio (datos estructurados del
                            pedido; después queda fijo para toda la corrida).
    - aportes            → investigador y analista. Reductor operator.add: cada
                            aporte nuevo se SUMA a la lista, nunca pisa los anteriores.
    - pasos_supervisor    → supervisor (el contador del freno).
    - siguiente           → supervisor (a quién le toca).
    - instruccion_actual  → supervisor (lo que debe hacer el especialista elegido).
    - temas_pedidos       → supervisor (temas que tiene que buscar el investigador; estructurado).
    - decisiones          → supervisor (historial de decisiones, se suma).
    - respuesta_final     → síntesis.
    """

    pregunta: str
    pedido: PedidoViaje | None
    aportes: Annotated[list[Aporte], operator.add]
    pasos_supervisor: int
    siguiente: Destino
    instruccion_actual: str
    temas_pedidos: list[Tema]
    decisiones: Annotated[list[DecisionRegistrada], operator.add]
    respuesta_final: str


# Qué campos puede escribir cada nodo. `messages` lo pueden usar todos para
# dejar un mensaje corto (firmado con su nombre) en el historial.
CAMPOS_POR_NODO: dict[str, frozenset[str]] = {
    "supervisor": frozenset({"messages", "pedido", "pasos_supervisor", "siguiente", "instruccion_actual",
                             "temas_pedidos", "decisiones"}),
    "investigador": frozenset({"messages", "aportes"}),
    "analista": frozenset({"messages", "aportes"}),
    "sintesis": frozenset({"messages", "respuesta_final"}),
}


class ErrorDeEstado(Exception):
    """Un nodo intentó escribir algo que no le corresponde o con datos inválidos."""


def validar_actualizacion(nodo: Nodo, actualizacion: dict[str, Any]) -> dict[str, Any]:
    """Revisa lo que devuelve un nodo ANTES de que entre al estado.

    - Solo puede tocar sus campos (CAMPOS_POR_NODO).
    - Cada aporte tiene que ser un Aporte válido y estar firmado por ese mismo nodo.
    - Cada decisión tiene que ser una DecisionRegistrada válida.
    Devuelve la misma actualización (con dicts convertidos a modelos Pydantic).
    """
    permitidos = CAMPOS_POR_NODO[nodo]
    ajenos = set(actualizacion) - permitidos
    if ajenos:
        raise ErrorDeEstado(f"El nodo '{nodo}' intentó escribir campos ajenos: {sorted(ajenos)}")

    resultado = dict(actualizacion)
    try:
        if "aportes" in resultado:
            aportes = [a if isinstance(a, Aporte) else Aporte.model_validate(a) for a in resultado["aportes"]]
            firmados_por_otro = [a.agente for a in aportes if a.agente != nodo]
            if firmados_por_otro:
                raise ErrorDeEstado(f"El nodo '{nodo}' intentó firmar aportes como {firmados_por_otro}")
            resultado["aportes"] = aportes
        if resultado.get("pedido") is not None and not isinstance(resultado["pedido"], PedidoViaje):
            resultado["pedido"] = PedidoViaje.model_validate(resultado["pedido"])
        if "decisiones" in resultado:
            resultado["decisiones"] = [
                d if isinstance(d, DecisionRegistrada) else DecisionRegistrada.model_validate(d)
                for d in resultado["decisiones"]
            ]
    except ValidationError as e:
        raise ErrorDeEstado(f"El nodo '{nodo}' devolvió datos inválidos: {e}") from e
    return resultado


def estado_inicial(pregunta: str) -> dict[str, Any]:
    """El estado con el que arranca cada corrida."""
    from langchain_core.messages import HumanMessage

    pregunta = pregunta.strip()
    if not pregunta:
        raise ValueError("La pregunta no puede estar vacía.")
    return {
        "messages": [HumanMessage(content=pregunta)],
        "pregunta": pregunta,
        "pedido": None,
        "aportes": [],
        "pasos_supervisor": 0,
        "siguiente": "investigador",
        "instruccion_actual": "",
        "temas_pedidos": [],
        "decisiones": [],
        "respuesta_final": "",
    }


def validacion_del_pedido(llamadas: list[LlamadaHerramienta], pedido: PedidoViaje | None) -> str | None:
    """Resultado de validar_viaje para LAS FECHAS QUE DIO EL USUARIO: 'valido', 'invalido' o None (no se validó).

    Si el analista además probó fechas inventadas (pasó en la corrida real del escenario 4),
    esas validaciones NO cuentan: lo que importa es si el pedido real es posible.
    """
    validaciones = [ll for ll in llamadas if ll.herramienta == "validar_viaje"]
    if not validaciones:
        return None
    if pedido is not None and pedido.fecha_ida and pedido.fecha_vuelta:
        del_pedido = [ll for ll in validaciones
                      if str(ll.argumentos.get("fecha_ida")) == pedido.fecha_ida.isoformat()
                      and str(ll.argumentos.get("fecha_vuelta")) == pedido.fecha_vuelta.isoformat()]
        validaciones = del_pedido or validaciones
    return "valido" if validaciones[0].resultado.startswith("VÁLIDO") else "invalido"


def aportes_de(state: OrquestadorState, agente: Especialista) -> list[Aporte]:
    """Los aportes de un especialista, en orden."""
    return [a for a in state.get("aportes", []) if a.agente == agente]
