"""
mock.py — Las "políticas" del modo mock: qué respondería cada agente.

Cada política es una función Python que mira los mensajes que recibe el LLM
(igual que Claude) y decide: llamar herramientas o dar la respuesta final.
No hay respuestas fijas por escenario: las políticas REACCIONAN a lo que
devuelven las herramientas reales (memoria ChromaDB, Tavily simulado,
calculadora). Por eso el modo mock ejercita el grafo de verdad: si la memoria
tiene el dato, el investigador mock no va a la web, igual que el real.

Convención para que el supervisor mock pueda pedir búsquedas puntuales: las
líneas de la instrucción con el formato "- <tema>: <consulta>" se usan como
consultas literales (así el escenario de refinamiento puede pedir primero una
búsqueda que no trae precios).
"""

import re
from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from pydantic import BaseModel

from llm import Politica, PoliticaEstructurada, nueva_llamada

if TYPE_CHECKING:
    from state import DecisionSupervisor, PedidoViaje

TEMAS: tuple[str, ...] = ("pasajes", "alojamiento", "clima", "actividades", "cambio")

# Palabras de la instrucción que indican cada tema.
PALABRAS_TEMA: dict[str, tuple[str, ...]] = {
    "pasajes": ("pasaje", "vuelo", "micro", "transporte"),
    "alojamiento": ("alojamiento", "hotel", "hostel", "cabaña"),
    "clima": ("clima", "temperatura", "tiempo"),
    "actividades": ("actividad", "excursion", "excursión", "qué hacer", "que hacer", "comida"),
    "cambio": ("cambio", "dólar", "dolar", "cotización", "cotizacion"),
}

# Consulta por defecto de cada tema.
CONSULTA_POR_TEMA: dict[str, str] = {
    "pasajes": "precio pasajes vuelo y micro {origen} {destino} ida y vuelta",
    "alojamiento": "precio alojamiento hostel hotel habitacion doble por noche {destino}",
    "clima": "clima en {destino} en {mes}",
    "actividades": "que hacer actividades excursiones y comida en {destino}",
    "cambio": "cotizacion dolar oficial tarjeta MEP hoy",
}


# ---------------------------------------------------------------------------
# Utilidades para leer los mensajes
# ---------------------------------------------------------------------------
def _instruccion(mensajes: list[BaseMessage]) -> str:
    humanos = [m for m in mensajes if isinstance(m, HumanMessage)]
    return str(humanos[0].content) if humanos else ""


def _campo(texto: str, nombre: str, defecto: str) -> str:
    m = re.search(rf"{nombre}\s*:\s*([^\n.,;|]+)", texto, re.IGNORECASE)
    return m.group(1).strip() if m else defecto


def _resultados(mensajes: list[BaseMessage]) -> list[tuple[dict[str, Any], ToolMessage]]:
    """Pares (pedido de herramienta, resultado), en orden."""
    pedidos = {tc["id"]: tc for m in mensajes if isinstance(m, AIMessage) for tc in m.tool_calls}
    return [(pedidos.get(m.tool_call_id, {}), m) for m in mensajes if isinstance(m, ToolMessage)]


# ---------------------------------------------------------------------------
# Investigador
# ---------------------------------------------------------------------------
def consultas_pedidas(instruccion: str) -> dict[str, str]:
    """Qué temas pide la instrucción y con qué consulta buscar cada uno."""
    pedidos: list[str] = []
    if (m := re.search(r"TEMAS A BUSCAR:\s*(.+)", instruccion)):  # lista estructurada del supervisor
        pedidos = [t.strip() for t in m.group(1).split(",") if t.strip() in (*TEMAS, "comida")]
        instruccion = instruccion.replace(m.group(0), "")
    explicitas = {
        t: c.strip()
        for t, c in re.findall(r"^\s*-\s*(" + "|".join(TEMAS) + r")\s*:\s*(.+)$", instruccion, re.IGNORECASE | re.MULTILINE)
    }
    if explicitas:
        return {t.lower(): c for t, c in explicitas.items()}
    texto = instruccion.lower()
    destino = _campo(instruccion, "destino", "Bariloche")
    origen = _campo(instruccion, "origen", "")
    mes = _campo(instruccion, "mes", "enero")
    # En el mock la comida viene en la misma página que las actividades.
    temas = list(dict.fromkeys("actividades" if t == "comida" else t for t in pedidos)) \
        or [t for t in TEMAS if any(p in texto for p in PALABRAS_TEMA[t])] or ["pasajes", "alojamiento"]
    return {
        t: re.sub(r"\s+", " ", CONSULTA_POR_TEMA[t].format(origen=origen, destino=destino, mes=mes)).strip()
        for t in temas
    }


def politica_investigador(mensajes: list[BaseMessage], _herramientas: list[str]) -> AIMessage:
    instruccion = _instruccion(mensajes)
    consultas = consultas_pedidas(instruccion)
    destino = _campo(instruccion, "destino", "Bariloche")
    forzar = "forzar" in instruccion.lower()
    hechos = _resultados(mensajes)

    def llamadas_de(herramienta: str, tema: str) -> list[ToolMessage]:
        return [r for p, r in hechos if p.get("name") == herramienta and p.get("args", {}).get("tema") == tema]

    # Paso 1: memoria primero (salvo que se pida forzar la web).
    if not forzar:
        pendientes = [t for t in consultas if not llamadas_de("buscar_en_memoria", t)]
        if pendientes:
            return AIMessage(
                content=f"Primero reviso la memoria para: {', '.join(pendientes)}.",
                tool_calls=[nueva_llamada("buscar_en_memoria", {"consulta": consultas[t], "destino": destino, "tema": t})
                            for t in pendientes],
            )

    # Paso 2: web para lo que la memoria no tenía (o todo, si se pidió forzar).
    a_la_web = [
        t for t in consultas
        if not llamadas_de("buscar_en_web", t)
        and (forzar or any(str(r.content).startswith(("SIN RESULTADOS", "ERROR")) for r in llamadas_de("buscar_en_memoria", t)))
    ]
    if a_la_web:
        return AIMessage(
            content=f"La memoria no tiene datos vigentes de: {', '.join(a_la_web)}. Busco en la web.",
            tool_calls=[nueva_llamada("buscar_en_web", {"consulta": consultas[t], "destino": destino, "tema": t, "forzar": forzar})
                        for t in a_la_web],
        )

    # Paso 3: respuesta final con lo encontrado.
    return AIMessage(content=_redactar_hallazgos(consultas, hechos))


def _redactar_hallazgos(consultas: dict[str, str], hechos: list[tuple[dict[str, Any], ToolMessage]]) -> str:
    lineas = ["## Hallazgos"]
    faltan: list[str] = []
    for tema in consultas:
        del_tema = [r for p, r in hechos if p.get("args", {}).get("tema") == tema]
        utiles = [r for r in del_tema if r.artifact]
        if not utiles:
            motivo = next((str(r.content)[:120] for r in del_tema if str(r.content).startswith("ERROR")), "sin resultados")
            faltan.append(f"{tema}: {motivo}")
            continue
        lineas.append(f"### {tema.capitalize()}")
        vistas: set[str] = set()
        for r in utiles:
            for bloque in str(r.content).split("\n\n")[1:]:
                cabecera, _, cuerpo = bloque.partition("\n")
                ref = cabecera[1:cabecera.find("]")] if cabecera.startswith("[") else ""
                if not ref or ref in vistas:
                    continue
                vistas.add(ref)
                origen = "memoria" if ref.startswith("mem-") else "web"
                if origen == "web" and "\n" in cuerpo:  # en la web, la 1.ª línea es el título de la página
                    titulo, _, cuerpo = cuerpo.partition("\n")
                    cuerpo = f"{titulo.strip()}: {cuerpo}"
                texto = " ".join(cuerpo.split())[:700]
                lineas.append(f"- {texto} — [{ref}] ({origen})")
    lineas.append("## Datos que faltan")
    lineas.extend(f"- {f}" for f in faltan) if faltan else lineas.append("- ninguno")
    return "\n".join(lineas)


# ---------------------------------------------------------------------------
# Analista
# ---------------------------------------------------------------------------
def _numero(texto: str) -> float:
    """'800.000' → 800000.0 ; '30' → 30.0 (formato argentino)."""
    return float(texto.replace(".", "").replace(",", "."))


def leer_pedido(texto: str) -> dict[str, str]:
    """Lee el bloque DATOS DEL PEDIDO ('- Campo: valor')."""
    bloque = texto.split("DATOS DEL PEDIDO:", 1)[-1].split("HALLAZGOS DEL INVESTIGADOR", 1)[0]
    return {k.strip().lower(): v.strip() for k, v in re.findall(r"^- ([^:]+):\s*(.+)$", bloque, re.MULTILINE)}


def _precio(hallazgos: str, patron: str) -> tuple[float, str] | None:
    """Busca un precio con `patron` (grupo 1 = monto) y devuelve (monto, fuente de esa línea)."""
    for linea in hallazgos.splitlines():
        m = re.search(patron, linea, re.IGNORECASE)
        if m:
            ref = re.search(r"\[((?:https?://|mem-)[^\]]+)\]", linea)
            return _numero(m.group(1)), (ref.group(1) if ref else "")
    return None


def precios_de_hallazgos(hallazgos: str) -> dict[str, Any]:
    """Extrae del texto del investigador los precios que necesita el cálculo (datos del mock)."""
    p: dict[str, Any] = {}
    if (x := _precio(hallazgos, r"[Ii]da y vuelta en semicama: \$([\d.]+) por persona")):
        p["pasaje"] = {"monto": x[0], "moneda": "ARS", "tramo": "ida_y_vuelta", "medio": "micro semicama", "fuente": x[1]}
    elif (x := _precio(hallazgos, r"[Tt]arifa promedio ida y vuelta: \$([\d.]+) por persona")):
        p["pasaje"] = {"monto": x[0], "moneda": "ARS", "tramo": "ida_y_vuelta", "medio": "avión", "fuente": x[1]}
    if (x := _precio(hallazgos, r"habitación doble privada desde \$([\d.]+)")):
        p["alojamiento"] = {"monto": x[0], "moneda": "ARS", "unidad": "por_noche_por_habitacion",
                            "personas_por_habitacion": 2, "tipo": "hostel habitación doble", "fuente": x[1]}
    if (x := _precio(hallazgos, r"cocinando en el hostel, \$([\d.]+) por persona por día")):
        p["comida"] = {"monto": x[0], "moneda": "ARS", "fuente": x[1]}
    if (x := _precio(hallazgos, r"gasto promedio es de \$([\d.]+) por persona por día")):
        p["actividades"] = {"monto": x[0], "moneda": "ARS", "fuente": x[1]}
    if (x := _precio(hallazgos, r"Isla Victoria[^.]*?US\$ ?([\d.]+) por persona")):
        p["extra"] = {"descripcion": "excursión Isla Victoria", "monto": x[0], "moneda": "USD",
                      "medio_pago": "tarjeta", "fuente": x[1]}
    cambios = []
    for tipo, patron in (("oficial", r"dólar oficial \$([\d.]+)"), ("tarjeta", r"dólar tarjeta \$([\d.]+)"),
                         ("MEP", r"dólar MEP \$([\d.]+)")):
        if (x := _precio(hallazgos, patron)):
            cambios.append({"tipo": tipo, "valor_ars": x[0], "fuente": x[1]})
    p["tipos_de_cambio"] = cambios
    return p


def _args_presupuesto(pedido: dict[str, str], precios: dict[str, Any], noches: int, opcion: str,
                      con_extras: bool = True) -> dict[str, Any]:
    monto, _, moneda = pedido.get("presupuesto total", "0 ARS").rpartition(" ")
    args: dict[str, Any] = {
        "opcion": opcion, "personas": int(pedido.get("personas", "1")), "noches": noches,
        "presupuesto": _numero(monto), "moneda_presupuesto": moneda,
        "pasaje": precios["pasaje"], "alojamiento": precios["alojamiento"],
        "comida_por_persona_por_dia": precios["comida"],
        "tipos_de_cambio": precios["tipos_de_cambio"],
    }
    if "actividades" in precios:
        args["actividades_por_persona_por_dia"] = precios["actividades"]
    if con_extras and "extra" in precios:
        args["extras"] = [precios["extra"]]
    return args


def politica_analista(mensajes: list[BaseMessage], _herramientas: list[str]) -> AIMessage:
    from agents.analyst_tools import CalcularPresupuestoInput, ErrorDeCambio, calcular

    entrada = _instruccion(mensajes)
    pedido = leer_pedido(entrada)
    hallazgos = entrada.split("HALLAZGOS DEL INVESTIGADOR:", 1)[-1]
    hechos = _resultados(mensajes)
    validaciones = [r for p, r in hechos if p.get("name") == "validar_viaje"]
    calculos = [(p, r) for p, r in hechos if p.get("name") == "calcular_presupuesto"]

    # Paso 1: validar el viaje.
    if not validaciones:
        args = {
            "origen": pedido.get("origen", ""), "destino": pedido.get("destino", ""),
            "fecha_ida": pedido.get("fecha de ida"), "fecha_vuelta": pedido.get("fecha de vuelta"),
            "personas": int(pedido.get("personas", "0") or 0),
            "presupuesto": _numero(pedido.get("presupuesto total", "0 ARS").rpartition(" ")[0] or "0"),
            "moneda": pedido.get("presupuesto total", "0 ARS").rpartition(" ")[2],
        }
        if pedido.get("días", "no informado").isdigit():
            args["dias_pedidos"] = int(pedido["días"])
        return AIMessage(content="Primero valido los datos del viaje.", tool_calls=[nueva_llamada("validar_viaje", args)])

    validacion = str(validaciones[-1].content)
    if not validacion.startswith("VÁLIDO"):  # INVÁLIDO, o argumentos que ni siquiera se pudieron leer
        return AIMessage(content=f"## Validación\n{validacion}\n## Presupuesto\nNo se calculó: primero hay que "
                                 "corregir los datos del viaje.\n## Conclusión\nEl viaje, tal como está pedido, no es "
                                 "posible.\n## Datos que faltan\n- fechas válidas")

    noches = int(re.search(r"(\d+) noches", validacion).group(1))  # type: ignore[union-attr]
    precios = precios_de_hallazgos(hallazgos)
    faltan = [k for k in ("pasaje", "alojamiento", "comida") if k not in precios]
    if faltan:
        return AIMessage(content=f"## Validación\n{validacion}\n## Presupuesto\nNo se puede calcular.\n"
                                 f"## Conclusión\nFaltan precios.\n## Datos que faltan\n- precios de: {', '.join(faltan)}")

    # Paso 2: opción económica.
    if not calculos:
        return AIMessage(content="Calculo la opción económica con los precios más baratos.",
                         tool_calls=[nueva_llamada("calcular_presupuesto",
                                                   _args_presupuesto(pedido, precios, noches, "económica"))])

    # Paso 3: si no entra, UNA alternativa con menos días y sin extras (la mayor que entre).
    primero = str(calculos[0][1].content)
    if "NO ENTRA" in primero and len(calculos) == 1:
        alternativa = 1
        for n in range(noches - 1, 0, -1):
            try:
                if calcular(CalcularPresupuestoInput.model_validate(
                        _args_presupuesto(pedido, precios, n, "prueba", con_extras=False))).entra:
                    alternativa = n
                    break
            except ErrorDeCambio:
                break
        return AIMessage(content="No entra: calculo una alternativa con menos días y sin excursiones pagas.",
                         tool_calls=[nueva_llamada("calcular_presupuesto", _args_presupuesto(
                             pedido, precios, alternativa, f"alternativa: {alternativa + 1} días sin excursión", False))])

    # Paso 4: respuesta final.
    textos = [str(r.content) for _, r in calculos]
    entra_principal = "ENTRA en el presupuesto" in textos[0] and "NO ENTRA" not in textos[0]
    if entra_principal:
        conclusion = "Con la opción económica el viaje ENTRA en el presupuesto."
    elif len(textos) > 1 and "NO ENTRA" not in textos[1]:
        conclusion = "Con la opción económica NO ENTRA; la alternativa con menos días sí entra."
    else:
        conclusion = "NO ENTRA en el presupuesto, ni siquiera con la alternativa más barata."
    return AIMessage(content=f"## Validación\n{validacion}\n## Presupuesto\n" + "\n\n".join(textos)
                     + f"\n## Conclusión\n{conclusion}\n## Datos que faltan\n- ninguno")


# ---------------------------------------------------------------------------
# Supervisor (salida estructurada: PedidoViaje y DecisionSupervisor)
# ---------------------------------------------------------------------------
def extraer_pedido_mock(pregunta: str) -> "PedidoViaje":
    """Saca los datos del pedido con expresiones regulares (lo que haría el LLM)."""
    from datetime import date, timedelta

    import config
    from state import MESES, PedidoViaje

    q = pregunta
    def num(patron: str) -> int | None:
        m = re.search(patron, q, re.IGNORECASE)
        return int(m.group(1)) if m else None

    dias = num(r"(\d+)\s*d[ií]as")
    personas = num(r"somos\s+(\d+)") or num(r"(\d+)\s+personas")
    presupuesto, moneda = None, None
    if (m := re.search(r"([\d.]{3,})\s*(pesos|ARS|d[oó]lares|USD)", q, re.IGNORECASE)):
        presupuesto = _numero(m.group(1))
        moneda = "USD" if m.group(2).lower().startswith(("d", "u")) else "ARS"
    ciudad = r"([A-ZÁÉÍÓÚÑ][\wáéíóúñ]+)"
    origen = (re.search(rf"desde\s+{ciudad}", q) or re.search(rf"\bde\s+{ciudad}\s+a\s+", q))
    destino = re.search(rf"\b(?:a|en)\s+{ciudad}", q)

    supuestos: list[str] = []
    ida = vuelta = None
    hoy = config.FECHA_REFERENCIA
    mes = next((i + 1 for i, nombre in enumerate(MESES) if nombre in q.lower()), None)
    if mes:
        anio = hoy.year if (mes, 10) >= (hoy.month, hoy.day) else hoy.year + 1
        if (m := re.search(r"del\s+(\d{1,2})\s+al\s+(\d{1,2})", q)):
            ida, vuelta = date(anio, mes, int(m.group(1))), date(anio, mes, int(m.group(2)))
        else:
            ida = date(anio, mes, 10)
            if dias:
                vuelta = ida + timedelta(days=dias - 1)
            supuestos.append(f"{MESES[mes - 1]} sin día exacto: se asume del {ida:%d/%m/%Y}"
                             + (f" al {vuelta:%d/%m/%Y}" if vuelta else ""))
    return PedidoViaje(
        origen=origen.group(1) if origen else None, destino=destino.group(1) if destino else None,
        fecha_ida=ida, fecha_vuelta=vuelta, dias=dias, personas=personas,
        presupuesto=presupuesto, moneda=moneda, supuestos=supuestos,  # type: ignore[arg-type]
    )


def _aportes_resumidos(texto: str) -> list[dict[str, str]]:
    """Lee los 'APORTE #n | agente=... ' que ve el supervisor."""
    aportes = []
    for bloque in re.split(r"\n\n(?=APORTE #)", texto.split("APORTES HASTA AHORA:", 1)[-1]):
        m = re.search(r"APORTE #\d+ \| agente=(\w+) \| ronda=(\d+) \| herramientas=([^|]*)\|", bloque)
        if not m:
            continue
        obs = re.search(r"Observaciones del chequeo por código: (.*)", bloque)
        aportes.append({"agente": m.group(1), "herramientas": m.group(3).strip(),
                        "observaciones": obs.group(1).strip() if obs else "ninguna", "texto": bloque})
    return aportes


def decidir_mock(texto: str) -> "DecisionSupervisor":
    from state import DecisionSupervisor

    pregunta = texto.split("PREGUNTA DEL USUARIO:\n", 1)[-1].split("\n\nDATOS DEL PEDIDO", 1)[0]
    pedido = leer_pedido(texto)
    con_presupuesto = pedido.get("presupuesto total", "no informado") != "no informado"
    aportes = _aportes_resumidos(texto)
    inv = [a for a in aportes if a["agente"] == "investigador"]
    ana = [a for a in aportes if a["agente"] == "analista"]
    origen, destino = pedido.get("origen", ""), pedido.get("destino", "")

    if not inv:
        if "como llegar" in pregunta.lower().replace("ó", "o"):
            instr = f"Averiguá cómo llegar y cuánto sale el pasaje.\n- pasajes: como llegar de {origen} a {destino}"
        elif con_presupuesto:
            instr = "Buscá precios de pasajes y alojamiento, comida y actividades, el clima y la cotización del dólar."
        else:
            instr = f"Buscá lo que pide el usuario: {pregunta}"
        temas = ["pasajes"] if "como llegar" in pregunta.lower().replace("ó", "o") else list(consultas_pedidas(instr))
        return DecisionSupervisor(evaluacion="R1: todavía no hay datos. Empieza el investigador.",
                                  siguiente="investigador", instruccion=instr, temas=temas)  # type: ignore[arg-type]

    ultimo_inv = inv[-1]
    if "precio" in ultimo_inv["observaciones"].lower() and len(inv) < 2:
        return DecisionSupervisor(
            evaluacion=f"R3/R5 NO se cumplen: {ultimo_inv['observaciones']} Se devuelve el trabajo al investigador.",
            siguiente="investigador",
            instruccion=("Rehacer con forzar=true: la búsqueda anterior no trajo precios.\n"
                         f"- pasajes: precio pasaje vuelo micro {origen} {destino} ida y vuelta"),
            temas=["pasajes"],
        )
    if con_presupuesto and not ana:
        return DecisionSupervisor(
            evaluacion="R1-R3 OK (hay datos con fuentes y precios). R4 pendiente: falta validar y calcular.",
            siguiente="analista",
            instruccion=("Validá el viaje con los DATOS DEL PEDIDO y calculá el presupuesto (opción económica; "
                         "si no alcanza, una alternativa) con los precios de los hallazgos."),
        )
    falla_web = "Falló una búsqueda" in ultimo_inv["observaciones"]
    if ana and ana[-1]["observaciones"] != "ninguna" and len(ana) < 2 and not falla_web:
        return DecisionSupervisor(evaluacion=f"R5 NO se cumple: {ana[-1]['observaciones']}", siguiente="analista",
                                  instruccion=f"Corregí lo siguiente y respondé de nuevo: {ana[-1]['observaciones']}")
    evaluacion = "R1-R6 OK: hay datos con fuentes" + (", validación y cálculo del analista." if ana else ".")
    if falla_web:
        evaluacion = "R5: la búsqueda web falló y no se puede resolver; se finaliza avisando qué faltó."
    elif ana and ana[-1]["observaciones"] != "ninguna":
        evaluacion = (f"R5 NO se cumple y ya se reintentó: {ana[-1]['observaciones']} "
                      "Se finaliza avisando qué no se pudo resolver.")
    return DecisionSupervisor(evaluacion=evaluacion, siguiente="FINALIZAR")


def politica_supervisor_estructurada(mensajes: list[BaseMessage], esquema: type[BaseModel]) -> BaseModel:
    texto = _instruccion(mensajes)
    if esquema.__name__ == "PedidoViaje":
        return extraer_pedido_mock(texto)
    return decidir_mock(texto)


# ---------------------------------------------------------------------------
# Síntesis
# ---------------------------------------------------------------------------
def _seccion(texto: str, titulo: str) -> str:
    m = re.search(rf"## {titulo}\n(.*?)(?=\n## |\Z)", texto, re.DOTALL)
    return m.group(1).strip() if m else ""


def politica_sintesis(mensajes: list[BaseMessage], _herramientas: list[str]) -> AIMessage:
    from datetime import date, timedelta

    texto = _instruccion(mensajes)
    pedido = leer_pedido(texto)
    analista = texto.rsplit("APORTE DEL ANALISTA", 1)[-1] if "APORTE DEL ANALISTA" in texto else ""
    investigador = "\n".join(re.findall(r"APORTE DEL INVESTIGADOR.*?(?=\n\nAPORTE DEL |\n\nATENCIÓN|\Z)", texto, re.DOTALL))
    partes: list[str] = []

    if "INVÁLIDO" in analista:
        errores = [ln for ln in analista.splitlines() if ln.startswith("- ") and ":" in ln]
        partes.append("## El viaje no se puede armar tal como está pedido\n" + "\n".join(errores)
                      + "\n\nCorregí esos datos (por ejemplo, que la vuelta sea después de la ida) y lo armamos.")
    elif "PRESUPUESTO —" in analista:
        destino = pedido.get("destino", "el destino")
        ida_txt = pedido.get("fecha de ida", "")
        actividades = [a for a in ("Circuito Chico en bici", "trekking al Refugio Frey", "playas del lago Gutiérrez",
                                   "excursión a Isla Victoria y Bosque de Arrayanes", "miradores y centro cívico")
                       if a.split()[-1].lower() in investigador.lower()] or ["recorrer la ciudad"]
        clima = next((ln.split(" — ")[0][2:].split(": ", 1)[-1] for ln in investigador.splitlines() if "°C" in ln), "")
        dias = int(pedido.get("días", "0")) if pedido.get("días", "").isdigit() else 0
        lineas = ["## Itinerario"]
        try:
            ida = date.fromisoformat(ida_txt)
        except ValueError:
            ida = None
        for i in range(dias):
            fecha = f" ({(ida + timedelta(days=i)):%d/%m})" if ida else ""
            if i == 0:
                plan = f"viaje desde {pedido.get('origen', 'el origen')} y llegada a {destino}"
            elif i == dias - 1:
                plan = f"regreso a {pedido.get('origen', 'casa')}"
            elif i - 1 < len(actividades):
                plan = actividades[i - 1]
            else:  # más días que actividades: días libres distintos, nunca se repite una actividad
                libres = ["día libre: descanso y paseo por el centro",
                          "día de reserva por si algún día llueve o hay mucho viento"]
                k = i - 1 - len(actividades)
                plan = libres[k] if k < len(libres) else f"día libre ({k - len(libres) + 3})"
            lineas.append(f"- Día {i + 1}{fecha}: {plan}")
        if clima:
            lineas.append(f"\nClima esperado: {clima}")
        partes.append("\n".join(lineas))
        partes.append("## Presupuesto\n" + _seccion(analista, "Presupuesto"))
        conclusion = _seccion(analista, "Conclusión")
        if conclusion:
            partes.append(f"**{conclusion}**")
        if "NO ENTRA" in _seccion(analista, "Presupuesto").split("PRESUPUESTO —")[1]:
            partes.append("## Alternativas\n- La opción calculada por el analista con menos días (ver arriba).\n"
                          "- Viajar en micro en lugar de avión y cocinar en el hostel.\n"
                          "- Priorizar actividades gratuitas (trekking, playas, miradores) y dejar las excursiones pagas.\n"
                          "- Habitación compartida de hostel en lugar de habitación privada.")
    else:
        hallazgos = [ln for ln in investigador.splitlines() if ln.startswith("- ") and "[" in ln]
        partes.append("## Lo que encontramos\n" + ("\n".join(hallazgos) if hallazgos else "- No encontramos datos."))

    faltan = [ln for bloque in re.findall(r"## Datos que faltan\n(.*?)(?=\n## |\n\nAPORTE|\Z)", texto, re.DOTALL)
              for ln in bloque.splitlines() if ln.startswith("- ") and "ninguno" not in ln]
    if faltan:
        partes.append("## Lo que no pudimos averiguar\n" + "\n".join(dict.fromkeys(faltan)))
    if "ATENCIÓN: el trabajo se cortó" in texto:
        partes.append("_Nota: el sistema llegó a su límite de pasos; puede faltar información._")
    return AIMessage(content="\n\n".join(partes))


# ---------------------------------------------------------------------------
# Registro de políticas por rol: (política de texto, política estructurada)
# ---------------------------------------------------------------------------
def _sin_politica(rol: str) -> Politica:
    def _p(_m: list[BaseMessage], _h: list[str]) -> AIMessage:
        raise NotImplementedError(f"Todavía no hay política mock para el rol '{rol}'.")
    return _p


def _sin_estructurada(rol: str) -> PoliticaEstructurada:
    def _p(_m: list[BaseMessage], _s: type[BaseModel]) -> BaseModel:
        raise NotImplementedError(f"Todavía no hay política estructurada mock para el rol '{rol}'.")
    return _p


POLITICAS: dict[str, tuple[Politica, PoliticaEstructurada]] = {
    "investigador": (politica_investigador, _sin_estructurada("investigador")),
    "analista": (politica_analista, _sin_estructurada("analista")),
    "supervisor": (_sin_politica("supervisor"), politica_supervisor_estructurada),
    "sintesis": (politica_sintesis, _sin_estructurada("sintesis")),
}
