"""
tracing.py — La traza de cada corrida, en traces/<escenario>_<fecha>.json y .log.

- .json: estructurado (ruta, decisiones, llamadas a herramientas, fuentes,
  observaciones, métricas, tokens, timestamps). Es la evidencia de cada
  escenario y de ahí sale la tabla de resultados del README.
- .log: el mismo recorrido en texto legible, paso a paso.
"""

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import config
from state import Aporte, DecisionRegistrada, PedidoViaje, ahora

LARGO_LOG: int = 300  # caracteres de cada resultado de herramienta en el .log


@dataclass
class Evento:
    """Lo que hizo un nodo en un paso del grafo."""

    n: int
    nodo: str
    timestamp: str
    detalle: dict[str, Any]


@dataclass
class Traza:
    pregunta: str
    escenario: str = ""
    modo: str = "real"  # "real" | "mock"
    modelo: str = config.ANTHROPIC_MODEL
    max_pasos: int = config.MAX_PASOS_SUPERVISOR
    recursion_limit: int = config.RECURSION_LIMIT
    inicio: str = field(default_factory=ahora)
    fin: str = ""
    duracion_s: float = 0.0
    estado: str = "en_curso"  # "completo" | "cortado_por_recursion_limit" | "error"
    ruta: list[str] = field(default_factory=list)
    pedido: dict[str, Any] | None = None
    eventos: list[Evento] = field(default_factory=list)
    respuesta_final: str = ""
    metricas: dict[str, Any] = field(default_factory=dict)
    tokens: dict[str, Any] = field(default_factory=dict)
    archivo: str = ""

    # ------------------------------------------------------------------
    def registrar(self, nodo: str, salida: dict[str, Any]) -> None:
        """Registra lo que escribió un nodo en el estado."""
        self.ruta.append(nodo)
        detalle: dict[str, Any] = {}
        if nodo == "supervisor":
            if isinstance(salida.get("pedido"), PedidoViaje):
                detalle["pedido_extraido"] = salida["pedido"].model_dump(mode="json", exclude_none=True)
            for d in salida.get("decisiones", []):
                if isinstance(d, DecisionRegistrada):
                    detalle.update(d.model_dump(mode="json", exclude={"timestamp"}))
        elif nodo in ("investigador", "analista"):
            for a in salida.get("aportes", []):
                if isinstance(a, Aporte):
                    detalle.update(a.model_dump(mode="json", exclude={"timestamp", "agente"}))
        elif nodo == "sintesis":
            detalle["respuesta"] = salida.get("respuesta_final", "")
        self.eventos.append(Evento(n=len(self.eventos) + 1, nodo=nodo, timestamp=ahora(), detalle=detalle))

    def cerrar(self, estado: str, ultimo: dict[str, Any], uso_tokens: dict[str, Any] | None = None,
               aviso: str | None = None) -> None:
        """Marca el final de la corrida y calcula las métricas."""
        self.estado = estado
        self.fin = ahora()
        self.duracion_s = round((datetime.fromisoformat(self.fin) - datetime.fromisoformat(self.inicio)).total_seconds(), 2)
        self.respuesta_final = str(ultimo.get("respuesta_final") or aviso or "")
        if aviso and estado != "completo":
            self.respuesta_final = aviso
        pedido = ultimo.get("pedido")
        if isinstance(pedido, PedidoViaje):
            self.pedido = pedido.model_dump(mode="json")
        self.tokens = {modelo: dict(uso) for modelo, uso in (uso_tokens or {}).items()}
        self.metricas = self._calcular_metricas(ultimo)

    # ------------------------------------------------------------------
    def _calcular_metricas(self, ultimo: dict[str, Any]) -> dict[str, Any]:
        decisiones: list[DecisionRegistrada] = list(ultimo.get("decisiones", []))
        aportes: list[Aporte] = list(ultimo.get("aportes", []))
        llamadas = [ll for a in aportes for ll in a.llamadas]
        web_real = [ll for ll in llamadas if ll.herramienta == "buscar_en_web"
                    and not ll.resultado.startswith("NO SE BUSCÓ EN LA WEB")]
        fuentes = {(f.ref, f.origen) for a in aportes if a.agente == "investigador" for f in a.fuentes}
        validaciones = [ll.resultado for ll in llamadas if ll.herramienta == "validar_viaje"]
        calculos = [ll.resultado for ll in llamadas if ll.herramienta == "calcular_presupuesto"]
        delegaciones = [d.siguiente for d in decisiones if d.siguiente != "FINALIZAR"]
        veces = Counter(delegaciones)
        return {
            "ruta": " → ".join(self.ruta),
            "pasos_supervisor": len(decisiones),
            "decision_forzada_por_freno": any(d.forzada for d in decisiones),
            "delegaciones": dict(veces),
            "refinamientos": sum(n - 1 for n in veces.values() if n > 1),
            "llamadas_por_herramienta": dict(Counter(ll.herramienta for ll in llamadas)),
            "busquedas_web": len(web_real),
            "fuentes_memoria": sum(1 for _, o in fuentes if o == "memoria"),
            "fuentes_web": sum(1 for _, o in fuentes if o == "web"),
            "observaciones": [o for a in aportes for o in a.observaciones],
            "validacion": ("aprobada" if any(v.startswith("VÁLIDO") for v in validaciones)
                           else "rechazada" if validaciones else "no se validó"),
            "presupuesto_entra": (None if not calculos else "NO ENTRA" not in calculos[0]),
            "alternativa_calculada": len(calculos) > 1,
        }

    # ------------------------------------------------------------------
    def texto_ruta(self) -> str:
        return " → ".join(self.ruta)

    def etiqueta_busquedas(self) -> str:
        """En mock no hay búsquedas reales: Tavily está simulado."""
        return "búsquedas web simuladas" if self.modo == "mock" else "búsquedas web reales (Tavily)"

    def texto_origenes(self) -> str:
        m = self.metricas
        return f"{m.get('fuentes_memoria', 0)} de memoria · {m.get('fuentes_web', 0)} de la web"

    def guardar(self, carpeta: Path | None = None) -> Path:
        """Escribe <etiqueta>_<fecha>.json y .log; devuelve la ruta del .json."""
        carpeta = carpeta or config.TRACES_DIR  # se lee al guardar (los tests la cambian)
        carpeta.mkdir(parents=True, exist_ok=True)
        marca = datetime.fromisoformat(self.inicio).strftime("%Y%m%d-%H%M%S")
        etiqueta = re.sub(r"[^\w-]+", "_", self.escenario or "consulta").strip("_")
        base = carpeta / f"{etiqueta}_{self.modo}_{marca}"
        self.archivo = str(base.with_suffix(".json").relative_to(config.BASE_DIR)
                           if base.is_relative_to(config.BASE_DIR) else base.with_suffix(".json"))
        base.with_suffix(".json").write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")
        base.with_suffix(".log").write_text(self.como_texto(), encoding="utf-8")
        return base.with_suffix(".json")

    def como_texto(self) -> str:
        """Versión legible de la traza."""
        L: list[str] = [
            "=" * 90,
            f"TRAZA MULTI-AGENTE | escenario: {self.escenario or '-'} | modo: {self.modo} | modelo: "
            f"{self.modelo if self.modo == 'real' else 'LLMMock'}",
            f"MAX_PASOS_SUPERVISOR: {self.max_pasos} | RECURSION_LIMIT: {self.recursion_limit} | inicio: {self.inicio}",
            "=" * 90,
            f"PREGUNTA: {self.pregunta}",
            f"RUTA: {self.texto_ruta()}",
            "",
        ]
        for e in self.eventos:
            hora = e.timestamp[11:23]
            d = e.detalle
            if e.nodo == "supervisor":
                if "pedido_extraido" in d:
                    L.append(f"[{hora}] SUPERVISOR · pedido extraído: {json.dumps(d['pedido_extraido'], ensure_ascii=False)}")
                forz = " (FORZADA POR EL FRENO)" if d.get("forzada") else ""
                L.append(f"[{hora}] SUPERVISOR · decisión {d.get('paso')} → {d.get('siguiente')}{forz}")
                L.append(f"           evaluación: {d.get('evaluacion', '')}")
                if d.get("instruccion"):
                    L.append("           instrucción: " + d["instruccion"].replace("\n", " / "))
            elif e.nodo in ("investigador", "analista"):
                L.append(f"[{hora}] {e.nodo.upper()} · ronda {d.get('ronda')} · herramientas: "
                         f"{', '.join(d.get('herramientas', [])) or 'ninguna'}")
                for ll in d.get("llamadas", []):
                    args = json.dumps(ll.get("argumentos", {}), ensure_ascii=False)
                    res = ll.get("resultado", "").replace("\n", " ")
                    L.append(f"           ACCIÓN: {ll.get('herramienta')}({args[:250]}{'...' if len(args) > 250 else ''})")
                    L.append(f"           OBSERVACIÓN: {res[:LARGO_LOG]}{'...' if len(res) > LARGO_LOG else ''}")
                fuentes = d.get("fuentes", [])
                if fuentes:
                    L.append("           fuentes: " + ", ".join(f"{f['ref']} ({f['origen']}, {f['tema']})" for f in fuentes))
                L.append(f"           chequeo por código: {' | '.join(d.get('observaciones', [])) or 'sin observaciones'}")
            elif e.nodo == "sintesis":
                L.append(f"[{hora}] SÍNTESIS · respuesta final ({len(d.get('respuesta', ''))} caracteres)")
        m = self.metricas
        L += [
            "",
            "RESPUESTA FINAL:",
            self.respuesta_final,
            "",
            "-" * 90,
            f"ESTADO: {self.estado} | DURACIÓN: {self.duracion_s} s | DECISIONES DEL SUPERVISOR: {m.get('pasos_supervisor')}"
            f"{' (la última forzada por el freno)' if m.get('decision_forzada_por_freno') else ''}",
            f"REFINAMIENTOS: {m.get('refinamientos')} | {self.etiqueta_busquedas().upper()}: {m.get('busquedas_web')} | "
            f"DATOS: {self.texto_origenes()}",
            f"VALIDACIÓN: {m.get('validacion')} | PRESUPUESTO ENTRA: {m.get('presupuesto_entra')} | "
            f"HERRAMIENTAS: {json.dumps(m.get('llamadas_por_herramienta', {}), ensure_ascii=False)}",
            f"TOKENS: {json.dumps(self.tokens, ensure_ascii=False) if self.tokens else 'n/a (modo mock)'}",
            "",
        ]
        return "\n".join(L)
