"""
config.py — Configuración central del Asesor de Viajes multi-agente.

Todo lo que puede cambiar (claves, modelo, rutas, límites, modo mock) vive
acá, en UN solo lugar. El resto de los módulos importa estos valores.
"""

import logging
import os
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# RUTAS
# ---------------------------------------------------------------------------
BASE_DIR: Path = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")  # las claves se leen del .env (que NO se sube)

DATA_DIR: Path = BASE_DIR / "data"
CHROMA_DIR: Path = DATA_DIR / "memoria_chroma"            # memoria persistente (no se sube)
BUSQUEDAS_MOCK: Path = DATA_DIR / "mock_busquedas.json"   # resultados de Tavily prefijados
TRACES_DIR: Path = BASE_DIR / "traces"                    # trazas (.json y .log)

# ---------------------------------------------------------------------------
# CLAVES Y SERVICIOS (leídos del .env)
# ---------------------------------------------------------------------------
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "").strip()
ANTHROPIC_MODEL: str = os.getenv("ANTHROPIC_MODEL", "claude-haiku-5-5").strip()
TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "").strip()

# ---------------------------------------------------------------------------
# MODO MOCK: LLM con respuestas prefijadas + Tavily simulado + ChromaDB en una
# carpeta temporal. Permite correr la demo y los tests SIN claves.
# Se activa con MODO_MOCK=1 en el entorno o con `python main.py --mock`.
# ---------------------------------------------------------------------------
MODO_MOCK: bool = os.getenv("MODO_MOCK", "0").strip().lower() in {"1", "true", "si", "sí", "yes"}

# ---------------------------------------------------------------------------
# MEMORIA DE BÚSQUEDAS (ChromaDB)
# ---------------------------------------------------------------------------
CHROMA_COLECCION: str = "busquedas_viajes"
# Un dato guardado sirve si tiene a lo sumo esta antigüedad (precios y clima
# cambian: pasado este plazo se vuelve a buscar en la web).
MEMORIA_MAX_DIAS: int = 7
# Distancia coseno máxima para considerar relevante un resultado de memoria
# (0 = idéntico, 2 = opuesto). Calibrado con el modelo real (all-MiniLM-L6-v2
# ONNX) el 5/11/2026: la misma consulta da 0.09; consultas parecidas dichas
# con otras palabras, 0.24-0.44 ("vuelos baratos Mendoza Bariloche",
# "a cuánto está el dólar tarjeta"); consultas de otro tema, 0.64-0.72
# ("alquiler de autos en el aeropuerto", "nieve en julio para esquiar").
# El umbral queda en el medio del hueco entre los dos grupos.
MEMORIA_MAX_DISTANCIA: float = 0.55
TAVILY_MAX_RESULTADOS: int = 3  # pocos resultados = menos ruido y menos tokens

# Fecha de "hoy" para el sistema. Sirve para interpretar "en enero" como
# enero de 2027 y para validar que el viaje no sea en el pasado.
FECHA_REFERENCIA: date = date.today()

# ---------------------------------------------------------------------------
# LÍMITES (los frenos contra el "supervisor infinito")
# ---------------------------------------------------------------------------
# Freno 1 (el principal): cuántas veces puede decidir el supervisor. Al
# llegar a este número, el código fuerza FINALIZAR sin consultar al LLM.
MAX_PASOS_SUPERVISOR: int = 6
# Freno 2 (red de seguridad de LangGraph): techo de nodos ejecutados en el
# grafo principal. Cada vuelta supervisor → especialista son 2 nodos, así que
# 6 decisiones + síntesis entran holgadas en 20.
RECURSION_LIMIT: int = 20
# Freno 3: techo de pasos del ciclo ReAct INTERNO de cada especialista.
RECURSION_LIMIT_ESPECIALISTA: int = 12

# Tope de tokens de salida por llamada al LLM. Con 2500 se cortaban las llamadas grandes a
# calcular_presupuesto (argumentos vacíos) y la síntesis (corrida real del 5/11/2026).
MAX_TOKENS_RESPUESTA: int = 8000


class ErrorDeConfiguracion(Exception):
    """Falta una clave del .env: el programa no puede arrancar."""


def validar_entorno(modo_mock: bool = MODO_MOCK) -> None:
    """Verifica que las claves obligatorias estén en el .env.

    En modo mock no hace falta ninguna clave.
    """
    if modo_mock:
        return
    faltan: list[str] = [
        nombre
        for nombre, valor in {"ANTHROPIC_API_KEY": ANTHROPIC_API_KEY, "TAVILY_API_KEY": TAVILY_API_KEY}.items()
        if not valor
    ]
    if faltan:
        raise ErrorDeConfiguracion(
            f"Faltan variables en el .env: {', '.join(faltan)}. "
            "Completalas en el .env (copiá .env.example si no existe), "
            "o corré en modo mock con --mock."
        )


def configurar_logs(nivel: int = logging.INFO) -> None:
    """Logs con hora y milisegundos; se silencian las librerías ruidosas."""
    logging.basicConfig(
        level=nivel,
        format="%(asctime)s.%(msecs)03d | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    for ruidoso in ("httpx", "httpcore", "urllib3", "anthropic", "chromadb", "onnxruntime"):
        logging.getLogger(ruidoso).setLevel(logging.ERROR)
