# PROGRESO — Pre-entrega 6 (Asesor de viajes multi-agente)

> Última actualización: 5/11/2026, 21:40. Estado: **código completo; 120 tests OK; los 8 escenarios corridos en
> mock y los 7 posibles en real (el 7 es solo mock), con sus trazas.**
> Faltan: el notebook, el README, la prueba de instalación limpia, la revisión final y la subida a GitHub.

## 1. Qué está terminado

| Pieza | Archivo | Estado |
|---|---|---|
| Configuración (claves, rutas, 3 frenos, modo mock) | `config.py`, `.env.example`, `.gitignore` | ✅ |
| Estado `OrquestadorState(MessagesState)` + Pydantic (`Fuente`, `Aporte`, `LlamadaHerramienta`, `PedidoViaje`, `DecisionSupervisor`, `DecisionRegistrada`) + `validar_actualizacion` (cada nodo escribe solo su parte) | `state.py` | ✅ |
| Fábrica de LLM: Claude real o `LLMMock` (políticas por rol) | `llm.py`, `agents/mock.py` | ✅ |
| Memoria ChromaDB (filtros: destino/tema + 7 días + distancia ≤ 0,55, calibrada con el modelo real) | `agents/memory.py` | ✅ |
| Investigador: `buscar_en_memoria` + `buscar_en_web` (Tavily) + chequeo por código | `agents/research_tools.py`, `agents/research_agent.py`, `agents/common.py` | ✅ |
| Analista: `validar_viaje` + `calcular_presupuesto` + chequeo por código | `agents/analyst_tools.py`, `agents/analyst_agent.py` | ✅ |
| Supervisor: extrae el pedido, salida estructurada con `Literal`, rúbrica R1–R6, 4 reglas duras, freno de pasos, `ruta_supervisor` | `agents/supervisor.py` | ✅ |
| Síntesis: fuentes y avisos armados por código + **chequeo de actividades repetidas** (pide una corrección) | `agents/synthesizer.py` | ✅ |
| Grafo (`StateGraph` + `add_conditional_edges`) y `consultar()` | `graph.py` | ✅ |
| Trazas `.json` + `.log` con métricas | `tracing.py` | ✅ |
| CLI: `--mock`, `--chat`, `--max-pasos`, `--fallar-web`, `--mermaid`, `-v` | `main.py` | ✅ |
| Corrida de los 8 escenarios + tabla `traces/resultados.md` armada desde las trazas | `escenarios.py` | ✅ |
| Tests: **120 pasan en ~8 s, sin claves** | `tests/` | ✅ |

Para retomar:
- `.venv\Scripts\python.exe -m pytest -q` → tiene que dar 120 passed.
- `.venv\Scripts\python.exe escenarios.py --tabla` → rearma la tabla sin correr nada.

## 2. Los 3 detalles que pediste

| Detalle | Estado | Verificación |
|---|---|---|
| Actividades repetidas entre días | ✅ | Mock: los días que sobran son "día libre". Real: chequeo por código en la síntesis; si detecta un lugar repetido, pide UNA corrección al LLM y, si sigue, avisa. **Se activó en vivo** en el escenario 5 (Cerro Catedral en los días 2 y 6) y la corrección quedó sin repeticiones. Tests en `test_synthesizer.py`. |
| "Búsquedas web simuladas" en mock | ✅ | `Traza.etiqueta_busquedas()`. |
| Plurales ("2 personas", "1 habitación") | ✅ | `cantidad()` en `state.py` + test. |

## 3. Resultados de los escenarios (resumen; detalle en `traces/resultados.md`)

| # | Escenario | Mock | Real |
|---|---|---|---|
| 1 | Flujo completo | ✅ sup → inv → sup → ana → sup → síntesis; entra ($785.100) | ✅ con 1 refinamiento del investigador; "no alcanza" ($1,6 M–$1,9 M) y alternativa de 4 días que entra |
| 2 | Solo clima | ✅ sin analista | ✅ sin analista, 23 s |
| 3 | No alcanza ($400.000) | ✅ alternativa calculada | ✅ no entra; datos 100 % de memoria (0 búsquedas en Tavily) |
| 4 | Fechas inválidas | ✅ rechazada | ✅ rechazada, sin itinerario |
| 5 | Memoria (misma pregunta que el 1) | ✅ 6 de memoria · 0 de la web | ⚠️ **parcial:** 22 de memoria pero 3 búsquedas nuevas en Tavily, y el supervisor volvió al investigador después del analista |
| 6 | Refinamiento | ✅ rechaza la búsqueda sin precios y la rehace | ⚠️ **no se dio en vivo:** la primera búsqueda ya trajo precios, así que no hubo nada que rechazar (la evidencia del refinamiento es la del mock y el refinamiento espontáneo de los escenarios 1, 3 y 4 reales) |
| 7 | Tavily caído | ✅ termina y avisa qué faltó | — (solo mock) |
| 8 | Freno (`max_pasos=2`) | ✅ forzado + aviso | ✅ forzado + aviso de información parcial |

## 4. Qué falta, en orden

1. **Revisar el escenario 5 real:** por qué el investigador hizo 3 búsquedas en Tavily habiendo memoria (probablemente consultas con otras palabras o temas nuevos) y por qué el supervisor volvió al investigador después del analista. Decidir si se ajusta o si se documenta tal cual.
2. **`demo.ipynb`** ejecutado con salidas visibles: el flujo de delegación de la pregunta de demo (mock y real).
3. **README.md**:
   - diagrama Mermaid (`python main.py --mermaid`);
   - por qué la topología jerárquica y cómo se manejan los conflictos entre agentes;
   - qué es Tavily y por qué la web; por qué ChromaDB y no Pinecone; `TavilySearch` en lugar de `TavilySearchResults`;
   - `create_react_agent` deprecado (se usa igual porque lo nombra la consigna); async;
   - instalación desde cero; tabla de escenarios (de `traces/resultados.md`); tabla de criterios;
   - "lo que aprendimos de las corridas reales" (trazas `0_` y `0b_`).
4. **`requirements.txt` final** (sumar `langchain` 1.4.7, que entra como dependencia de `langchain-tavily`) y **prueba en un venv limpio**.
5. **Revisión punto por punto** contra `CONSIGNA.md`, los 4 criterios y `MEJORAS_TICHER.md`.
6. **Repo público `pre-entrega-6`** en GitHub (anukvilarasau) y push. Antes, preguntar si se borra `EXTRA_MCP.md` del historial (entró en el primer commit). NO entregar en Coderhouse.

## 5. Decisiones que tomamos

- **Tema:** asesor de viajes (reemplazó a httpx/Pinecone). Pregunta de demo: Bariloche, 7 días, 2 personas, desde Mendoza, $800.000.
- **Investigador:** memoria ChromaDB primero, Tavily solo si no hay datos vigentes (7 días). `langchain-tavily` (`TavilySearch`). Como máximo una búsqueda web por tema, 3 resultados por búsqueda.
- **Embeddings:** en real, el ONNX por defecto de ChromaDB (sin torch); en mock y tests, `EmbeddingHash` (offline). La memoria compara contra "la consulta original + el título".
- **Analista: A + B** (`validar_viaje` + `calcular_presupuesto`), sin LLM adentro. Con tarjeta se usa el dólar **tarjeta**; en efectivo, el **MEP** (o, si no está, el oficial). La unidad del alojamiento es obligatoria.
- **Contexto acotado:** los especialistas reciben solo la instrucción + `DATOS DEL PEDIDO` (+ el texto final de los hallazgos, en el caso del analista). Nunca la pregunta ni el historial.
- **`PedidoViaje`** lo extrae el supervisor una sola vez; no valida reglas de negocio.
- **Fuentes y avisos los pone el código.** Solo se listan las fuentes citadas.
- **Frenos:** `MAX_PASOS_SUPERVISOR` = 6 (la última decisión es FINALIZAR forzado), `RECURSION_LIMIT` = 20, `RECURSION_LIMIT_ESPECIALISTA` = 12.
- **Reglas duras:** no se finaliza con presupuesto sin el analista; el investigador trabaja como máximo 2 veces; el analista no trabaja sin datos; no se finaliza sin aportes.
- **`MAX_TOKENS_RESPUESTA` = 8000** (con 2500 se cortaban las respuestas).
- **Se mantiene `create_react_agent`** porque lo nombra la consigna.
- **Sin MCP:** `EXTRA_MCP.md` no es parte de esta entrega (se movió a `../EXTRA_MCP.md`).

## 6. Lo que mostraron las primeras corridas reales (para el README)

| Traza | Qué pasó | Qué se cambió |
|---|---|---|
| `0_primera_corrida_real_antes_del_ajuste` | El supervisor pidió 3 veces "precios exactos" y finalizó sin el analista; 75 fuentes. | Rúbrica de precios de referencia, reglas duras, solo fuentes citadas, menos búsquedas. |
| `0b_segunda_corrida_real_tope_de_tokens` | Las llamadas a `calcular_presupuesto` se cortaron (`kwargs {}`) y la síntesis quedó trunca. El chequeo por código lo detectó. | Tope de 8000 tokens, cálculos de a uno, síntesis breve y aviso si se corta. |
| `1_flujo_completo_real` | ✅ Funciona de punta a punta. | — |

## 7. Próximo paso exacto

Abrir `traces/5_memoria_real_*.log` y revisar las 3 búsquedas en Tavily y la vuelta al investigador después del analista. Decidir con Anuk si se ajusta o se documenta tal cual. Después, armar `demo.ipynb`.
