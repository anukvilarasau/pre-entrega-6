# PROGRESO — Pre-entrega 6 (Asesor de viajes multi-agente)

> Última actualización: 5/11/2026, 21:30. Estado: **código completo y probado en mock; primera corrida real OK.**
> Faltan: el resto de las corridas reales, el notebook, el README, la revisión final y la subida a GitHub.

## 1. Qué está terminado

| Pieza | Archivo | Estado |
|---|---|---|
| Configuración (claves, rutas, 3 frenos, modo mock) | `config.py`, `.env.example`, `.gitignore` | ✅ |
| Estado `OrquestadorState(MessagesState)` + Pydantic (`Fuente`, `Aporte`, `LlamadaHerramienta`, `PedidoViaje`, `DecisionSupervisor`, `DecisionRegistrada`) + `validar_actualizacion` (cada nodo escribe solo su parte) | `state.py` | ✅ |
| Fábrica de LLM: Claude real o `LLMMock` (políticas por rol) | `llm.py`, `agents/mock.py` | ✅ |
| Memoria ChromaDB (filtros: destino/tema + 7 días + distancia ≤ 0,55, calibrada con el modelo real) | `agents/memory.py` | ✅ |
| Herramientas del investigador: `buscar_en_memoria` + `buscar_en_web` (Tavily, con guarda para no repetir búsquedas) | `agents/research_tools.py` | ✅ |
| Investigador (`create_react_agent`) + chequeo por código | `agents/research_agent.py`, `agents/common.py` | ✅ |
| Herramientas del analista: `validar_viaje` + `calcular_presupuesto` (dólar tarjeta/MEP/oficial con fuente; alojamiento por habitación o por persona) | `agents/analyst_tools.py` | ✅ |
| Analista (`create_react_agent`) + chequeo por código | `agents/analyst_agent.py` | ✅ |
| Supervisor: extrae el pedido, salida estructurada con `Literal`, rúbrica R1–R6, 4 reglas duras, freno de pasos, `ruta_supervisor` | `agents/supervisor.py` | ✅ |
| Síntesis (fuentes y avisos armados por código) | `agents/synthesizer.py` | ✅ |
| Grafo (`StateGraph` + `add_conditional_edges`) y `consultar()` | `graph.py` | ✅ |
| Trazas `.json` + `.log` con métricas | `tracing.py` | ✅ |
| CLI: `--mock`, `--chat`, `--max-pasos`, `--fallar-web`, `--mermaid`, `-v` | `main.py` | ✅ |
| Tests: **117 pasan en ~8 s, sin claves** (incluye los 8 escenarios de punta a punta en mock) | `tests/` | ✅ |
| Corrida real del escenario 1 (pregunta de demo) | `traces/1_flujo_completo_real_*` | ✅ |

Para retomar: `.venv\Scripts\python.exe -m pytest -q` (tiene que dar 117 passed) y
`.venv\Scripts\python.exe main.py --mock "Quiero ir 7 días a Bariloche..."`.

## 2. Los 3 detalles que pediste

| Detalle | Estado | Verificación |
|---|---|---|
| Actividades repetidas entre días | ✅ en mock · ⚠️ parcial en vivo | Mock: los días que sobran son "día libre", con test (`test_escenario_1_flujo_completo`). El prompt de Claude ya lo prohíbe, pero en la corrida real el día 3 ("Circuito Chico y Cerro Catedral") repite el Cerro Catedral del día 2 (la respuesta lo advierte, no lo arregla). **Pendiente:** un chequeo por código en la síntesis que detecte planes repetidos. |
| "Búsquedas web simuladas" en mock | ✅ | `Traza.etiqueta_busquedas()`: en mock dice "simuladas"; en real, "reales (Tavily)". |
| Plurales ("2 personas", "1 habitación") | ✅ | Función `cantidad()` en `state.py`; test que verifica que no aparezca "(s)". |

## 3. Qué falta, en orden

1. **Chequeo de actividades repetidas en la síntesis real** (ver la tabla de arriba).
2. **Script `escenarios.py`** que corra los 8 escenarios y genere la tabla de resultados (ruta, pasos, validación, memoria/web) para el README. En mock y en real.
3. **Corridas reales** de los escenarios 2, 3, 4, 5, 6 y 8 (el 7 es solo mock), con traza:
   - el 5 (memoria) tiene que correr **dentro de los 7 días** de la corrida 1, con la misma pregunta: la memoria real ya tiene esos datos en `data/memoria_chroma/`;
   - el 6 (refinamiento) puede no salir en vivo: si no sale, el README lo aclara y la evidencia es la de mock.
4. **Corridas mock** de los 8 escenarios, con traza (evidencia offline).
5. **`demo.ipynb`** ejecutado con salidas visibles, que muestre el flujo de delegación.
6. **README.md**: Mermaid (`python main.py --mermaid`), por qué la topología jerárquica, conflictos entre agentes, qué es Tavily y por qué la web, por qué ChromaDB y no Pinecone, `TavilySearch` en lugar de `TavilySearchResults`, `create_react_agent` deprecado (se usa igual porque lo nombra la consigna), async, instalación desde cero, tabla de escenarios, tabla de criterios y "lo que aprendimos de las corridas reales" (trazas `0_` y `0b_`).
7. **`requirements.txt` final** (sumar `langchain` 1.4.7, que entra como dependencia de `langchain-tavily`) y **prueba en un venv limpio**.
8. **Revisión punto por punto** contra `CONSIGNA.md`, los 4 criterios y `MEJORAS_TICHER.md`.
9. **Repo público `pre-entrega-6`** en GitHub (anukvilarasau) y push. NO entregar en Coderhouse.

## 4. Decisiones que tomamos

- **Tema:** asesor de viajes. Reemplazó a httpx/Pinecone. Pregunta de demo: Bariloche, 7 días, 2 personas, desde Mendoza, $800.000.
- **Investigador:** memoria ChromaDB primero, Tavily solo si no hay datos vigentes (7 días). `langchain-tavily` (`TavilySearch`) reemplaza a `TavilySearchResults`. Como máximo una búsqueda web por tema, 3 resultados por búsqueda.
- **Embeddings:** en real, el ONNX por defecto de ChromaDB (sin torch); en mock y tests, `EmbeddingHash`, offline. La memoria compara contra "la consulta original + el título", no contra la página entera.
- **Analista: A + B** (`validar_viaje` + `calcular_presupuesto`), sin LLM adentro. Con tarjeta se usa el dólar **tarjeta**; en efectivo, el **MEP** (o, si no está, el oficial). La unidad del alojamiento es obligatoria.
- **Contexto acotado:** los especialistas reciben solo la instrucción + `DATOS DEL PEDIDO` (y el analista, además, el texto final de los hallazgos). Nunca la pregunta original ni el historial (test del SECRETO).
- **`PedidoViaje`** lo extrae el supervisor una sola vez; no valida reglas de negocio (eso es de `validar_viaje`).
- **Fuentes y avisos los pone el código**, no el LLM. Solo se listan las fuentes citadas.
- **Freno:** la decisión número `MAX_PASOS_SUPERVISOR` (6) es siempre FINALIZAR forzado. Hay además `RECURSION_LIMIT` = 20 y `RECURSION_LIMIT_ESPECIALISTA` = 12.
- **Reglas duras** (salieron de las corridas reales):
  - no se finaliza con presupuesto sin el cálculo del analista;
  - el investigador trabaja como máximo 2 veces;
  - el analista no trabaja sin datos;
  - no se finaliza sin ningún aporte.
- **`MAX_TOKENS_RESPUESTA` = 8000:** con 2500 se cortaban las llamadas a `calcular_presupuesto` y la síntesis.
- **Se mantiene `create_react_agent`** (deprecado en LangGraph 1.x) porque es el nombre literal de la consigna; el aviso queda silenciado en `pytest.ini`.

## 5. Lo que mostraron las corridas reales (para el README)

| Traza | Qué pasó | Qué se cambió |
|---|---|---|
| `0_primera_corrida_real_antes_del_ajuste` | El supervisor pidió 3 veces "precios exactos por fecha" y finalizó sin el analista. La respuesta fue "no podemos confirmar", con 75 fuentes. | Rúbrica de precios de referencia, reglas duras (R4 por código y anti-bucle), solo fuentes citadas, menos búsquedas. |
| `0b_segunda_corrida_real_tope_de_tokens` | El analista llegó a trabajar, pero sus llamadas a `calcular_presupuesto` se cortaron (`kwargs {}`) y la síntesis quedó trunca. El chequeo por código lo detectó. | Tope de 8000 tokens, cálculos de a uno, síntesis más breve y aviso si se corta. |
| `1_flujo_completo_real` | ✅ Ruta supervisor → investigador → supervisor → investigador → supervisor → analista → supervisor → síntesis. "No les alcanza" ($1,6 M a $1,9 M); entra la opción de 4 días con alojamiento por habitación. 4 decisiones, 1 refinamiento, 8 búsquedas en Tavily, 91 s. | — |

## 6. Próximo paso exacto

Agregar a `agents/synthesizer.py` un chequeo por código de actividades repetidas entre días, con su test. Después, crear `escenarios.py` y correr en real los escenarios 5 (misma pregunta de demo, para que salga de la memoria), 2, 3, 4, 6 y 8.
