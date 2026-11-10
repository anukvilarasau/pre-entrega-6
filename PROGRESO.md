# PROGRESO — Pre-entrega 6 (Asesor de viajes multi-agente)

> **Estado: TERMINADA y subida a GitHub. NO se entregó en la plataforma de Coderhouse** (eso lo hace Anuk).
> Última actualización: 10/11/2026.

## Resumen final

- **Repo público:** https://github.com/anukvilarasau/pre-entrega-6
- **Tests:** 128, todos pasan en ~10 s, sin claves (`python -m pytest -q`).
- **Instalación desde cero:** probada en un venv nuevo con Python 3.12.10. `pip install -r requirements.txt` sin errores, `pip check` sin conflictos y los 128 tests pasan.
- **Escenarios:** los 8 en mock y los 7 posibles en vivo (el 7, Tavily caído, es solo mock), con sus trazas en `traces/` y la tabla en `traces/resultados.md` y en el README (§9).
- **`demo.ipynb`:** ejecutado con salidas (8 celdas de código, 0 errores). Incluye la pregunta de demo en mock y en vivo, la memoria y el freno.
- **README:** diagrama Mermaid, por qué la topología jerárquica, conflictos entre agentes, contaminación de contexto, Tavily, ChromaDB vs. Pinecone, `langchain-tavily` vs. `TavilySearchResults`, `create_react_agent`, escenarios (vivo y mock), lo aprendido de las corridas reales, tabla de criterios e instalación paso a paso.

## Resultados de los escenarios (código final, 10/11/2026)

| # | Escenario | En vivo (Claude + Tavily) | Mock |
|---|---|---|---|
| 1 | Flujo completo (demo) | ✅ sup → inv → sup → ana → sup → síntesis (3 decisiones). **No alcanza: $826.720 (faltan $26.720).** 6 búsquedas en Tavily, 18 fuentes. | ✅ misma ruta; entra ($785.100) |
| 2 | Solo clima | ✅ sin analista (2 decisiones) | ✅ |
| 3 | No alcanza ($400.000) | ✅ no entra; todo de memoria | ✅ con alternativa |
| 4 | Fechas inválidas | ✅ rechazada; pide corregir; sin itinerario | ✅ |
| 5 | Memoria | ✅ **17 de memoria · 0 búsquedas en Tavily** | ✅ |
| 6 | Refinamiento | ✅ el supervisor rechazó el aporte (faltaban precios de vuelos) y pidió rehacer; la 2.ª ronda no sumó datos (ver "Para revisar") | ✅ rechazo por búsqueda sin precios + nueva búsqueda |
| 7 | Tavily caído | — (solo mock) | ✅ termina, no inventa presupuesto |
| 8 | Freno (`max_pasos=2`) | ✅ FINALIZAR forzado + aviso de información parcial | ✅ |

## Qué se hizo en esta última etapa

1. **Historial de git reescrito:** `EXTRA_MCP.md` no aparece en ningún commit (verificado; respaldos y objetos viejos borrados). El archivo sigue en `../EXTRA_MCP.md`.
2. **Arreglos del escenario 5** (con tests):
   - `forzar` bloqueado por código si la memoria tiene precios vigentes;
   - **cobertura por tema** calculada por código (`CoberturaTema`) que el supervisor ve como tabla;
   - campo estructurado `DecisionSupervisor.temas` → `temas_pedidos`; el chequeo del investigador usa ese campo.
3. **Arreglo del escenario 4** (salió en la corrida en vivo): el supervisor le pidió al analista "suponer" otra fecha y la métrica marcó la validación como aprobada.
   - Ahora cuenta la **validación del pedido real** (`validacion_del_pedido`).
   - Hay una regla dura nueva: "pedido inválido → FINALIZAR".
   - El analista no puede inventar fechas.
4. Todos los escenarios se volvieron a correr con el código final (en vivo y en mock). Las trazas de versiones anteriores quedaron en `traces/anteriores/`.
5. `demo.ipynb`, README completo, `requirements.txt` final (se fijó `langchain==1.4.7`) y prueba en venv limpio.
6. Revisión contra `CONSIGNA.md` (21 puntos verificados en el código) y contra `MEJORAS_TICHER.md` (tabla en el README, §11).

## Decisiones tomadas sin consultar (según la consigna y MEJORAS_TICHER)

- **Frase literal de la consigna** en el mensaje del supervisor: "Dada la conversación actual, ¿quién debe intervenir ahora o es momento de finalizar?".
- **Diagrama:** se usa `draw_mermaid()` (texto Mermaid que GitHub dibuja solo). `draw_mermaid_png()` quedó **bloqueado por permisos**, porque manda el diagrama a un servicio público externo (mermaid.ink). No hace falta para la consigna.
- **Trazas anteriores:** se conservan en `traces/anteriores/` como evidencia de cada arreglo (el README las cita en §10).
- **Repetí todas las corridas reales** después del último arreglo, para que las trazas del repo correspondan exactamente al código final.

## Para revisar vos (Anuk)

1. **Veredicto cerca del límite:** en el escenario 1 en vivo la demo dio "no alcanza por $26.720"; en la corrida en vivo del notebook, con la misma memoria, dio "entra". El LLM eligió precios de referencia distintos y el total está muy cerca de $800.000. Está explicado en el README (§14). Si querés un veredicto estable, se puede probar la demo con otro presupuesto.
2. **Escenario 6 en vivo:** el refinamiento ocurrió (el supervisor rechazó por falta de precios de vuelos), pero el bloqueo de `forzar` trabaja **por tema**. Como ya había precios de micro en "pasajes", no dejó buscar vuelos en la web. Está documentado como limitación; la mejora sería separar subtemas (micro y avión).
3. **Memoria real:** `data/memoria_chroma/` (no se sube) tiene los datos del 10/11/2026. Vencen el 17/11; después, la demo en vivo vuelve a buscar en Tavily.
4. **Costo:** una pregunta completa en vivo usa del orden de 100.000 a 200.000 tokens de `claude-haiku-5-5`.
5. **Entrega en Coderhouse:** pendiente, la hacés vos.

## Cómo retomarlo

```powershell
.venv\Scripts\python.exe -m pytest -q                 # 128 passed
.venv\Scripts\python.exe main.py --mock --chat        # probarlo sin claves
.venv\Scripts\python.exe escenarios.py --tabla        # rearmar la tabla desde las trazas
```
