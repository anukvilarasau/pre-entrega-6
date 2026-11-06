# Devoluciones de Ticher (Pre-entregas 1 a 5) y cómo aplicarlas en la Pre-entrega 6

> Leídas de la plataforma de Coderhouse el 5/11/2026. Todas las pre-entregas están aprobadas.
> Notas: PE1 100 % · PE2 95 % · PE3 92 % · PE4 94 % · PE5 88 %.

## Checklist para la Pre-entrega 6

Cada punto sale de una o más devoluciones (entre paréntesis).

1. **Cumplir la consigna al pie de la letra, también en los nombres** (PE1, PE5). Si la consigna nombra una clase o un componente, usarlo literalmente:
   - El estado **hereda de `MessagesState`** (en la PE5 bajó nota por usar un TypedDict equivalente).
   - Archivos con los nombres exactos: `state.py`, `agents/research_agent.py`, `agents/analyst_agent.py`, `graph.py` o `main.py`.
   - Supervisor con `Literal` en el retorno, `StateGraph`, `add_conditional_edges`, `create_react_agent`.
   - Si se usa una variante (por ejemplo, async en lugar de sync), dejarlo explicado en el README.
2. **Tests automáticos con `pytest` y el LLM mockeado** (PE2, PE3, PE4). Que se pueda verificar el grafo, el ruteo del supervisor, el freno de pasos, la validación Pydantic del estado y las herramientas del analista **sin API key ni llamadas pagas**.
3. **Modo mock/offline** (PE2, PE4): una forma de correr la demo con respuestas prefijadas y Pinecone simulado, para que funcione en cualquier entorno (el evaluador no tiene nuestras claves).
4. **Correr todo en vivo con las claves reales y dejar la evidencia en el repo** (PE4, PE5): trazas `.json`/`.log`, salidas de ejemplo y el notebook ejecutado con sus celdas de salida visibles. Ticher no puede correr el código, así que la evidencia es lo que mira.
5. **Instalación reproducible desde cero** (PE3, PE4): `requirements.txt` con versiones fijas, probado en un entorno virtual limpio, y un **paso a paso en el README** (crear venv → instalar → crear `.env` → correr la demo → correr los tests).
6. **Probar casos límite** (PE2): pregunta que no necesita al analista, pregunta sin respuesta en la documentación, configuración inválida, herramienta que falla, y que el freno de pasos corte un bucle.
7. **Demostrar cada requisito con un escenario concreto** (PE5): por ejemplo, un caso donde el supervisor rechaza el trabajo de un especialista y le pide que lo refine, con su traza.
8. **Medir y mostrar resultados** (PE4): en el README, tabla con los escenarios corridos, la ruta que tomó el supervisor, cuántos pasos usó y si la validación aprobó.
9. **Mantener lo que ya destacó Ticher** (fortalezas repetidas): módulos separados, Pydantic estricto (`extra="forbid"`, rangos, validadores propios), resiliencia (reintentos, manejo de errores tipados), logging por etapa, `.env.example` + `.gitignore`, README prolijo.

---

## Detalle por pre-entrega

### Pre-entrega 1 · Cliente de LLM robusto y asíncrono (100 %)
- Mensaje del profesor (Federico Gravina): "¡Felicitaciones! No dejaste nada para corregir. Controlaste toda la modularización y la arquitectura de manera sobresaliente."
- **Fuertes:** arquitectura orientada a objetos con clases abstractas, factorías y decoradores; streaming async con fallback antes del primer token; validación Pydantic con rangos estrictos; errores de red y rate limit con excepciones tipadas y reintentos exponenciales.
- **Para mejorar:** el criterio pedía una clase `AsyncLLMManager` y no se nombró literalmente (se resolvió con `LLMClient` + `ResilientClient`). Lo aceptó, pero conviene usar los nombres de la consigna.

### Pre-entrega 2 · Pipeline de extracción validado (95 %)
- **Fuertes:** esquema Pydantic exhaustivo (enum, `extra='forbid'`, validador propio); cadena LCEL con prompt modular; `.with_retry()` y detección de respuestas cortadas; logging completo con evidencia; repo prolijo.
- **Para mejorar:**
  - No había tests automáticos: sumar pruebas con el LLM mockeado.
  - Agregar un modo mock/offline para correr sin API key.
  - Probar entradas límite (textos muy cortos, con HTML, con muchas tecnologías).
  - Opcional: exponer la función como endpoint async con FastAPI.

### Pre-entrega 3 · RAG local con ChromaDB (92 %)
- **Fuertes:** chunking por tokens; cadena RAG async con `asyncio.gather`; doble esquema Pydantic con fuentes validadas contra la metadata; pruebas con preguntas trampa y evidencia real; credenciales seguras.
- **Para mejorar:**
  - Tests formales con `pytest` que repliquen `probar_rag.py` y corran en CI sin API key.
  - Validar la instalación desde cero en un entorno limpio con `pip install -r requirements.txt`.

### Pre-entrega 4 · RAG escalable con Pinecone (94 %)
- **Fuertes:** Pinecone Serverless con validación de entorno; ingesta multi-formato; híbrido BM25 + semántico; evaluación con Recall@5 = 1.0 y salidas persistidas.
- **Para mejorar:**
  - Precision@5 del híbrido baja (0.36): probar pesos del ensemble o filtros por categoría.
  - Ampliar el golden set para que las métricas sean más estables.
  - Correr el pipeline completo en un entorno limpio con las claves y regenerar `resultados/`.
  - Smoke test con Pinecone mockeado para validar en CI.
  - Paso a paso en el README para replicar el índice desde cero.

### Pre-entrega 5 · Agente ReAct con memoria persistente (88 %)
- **Fuertes:** ciclo ReAct con `tools_condition`; razonamiento multi-paso demostrado con trazas; memoria que sobrevive al reinicio; README, `.env.example` y `.gitignore` prolijos.
- **Para mejorar:**
  - **Heredar el estado de `MessagesState`** en lugar de un TypedDict propio (fue "la única desviación real").
  - Documentar en el README el resultado de correr `demo.py` con las claves, porque el evaluador no pudo ejecutarlo.
  - Correr `python demo.py` localmente y subir las trazas nuevas.
  - Agregar un segundo turno con el mismo `thread_id` para reforzar la demo de memoria.
  - Opcional: probar también `SqliteSaver` síncrono (el async es válido).
