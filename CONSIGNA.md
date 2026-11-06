# Pre-entrega 6: Orquestador multi-agente especializado

> Curso AI Engineering · Coderhouse · Módulo 6 (Sistemas multi-agente: colaboración y especialización)
> Entrega: se habilita el 5/11/2026 después de las 21:00. Aprueba con 70/100.

## ¿Qué debes construir?

Debes entregar un prototipo funcional de un **Orquestador Multi-Agente de Análisis e Investigación**. El sistema debe ser capaz de procesar una solicitud de usuario que requiera **al menos dos dominios de especialización distintos** y una **fase de síntesis final**.

## Requerimientos técnicos

- **Topología Jerárquica:** un nodo Supervisor que actúa como router inteligente y controlador de flujo.
- **Mínimo de 2 Agentes Especialistas:**
  - **Agente de Búsqueda/Investigación:** con herramientas para consultar fuentes externas (puedes usar Tavily o una búsqueda simulada sobre tu Vector DB de pre-entregas anteriores).
  - **Agente de Análisis/Cómputo:** especializado en procesar los datos obtenidos (análisis de sentimiento, cálculos matemáticos o validación de esquemas).
- **Estado Compartido Estructurado:** un esquema de State en LangGraph que permita rastrear qué agente ha contribuido con qué información, evitando la pérdida de contexto en la comunicación asíncrona.
- **Flujo de Supervisión:** el supervisor debe decidir si la tarea está completa o si algún especialista debe refinar su output antes de dar la respuesta final.

## Pasos sugeridos

1. **Define tu Estado:** diseña una clase `TypedDict` que herede de `MessagesState`. Piensa si necesitas campos extra como `next_agent` o `task_completed`.
2. **Crea los Agentes Especialistas:** define funciones que utilicen `create_react_agent` de LangGraph o prompts específicos para cada rol. Asegúrate de que cada uno tenga herramientas acotadas.
3. **Implementa el Supervisor:** este es el cerebro. Su prompt debe ser claro: *"Dada la conversación actual, ¿quién debe intervenir ahora o es momento de finalizar?"*. Debe mapear sus respuestas a los nombres de los nodos del grafo.
4. **Construye el Grafo:** une los puntos. Usa `add_node` para cada agente y para el supervisor. Define las Conditional Edges que conecten al supervisor con los especialistas.
5. **Prueba la Interacción:** lanza una consulta que obligue al supervisor a enviar la tarea al Investigador, recibir el dato, enviarlo al Analista y finalmente cerrar la conversación.

## Errores comunes a evitar

- **El "Supervisor Infinito":** no definir una condición de parada clara, haciendo que el supervisor y los agentes entren en un bucle de correcciones eternas. *Tip:* implementa un contador de pasos o un criterio de "Suficiencia" estricto.
- **Contaminación de Contexto:** pasar todo el historial a todos los agentes en cada turno. Para sistemas escalables, a veces es mejor que el especialista solo reciba la instrucción específica y el contexto necesario, no toda la metadata del sistema.

## 📦 Qué entregás y en qué formato

- **Tipo:** 💻 Código — un repositorio de GitHub.
- **Artefacto concreto:** repo con `state.py`, la carpeta `agents/` (investigación + análisis), el grafo con nodo Supervisor, y un `README.md` con el diagrama Mermaid del grafo. Sumá un video corto o notebook que demuestre el flujo de delegación.
- **Qué NO hace falta:** el video/notebook es una demo del flujo, no un informe escrito.

### Entregable

Repositorio de GitHub con el código del orquestador, un archivo README explicando la topología elegida y un video corto o notebook demostrando el flujo de delegación.

## Instrucciones de la práctica

1. **Estructura del repositorio:**
   - Crea un archivo `state.py` para definir el esquema de datos compartido.
   - Crea un directorio `agents/` con los especialistas (ej. `research_agent.py`, `analyst_agent.py`).
   - Define el grafo principal en `main.py` o `graph.py`.
2. **Implementación del grafo:**
   - Utiliza `StateGraph` de LangGraph.
   - Define al menos un nodo de "Supervisor" que decida dinámicamente el flujo usando `Literal` en el retorno para las aristas condicionales.
3. **Herramientas (Tools):**
   - Los agentes deben tener al menos una herramienta funcional (puedes usar `TavilySearchResults` o una función propia que consulte una base de datos).
4. **Validación:**
   - Implementa un nodo de Validation o asegura que el Supervisor tenga una rúbrica personalizada en su prompt para validar los resultados de los especialistas antes de dar el END.
5. **Documentación:**
   - Tu README debe incluir un diagrama del grafo (puedes generarlo con `graph.get_graph().draw_mermaid_png()`).
   - Explica brevemente por qué elegiste esa topología y cómo manejas los posibles conflictos entre agentes.

## Criterios de evaluación (aprueba con 70/100)

| Criterio | Peso | Qué se evalúa |
|---|---|---|
| Documentación y Visualización | 20 % | Claridad en la explicación de la topología y representación visual del grafo en el README. |
| Arquitectura y Flujo del Grafo (Supervisor) | 35 % | Diseño del StateGraph y la implementación del nodo Supervisor para la delegación dinámica. |
| Especialización de Agentes y Herramientas | 25 % | Capacidad de los agentes para ejecutar tareas específicas mediante el uso de herramientas funcionales. |
| Gestión de Estado y Validación | 20 % | Uso de `state.py` para la coherencia de datos y presencia de mecanismos de validación de resultados. |
