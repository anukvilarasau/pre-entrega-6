"""
llm.py — Fábrica de modelos: Claude real o un LLM simulado (modo mock).

Cada rol (supervisor, investigador, analista, sintesis) pide su modelo con
`crear_llm(rol)`. El grafo NO sabe si está hablando con Claude o con el
simulado: los dos se usan igual (ainvoke, bind_tools, with_structured_output).
Así el mismo grafo corre en vivo, en la demo offline y en los tests.

`LLMMock` es un chat model de LangChain cuyo "cerebro" es una función Python
(una "política") que mira los mensajes y decide qué responder, igual que
haría el LLM: llamar herramientas o dar la respuesta final. Las políticas de
cada rol están en agents/mock.py.
"""

import itertools
import logging
from collections.abc import Callable, Sequence
from typing import Any, Literal

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, SecretStr

import config

log = logging.getLogger(__name__)

Rol = Literal["supervisor", "investigador", "analista", "sintesis"]

# Una política recibe los mensajes (y las herramientas disponibles) y devuelve
# lo que respondería el LLM: un AIMessage (con o sin tool_calls) o, si se
# pidió salida estructurada, una instancia del esquema Pydantic.
Politica = Callable[[list[BaseMessage], list[str]], AIMessage]
PoliticaEstructurada = Callable[[list[BaseMessage], type[BaseModel]], BaseModel]

_contador_ids = itertools.count(1)


def nueva_llamada(nombre: str, argumentos: dict[str, Any]) -> dict[str, Any]:
    """Arma un tool_call con un ID único (lo usan las políticas mock)."""
    return {"name": nombre, "args": argumentos, "id": f"mock-call-{next(_contador_ids)}", "type": "tool_call"}


class LLMMock(BaseChatModel):
    """Chat model determinístico para modo mock y tests (sin red, sin claves)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    politica: Politica | None = None
    politica_estructurada: PoliticaEstructurada | None = None
    herramientas: list[str] = []
    # Registro de lo que "vio" el modelo en cada llamada (para tests de contexto).
    vistos: list[list[BaseMessage]] = []

    @property
    def _llm_type(self) -> str:
        return "llm-mock"

    def _responder(self, mensajes: list[BaseMessage]) -> ChatResult:
        if self.politica is None:
            raise RuntimeError("Este LLMMock no tiene política para respuestas de texto.")
        self.vistos.append(list(mensajes))
        respuesta = self.politica(list(mensajes), list(self.herramientas))
        return ChatResult(generations=[ChatGeneration(message=respuesta)])

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return self._responder(messages)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return self._responder(messages)

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "LLMMock":  # type: ignore[override]
        """Igual que con Claude: devuelve un modelo que "conoce" estas herramientas."""
        nombres = [t.name if isinstance(t, BaseTool) else str(getattr(t, "__name__", t)) for t in tools]
        return self.model_copy(update={"herramientas": nombres})

    def with_structured_output(  # type: ignore[override]
        self, schema: type[BaseModel], **kwargs: Any
    ) -> Runnable[LanguageModelInput, BaseModel]:
        """Salida estructurada: la política devuelve directamente el objeto Pydantic."""
        if self.politica_estructurada is None:
            raise RuntimeError("Este LLMMock no tiene política para salida estructurada.")
        politica = self.politica_estructurada
        vistos = self.vistos

        def _invocar(entrada: Any) -> BaseModel:
            mensajes = list(entrada) if isinstance(entrada, list) else entrada.to_messages()
            vistos.append(mensajes)
            # model_validate: el resultado pasa por la MISMA validación que con Claude.
            return schema.model_validate(politica(mensajes, schema).model_dump())

        return RunnableLambda(_invocar)


def guion(respuestas: list[AIMessage]) -> Politica:
    """Política "libreto": devuelve las respuestas en orden (útil en tests)."""
    pendientes = list(respuestas)

    def _politica(_mensajes: list[BaseMessage], _herramientas: list[str]) -> AIMessage:
        if not pendientes:
            raise RuntimeError("El guion del LLMMock se quedó sin respuestas.")
        return pendientes.pop(0)

    return _politica


def crear_llm_real() -> BaseChatModel:
    """Claude vía ChatAnthropic (misma configuración que la Pre-entrega 5).

    La API actual de Claude no acepta `temperature` en los modelos nuevos: se
    usa effort="low" (respuestas directas y estables) en su lugar.
    """
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(
        model_name=config.ANTHROPIC_MODEL,
        api_key=SecretStr(config.ANTHROPIC_API_KEY),
        max_tokens_to_sample=config.MAX_TOKENS_RESPUESTA,
        effort="low",
        max_retries=3,   # reintentos con backoff ante 429/5xx
        timeout=90,
        stop=None,
    )


def crear_llm(rol: Rol, modo_mock: bool = config.MODO_MOCK) -> BaseChatModel:
    """El modelo de cada rol: Claude real, o el simulado con la política de ese rol."""
    if not modo_mock:
        return crear_llm_real()
    from agents import mock  # import tardío: mock depende de módulos de agents

    politica, politica_estructurada = mock.POLITICAS[rol]
    log.debug("LLM mock para el rol %s", rol)
    return LLMMock(politica=politica, politica_estructurada=politica_estructurada)
