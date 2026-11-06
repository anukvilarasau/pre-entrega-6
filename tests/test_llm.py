"""Tests de llm.py: la fábrica elige bien y el LLMMock se comporta como un chat model."""

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import ValidationError

import config
from llm import LLMMock, crear_llm, guion, nueva_llamada
from state import DecisionSupervisor


def test_modo_real_crea_claude_sin_llamar_a_la_api(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "sk-ant-falsa")
    llm = crear_llm("investigador", modo_mock=False)
    assert isinstance(llm, ChatAnthropic)
    assert llm.model == config.ANTHROPIC_MODEL


def test_modo_mock_crea_llm_mock() -> None:
    assert isinstance(crear_llm("investigador", modo_mock=True), LLMMock)


async def test_guion_devuelve_en_orden_y_registra_lo_visto() -> None:
    llm = LLMMock(politica=guion([AIMessage(content="uno"), AIMessage(content="dos")]))
    assert (await llm.ainvoke([HumanMessage(content="hola")])).content == "uno"
    assert (await llm.ainvoke([HumanMessage(content="chau")])).content == "dos"
    assert [m[0].content for m in llm.vistos] == ["hola", "chau"]
    with pytest.raises(RuntimeError, match="sin respuestas"):
        await llm.ainvoke([HumanMessage(content="otra")])


async def test_bind_tools_conserva_politica_y_registro() -> None:
    llm = LLMMock(politica=guion([AIMessage(content="", tool_calls=[nueva_llamada("buscar_en_web", {"x": 1})])]))
    atado = llm.bind_tools([])
    respuesta = await atado.ainvoke([HumanMessage(content="hola")])
    assert respuesta.tool_calls[0]["name"] == "buscar_en_web"
    assert len(llm.vistos) == 1  # el registro es compartido con el modelo original


async def test_salida_estructurada_pasa_por_la_validacion_pydantic() -> None:
    ok = LLMMock(politica_estructurada=lambda _m, s: s.model_construct(evaluacion="ok", siguiente="FINALIZAR", instruccion=""))
    decision = await ok.with_structured_output(DecisionSupervisor).ainvoke([HumanMessage(content="?")])
    assert isinstance(decision, DecisionSupervisor) and decision.siguiente == "FINALIZAR"

    # Una decisión inválida (delega sin instrucción) se rechaza igual que con Claude.
    mala = LLMMock(politica_estructurada=lambda _m, s: s.model_construct(evaluacion="x", siguiente="analista", instruccion=""))
    with pytest.raises(ValidationError):
        await mala.with_structured_output(DecisionSupervisor).ainvoke([HumanMessage(content="?")])
