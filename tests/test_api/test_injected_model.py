"""Host-constructed models and credential handling without os.environ."""

from __future__ import annotations

import os

import pytest
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from sqlsaber import SQLSaber, SQLSaberOptions
from sqlsaber.config.settings import Config


def _settings() -> Config:
    return Config.in_memory(model_name="anthropic:claude-test")


def test_in_memory_keys_never_written_to_environment(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = Config.in_memory(
        model_name="openai:gpt-test", api_keys={"openai": "secret-key"}
    )

    config.validate()

    assert config.api_key == "secret-key"
    assert "OPENAI_API_KEY" not in os.environ


@pytest.mark.asyncio
async def test_sdk_conversation_with_keys_leaves_environment_untouched(
    monkeypatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    saber = SQLSaber(
        options=SQLSaberOptions(
            database="sqlite:///:memory:",
            settings=Config.in_memory(
                model_name="openai:gpt-test", api_keys={"openai": "secret-key"}
            ),
        )
    )
    try:
        assert "OPENAI_API_KEY" not in os.environ
        assert saber.agent.agent.model.model_name == "gpt-test"
    finally:
        await saber.close()


@pytest.mark.asyncio
async def test_injected_model_is_used_for_runs() -> None:
    def respond(messages, info: AgentInfo) -> ModelResponse:
        del messages, info
        return ModelResponse(parts=[TextPart("from injected model")])

    model = FunctionModel(respond, model_name="injected")
    saber = SQLSaber(
        options=SQLSaberOptions(
            database="sqlite:///:memory:", settings=_settings(), model=model
        )
    )
    try:
        assert saber.agent.agent.model is model
        result = await saber.query("hello")
        assert str(result) == "from injected model"
    finally:
        await saber.close()


@pytest.mark.asyncio
async def test_subagents_inherit_injected_model() -> None:
    model = TestModel()
    captured = {}

    def factory(context):
        captured["resolved"] = context.resolve_subagent_model()
        return None

    saber = SQLSaber(
        options=SQLSaberOptions(
            database="sqlite:///:memory:",
            settings=_settings(),
            model=model,
            capabilities=[factory],
        )
    )
    try:
        name, resolved_model, provider = captured["resolved"]
        assert resolved_model is model
        assert provider == model.system
        assert name == f"{model.system}:{model.model_name}"
    finally:
        await saber.close()


@pytest.mark.asyncio
async def test_injected_model_excludes_model_name_and_key() -> None:
    with pytest.raises(ValueError, match="either a constructed `model`"):
        SQLSaber(
            options=SQLSaberOptions(
                database="sqlite:///:memory:",
                settings=_settings(),
                model=TestModel(),
                model_name="anthropic:claude-test",
            )
        )
