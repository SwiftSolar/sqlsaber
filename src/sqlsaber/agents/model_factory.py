"""Minimal pydantic-ai model construction helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

from pydantic_ai.models import Model
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.providers import Provider
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.providers.openai import OpenAIProvider

from sqlsaber.config import providers
from sqlsaber.config.settings import ThinkingLevel

if TYPE_CHECKING:
    from pydantic_ai.providers.openai_codex import OpenAICodexCredentialSource
    from sqlsaber.config.openai_codex import PreflightOpenAICodexCredentialSource

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")

type UnifiedEffort = Literal["minimal", "low", "medium", "high", "xhigh"]

UNIFIED_EFFORT_MAP: dict[ThinkingLevel, UnifiedEffort] = {
    ThinkingLevel.MINIMAL: "minimal",
    ThinkingLevel.LOW: "low",
    ThinkingLevel.MEDIUM: "medium",
    ThinkingLevel.HIGH: "high",
    ThinkingLevel.MAXIMUM: "xhigh",
}


class ModelAuth(Protocol):
    """Authentication operations needed during model resolution."""

    def get_api_key(self, model_name: str) -> str | None: ...
    def validate(self, model_name: str) -> None: ...


@dataclass(frozen=True, slots=True)
class ResolvedModel:
    """A model plus the provider and API key inherited by child agents."""

    model_name: str
    model: Model | str
    provider: str
    api_key: str | None


def build_model(
    full_model_str: str,
    api_key: str | None,
    *,
    codex_credential_source: OpenAICodexCredentialSource | None = None,
) -> Model | str:
    """Build a provider model only when explicit credentials require it.

    OpenAI intentionally uses the Responses API model. Without an explicit key,
    returning a provider-prefixed string lets pydantic-ai perform normal discovery.
    """
    provider = providers.provider_from_model(full_model_str)
    provider_prefix, separator, model_name = full_model_str.partition(":")
    if not separator:
        model_name = full_model_str

    # Normalize SQLSaber's provider aliases before pydantic-ai sees the string.
    normalized_model_str = full_model_str
    if provider is not None and provider_prefix != provider:
        normalized_model_str = f"{provider}:{model_name}"

    if provider == "openai-codex":
        if api_key:
            raise ValueError("OpenAI Codex subscription models do not accept API keys.")
        if codex_credential_source is None:
            raise ValueError("An explicit OpenAI Codex credential source is required.")
        from pydantic_ai.models.openai_codex import OpenAICodexModel
        from pydantic_ai.providers.openai_codex import OpenAICodexProvider

        return OpenAICodexModel(
            model_name,
            provider=OpenAICodexProvider(
                credential_source=codex_credential_source,
            ),
        )

    if not api_key:
        return normalized_model_str
    if provider == "anthropic":
        return AnthropicModel(
            model_name,
            provider=AnthropicProvider(api_key=api_key),
        )
    if provider == "google":
        return GoogleModel(model_name, provider=GoogleProvider(api_key=api_key))
    if provider == "openai":
        return OpenAIResponsesModel(
            model_name,
            provider=OpenAIProvider(api_key=api_key),
        )
    if provider == "xai":
        from pydantic_ai.models.xai import XaiModel
        from pydantic_ai.providers.xai import XaiProvider

        return XaiModel(model_name, provider=XaiProvider(api_key=api_key))
    if provider is not None:
        # Other providers: construct through pydantic-ai inference with the
        # explicit key, so credentials never travel via os.environ.
        from pydantic_ai.models import infer_model
        from pydantic_ai.providers import infer_provider_class

        def provider_factory(name: str) -> Provider[Any]:
            # Every API-key provider in sqlsaber's registry accepts `api_key`.
            provider_cls = cast(Any, infer_provider_class(name))
            return provider_cls(api_key=api_key)

        try:
            return infer_model(normalized_model_str, provider_factory=provider_factory)
        except ValueError:
            pass
    return normalized_model_str


def provider_for_model(full_model_str: str) -> str:
    """Return the canonical provider key, or the raw prefix when unknown."""
    provider = providers.provider_from_model(full_model_str)
    if provider is None:
        provider = full_model_str.partition(":")[0].strip().lower()
    return provider


def resolve_injected_model(model: Model) -> ResolvedModel:
    """Describe a host-constructed model without touching credentials.

    Args:
        model: A pydantic-ai model built (with its provider) by the host.

    Returns:
        A ResolvedModel named ``<provider>:<model_name>``; ``api_key`` is ``None``
        because credentials stay inside the model's provider.
    """
    provider = providers.canonical(model.system) or model.system
    return ResolvedModel(
        model_name=f"{provider}:{model.model_name}",
        model=model,
        provider=provider,
        api_key=None,
    )


def resolve_model(
    auth: ModelAuth,
    full_model_str: str,
    *,
    api_key_override: str | None = None,
    codex_credential_source: PreflightOpenAICodexCredentialSource | None = None,
) -> ResolvedModel:
    """Resolve authentication and construct one Pydantic AI model."""

    provider = provider_for_model(full_model_str)

    if provider == "openai-codex":
        if api_key_override:
            raise ValueError("OpenAI Codex subscription models do not accept API keys.")
        if codex_credential_source is None:
            from sqlsaber.config.openai_codex import OpenAICodexCredentialStore

            codex_credential_source = OpenAICodexCredentialStore()
        codex_credential_source.preflight()
        return ResolvedModel(
            model_name=full_model_str,
            model=build_model(
                full_model_str,
                None,
                codex_credential_source=codex_credential_source,
            ),
            provider=provider,
            api_key=None,
        )

    api_key = api_key_override or None
    if api_key is None:
        auth.validate(full_model_str)
        api_key = auth.get_api_key(full_model_str)
    return ResolvedModel(
        model_name=full_model_str,
        model=build_model(full_model_str, api_key),
        provider=provider,
        api_key=api_key,
    )
