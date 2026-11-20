"""Provider endpoints, key resolution and per-provider budget defaults.

The agent loop talks to any OpenAI-style ``/chat/completions`` endpoint. This
module is the single place that knows which URL that is, which environment
variable holds the key, and what the token budget looks like for each provider.

Nothing here ever reads a key into a log, a report or an exception: keys are
looked up on demand and only ever handed to an ``Authorization`` header.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping

#: The providers TaskGate speaks to out of the box.
PROVIDERS: tuple[str, ...] = ("openrouter", "groq", "openai-compatible")

#: Default chat-completions endpoint, before any ``--base-url`` override.
_ENDPOINTS: dict[str, str] = {
    "openrouter": "https://openrouter.ai/api/v1/chat/completions",
    "groq": "https://api.groq.com/openai/v1/chat/completions",
}

#: Environment variables that may hold the key, in priority order.
_KEY_ENVS: dict[str, tuple[str, ...]] = {
    "openrouter": ("OPENROUTER_API_KEY", "TASKGATE_OPENROUTER_API_KEY"),
    "groq": ("GROQ_API_KEY", "TASKGATE_GROQ_API_KEY"),
    "openai-compatible": ("OPENAI_API_KEY", "TASKGATE_API_KEY"),
}

#: Human-readable name used in error messages.
_DISPLAY: dict[str, str] = {
    "openrouter": "OpenRouter",
    "groq": "Groq",
    "openai-compatible": "the OpenAI-compatible endpoint",
}


@dataclass(frozen=True)
class ProviderDefaults:
    """Provider-specific knobs that keep a free key inside its budget."""

    #: Tool output is truncated before it goes back to the model.
    max_tool_output_chars: int
    #: Default ceiling on agent turns.
    max_turns: int
    #: Tokens-per-minute budget for the pacer; None disables pacing.
    tpm: int | None
    #: Chat-completions endpoint when no ``--base-url`` is given; None means required.
    endpoint: str | None


_DEFAULTS: dict[str, ProviderDefaults] = {
    # A paid/aggregator endpoint: generous output, no pacing by default.
    "openrouter": ProviderDefaults(4000, 30, None, _ENDPOINTS["openrouter"]),
    # Groq's free tier binds on tokens per minute, not requests: 8K TPM, 30 RPM,
    # 1K RPD, 200K TPD. Small tool output and few turns keep one rollout cheap.
    "groq": ProviderDefaults(1500, 12, 8000, _ENDPOINTS["groq"]),
    # Anything OpenAI-shaped: the caller must pass --base-url.
    "openai-compatible": ProviderDefaults(4000, 30, None, None),
}


def normalise_provider(name: str | None) -> str:
    """Validate and canonicalise a provider name."""
    value = (name or "openrouter").strip().lower()
    if value not in PROVIDERS:
        raise ValueError(
            f"unknown provider {name!r}; choose one of {', '.join(PROVIDERS)}"
        )
    return value


def provider_display(provider: str) -> str:
    return _DISPLAY.get(provider, provider)


def provider_defaults(provider: str) -> ProviderDefaults:
    return _DEFAULTS[normalise_provider(provider)]


def resolve_endpoint(provider: str, base_url: str | None = None) -> str:
    """The chat-completions URL for a provider, honouring ``--base-url``.

    A bare host is completed with ``/chat/completions``; a URL that already
    names the route is used verbatim.
    """
    provider = normalise_provider(provider)
    if base_url:
        url = base_url.strip().rstrip("/")
        if not url.endswith("/chat/completions"):
            url = f"{url}/chat/completions"
        return url
    endpoint = _DEFAULTS[provider].endpoint
    if endpoint is None:
        raise ValueError(
            f"provider {provider!r} needs --base-url "
            "(e.g. --base-url https://host/v1)"
        )
    return endpoint


def api_key_for(provider: str, env: Mapping[str, str] | None = None) -> str | None:
    """Look up the provider's API key. Missing key is ``None``, never an error."""
    provider = normalise_provider(provider)
    env = os.environ if env is None else env
    for name in _KEY_ENVS[provider]:
        value = env.get(name)
        if value:
            return value
    return None


def key_env_names(provider: str) -> tuple[str, ...]:
    """The variable name(s) to mention when the key is missing (never a URL)."""
    return _KEY_ENVS[normalise_provider(provider)]


def apply_defaults(config) -> None:
    """Fill any unset provider-derived fields on a :class:`RunConfig` in place."""
    provider = normalise_provider(getattr(config, "provider", None))
    defaults = _DEFAULTS[provider]
    if getattr(config, "max_tool_output_chars", None) is None:
        config.max_tool_output_chars = defaults.max_tool_output_chars
    if getattr(config, "tpm", None) is None:
        config.tpm = defaults.tpm
