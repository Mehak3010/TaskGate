"""Provider config: endpoints, key lookup, defaults and the free-model cost rule.

Everything here is offline: no request leaves the process.
"""

from __future__ import annotations

import pytest

from taskgate.models import RunConfig
from taskgate.runner.pricing import estimate_cost_usd
from taskgate.runner.providers import (
    PROVIDERS,
    api_key_for,
    apply_defaults,
    normalise_provider,
    provider_defaults,
    resolve_endpoint,
)


def test_providers_cover_the_three_shapes():
    assert set(PROVIDERS) == {"openrouter", "groq", "openai-compatible"}


def test_groq_endpoint_is_the_chat_completions_route():
    assert resolve_endpoint("groq") == "https://api.groq.com/openai/v1/chat/completions"


def test_base_url_completes_a_bare_host():
    assert (
        resolve_endpoint("openai-compatible", "https://example.test/v1")
        == "https://example.test/v1/chat/completions"
    )


def test_base_url_used_verbatim_when_it_already_names_the_route():
    url = "https://example.test/v1/chat/completions"
    assert resolve_endpoint("openai-compatible", url + "/") == url


def test_openai_compatible_requires_a_base_url():
    with pytest.raises(ValueError, match="--base-url"):
        resolve_endpoint("openai-compatible", None)


def test_unknown_provider_is_rejected():
    with pytest.raises(ValueError, match="unknown provider"):
        normalise_provider("nope")


def test_api_key_lookup_uses_the_provider_variable():
    env = {"OPENROUTER_API_KEY": "sk-or-x", "GROQ_API_KEY": "gsk-x"}
    assert api_key_for("groq", env) == "gsk-x"
    assert api_key_for("openrouter", env) == "sk-or-x"


def test_missing_key_is_none_not_an_error():
    assert api_key_for("groq", {}) is None


def test_groq_defaults_bound_the_token_budget():
    defaults = provider_defaults("groq")
    assert defaults.max_tool_output_chars == 1500
    assert defaults.max_turns == 12
    assert defaults.tpm == 8000


def test_apply_defaults_fills_only_unset_fields():
    config = RunConfig(provider="groq")
    apply_defaults(config)
    assert config.max_tool_output_chars == 1500
    assert config.tpm == 8000

    config = RunConfig(provider="groq", max_tool_output_chars=99, tpm=500)
    apply_defaults(config)
    assert config.max_tool_output_chars == 99
    assert config.tpm == 500


def test_free_provider_costs_zero():
    assert estimate_cost_usd("openai/gpt-oss-120b", 10_000, 5_000, provider="groq") == 0.0


def test_paid_provider_still_costs_something():
    assert estimate_cost_usd("openai/gpt-4o-mini", 1_000_000, 0, provider="openrouter") > 0.0
