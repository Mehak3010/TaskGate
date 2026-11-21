"""Model pricing used for the budget guard and the cost column.

Prices are USD per 1M tokens and are deliberately tiny/approximate: TaskGate
uses them for a spend guard, not for invoicing. Override with --price-in
/--price-out when the real model differs.
"""

from __future__ import annotations

#: Free tiers charge nothing. A token budget still applies, but the *dollar*
#: spend is zero, so the cost column and the budget guard see $0.
FREE_PROVIDERS: frozenset[str] = frozenset({"groq"})

#: USD per 1M tokens: (input, output).
PRICING: dict[str, tuple[float, float]] = {
    "mock": (0.15, 0.60),
    "openai/gpt-4o-mini": (0.15, 0.60),
    "openai/gpt-4.1-mini": (0.40, 1.60),
    "anthropic/claude-3.5-haiku": (0.80, 4.00),
    "anthropic/claude-sonnet-4": (3.00, 15.00),
    "google/gemini-2.0-flash-001": (0.10, 0.40),
    "deepseek/deepseek-chat": (0.14, 0.28),
}
DEFAULT_PRICING = (0.50, 2.00)


def price_for(model: str, override: tuple[float, float] | None = None) -> tuple[float, float]:
    if override is not None:
        return override
    if model in PRICING:
        return PRICING[model]
    for known, price in PRICING.items():
        if known.split("/")[-1] in model:
            return price
    return DEFAULT_PRICING


def estimate_cost_usd(
    model: str,
    tokens_in: int,
    tokens_out: int,
    override: tuple[float, float] | None = None,
    *,
    provider: str | None = None,
) -> float:
    # A free-tier provider costs $0 whatever the model name is: the binding
    # limit there is tokens, not dollars, so the budget guard must not fire.
    if provider is not None and provider in FREE_PROVIDERS:
        return 0.0
    price_in, price_out = price_for(model, override)
    return round(tokens_in / 1_000_000 * price_in + tokens_out / 1_000_000 * price_out, 6)
