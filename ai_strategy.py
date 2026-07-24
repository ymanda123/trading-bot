"""LLM-driven trading signal: Claude decides buy/sell/hold from recent price
action, replacing strategy.py's deterministic MA-crossover / support-resistance
rules for trending and range-bound regimes.

This does NOT touch risk management. The high-volatility standby rule still
overrides Claude — no new entries during a volatility spike, no matter what
the model says. And Claude only ever returns a direction; position sizing,
stop distance, the circuit breaker, and the cool-off are still entirely
risk_manager.py's job, unchanged.
"""

import json
import logging

import anthropic

from config import Config
from strategy import Regime, Signal, StrategyDecision, classify_regime

logger = logging.getLogger("trading-bot")

_client = None

_SIGNAL_SCHEMA = {
    "type": "object",
    "properties": {
        "signal": {"type": "string", "enum": ["buy", "sell", "hold"]},
        "reasoning": {"type": "string", "description": "One short sentence explaining the call."},
    },
    "required": ["signal", "reasoning"],
    "additionalProperties": False,
}


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    return _client


def _format_candles(df, n=30) -> str:
    tail = df.tail(n)
    return "\n".join(
        f"{row.timestamp} O:{row.open:.2f} H:{row.high:.2f} L:{row.low:.2f} C:{row.close:.2f}"
        for row in tail.itertuples()
    )


def ai_signal(df, regime, atr, config=Config):
    """Ask Claude for a buy/sell/hold call. Fails safe to HOLD on any error,
    refusal, or malformed response — a broken API call should never be able
    to force a trade."""
    price = df["close"].iloc[-1]
    prompt = (
        f"Symbol: {config.SYMBOL} ({config.TIMEFRAME} candles)\n"
        f"Current price: {price:.2f}\n"
        f"14-period ATR: {atr:.2f} ({atr / price * 100:.2f}% of price)\n"
        f"Detected volatility regime: {regime.value}\n\n"
        f"Last {min(30, len(df))} candles (oldest to newest):\n{_format_candles(df)}\n\n"
        "This is a paper-trading simulation on Binance Testnet — no real money "
        "moves on your call. Decide buy, sell, or hold based purely on the "
        "price action above."
    )

    try:
        response = _get_client().messages.create(
            model="claude-opus-4-8",
            max_tokens=1024,
            thinking={"type": "adaptive"},
            output_config={
                "effort": "low",
                "format": {"type": "json_schema", "schema": _SIGNAL_SCHEMA},
            },
            messages=[{"role": "user", "content": prompt}],
        )
        if response.stop_reason == "refusal":
            logger.warning("AI signal refused; defaulting to hold")
            return Signal.HOLD, "Model declined to answer; defaulting to hold."

        text = next(block.text for block in response.content if block.type == "text")
        data = json.loads(text)
        return Signal(data["signal"]), data["reasoning"]
    except Exception:
        logger.exception("AI signal call failed; defaulting to hold")
        return Signal.HOLD, "AI call failed; defaulting to hold (fail-safe)."


def decide_with_ai(df, config=Config) -> StrategyDecision:
    """Same return shape as strategy.decide(), but the entry signal comes
    from Claude instead of the fixed MA-crossover / support-resistance rules.
    """
    regime, atr = classify_regime(df, config)

    if regime is Regime.HIGH_VOL_STANDBY:
        return StrategyDecision(
            regime, Signal.HOLD, atr,
            "High volatility spike detected; standing by regardless of AI (safety rule).",
        )

    signal, reasoning = ai_signal(df, regime, atr, config)
    return StrategyDecision(regime, signal, atr, f"Claude: {reasoning}")
