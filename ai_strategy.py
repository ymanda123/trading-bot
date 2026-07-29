"""AI-driven trading decision. Replaces strategy.py's fixed rules for the
trending / range-bound regimes with an AI-for-context + backtest-for-
execution pipeline. Three earlier design choices each turned out to quietly
suppress trades the data actually supported, and were removed in favor of
letting the evidence decide instead of an LLM's single guess:

  - Asking the LLM to also pick parameters: it reliably converged on the
    same conservative, rarely-triggering values (e.g. RSI 25/75) regardless
    of how the prompt asked it to vary them. Replaced by backtester.
    search_best_params, which grid-searches a strategy template's spec'd
    parameter range and keeps the combination with the best backtested net
    return among those that pass the gate (trades >= BACKTEST_MIN_TRADES,
    net_pnl_pct > 0) -- preferring, among passing combinations, one that
    also fires on the most recent candle.
  - Asking the LLM to self-report a buy/sell/hold call: it frequently said
    "hold" out of generic caution even when the strategy it had just
    proposed was, by its own backtested math, sitting on a real edge.
    Replaced by reading the entry directly off the validated parameter
    combination's mechanical rule on the latest candle (backtester.
    generate_signals). The AI's own read is still collected and shown in
    the reasoning for transparency, but it doesn't gate anything.
  - Restricting the backtest search to only the one strategy family the LLM
    proposed: the model anchors hard to the regime label ("range-bound"
    reliably meant it proposed a mean-reversion family, "trending" a
    trend-following one), even when a different family had a real,
    currently-firing, validated edge in the same data. Replaced by
    backtester.search_best_across_types, which grid-searches every
    template and keeps the best result across all of them.

What the AI still does, and why it's still here: it reads live news
alongside price action and produces the human-readable reasoning shown in
the dashboard ("why is the bot doing this") -- context a pure backtest grid
search has no way to produce on its own, and the whole reason this project
uses an LLM instead of pure fixed rules. It just no longer gets to
unilaterally veto a trade the evidence supports.

This does NOT touch risk management. The high-volatility standby rule still
overrides everything below — no new entries during a volatility spike, no
matter what the model or the backtest says. And once a signal is allowed
through, position sizing, stop distance, the circuit breaker, and the
cool-off are still entirely risk_manager.py's job, unchanged.

Uses Google Gemini's free-tier API (no credit card required) via its
OpenAI-compatible endpoint, configured via the GEMINI_API_KEY environment
variable. https://aistudio.google.com/apikey
"""

import json
import logging
import os
import re

from openai import OpenAI

from backtester import STRATEGY_TYPES, search_best_across_types
from config import Config
from news_feed import fetch_news, format_news_for_prompt
from strategy import Regime, Signal, StrategyDecision, classify_regime

logger = logging.getLogger("trading-bot")

_client = None

_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
_GEMINI_MODEL = "gemini-flash-lite-latest"

_STRATEGY_DESCRIPTIONS = (
    '  - "ma_crossover": trend-following -- trades a fast/slow moving-average crossover\n'
    '  - "support_resistance": mean-reversion -- buys near a rolling support floor, sells near a rolling resistance ceiling\n'
    '  - "rsi_threshold": mean-reversion -- buys when RSI is oversold, sells when RSI is overbought'
)


def _get_client():
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ["GEMINI_API_KEY"], base_url=_GEMINI_BASE_URL)
    return _client


def _format_candles(df, n=30) -> str:
    tail = df.tail(n)
    return "\n".join(
        f"{row.timestamp} O:{row.open:.2f} H:{row.high:.2f} L:{row.low:.2f} C:{row.close:.2f}"
        for row in tail.itertuples()
    )


def _extract_json(text: str) -> dict:
    """Defensive parsing in case the model wraps the JSON in prose despite
    being asked not to — pull out the first {...} block rather than
    requiring the whole response to parse as-is."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("no JSON object found in response")
    return json.loads(match.group(0))


def propose_strategy(df, regime, atr, config=Config) -> dict:
    """Ask the LLM to weigh recent price action against live news, then
    propose one of STRATEGY_TYPES that fits the current market, plus its
    own (informational-only -- see module docstring) read on which way it
    would lean. Fails safe to a hold on any error, block, or malformed
    response — a broken API call should never be able to force a trade."""
    price = df["close"].iloc[-1]
    news_items = fetch_news()
    news_block = format_news_for_prompt(news_items)

    prompt = (
        f"Symbol: {config.SYMBOL} ({config.TIMEFRAME} candles)\n"
        f"Current price: {price:.2f}\n"
        f"14-period ATR: {atr:.2f} ({atr / price * 100:.2f}% of price)\n"
        f"Detected volatility regime: {regime.value}\n\n"
        f"Last {min(30, len(df))} candles (oldest to newest):\n{_format_candles(df)}\n\n"
        f"Recent financial/crypto news headlines (CNBC, Yahoo Finance, Bloomberg, WSJ, NYT, Financial Times):\n{news_block}\n\n"
        "This is a paper-trading simulation — no real money or real orders "
        "move on your call. Considering both the price action above and the "
        "news headlines, propose ONE trading strategy from this exact set "
        f"that you believe fits the current market:\n{_STRATEGY_DESCRIPTIONS}\n\n"
        "Its exact parameters will be chosen separately by backtesting, so "
        "don't propose parameters — just the strategy family. Then state "
        "the buy/sell/hold lean you'd take right now, for context.\n\n"
        'Respond with ONLY a JSON object, no other text: '
        '{"strategy_type": "ma_crossover" | "support_resistance" | "rsi_threshold", '
        '"signal": "buy" | "sell" | "hold", '
        '"news_summary": "one short sentence on which headline (if any) influenced this, or \'no notable news\'", '
        '"reasoning": "one short sentence on why this strategy family fits right now"}'
    )

    try:
        response = _get_client().chat.completions.create(
            model=_GEMINI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        data = _extract_json(response.choices[0].message.content)
        strategy_type = data["strategy_type"]
        if strategy_type not in STRATEGY_TYPES:
            raise ValueError(f"unknown strategy_type {strategy_type!r}")
        return {
            "strategy_type": strategy_type,
            "signal": Signal(data["signal"]),
            "news_summary": data.get("news_summary", ""),
            "reasoning": data["reasoning"],
        }
    except Exception as exc:
        logger.exception("AI strategy proposal failed; defaulting to hold")
        return {
            "strategy_type": None,
            "signal": Signal.HOLD,
            "news_summary": "",
            "reasoning": f"AI call failed ({type(exc).__name__}: {exc}); defaulting to hold (fail-safe).",
        }


def decide_with_ai(df, config=Config) -> StrategyDecision:
    """Same return shape as strategy.decide(), but the reasoning comes from
    an LLM reading news + price action, while the strategy family,
    parameters, and entry signal all come from a backtest-validated grid
    search across every template (see module docstring for why)."""
    regime, atr = classify_regime(df, config)

    if regime is Regime.HIGH_VOL_STANDBY:
        return StrategyDecision(
            regime, Signal.HOLD, atr,
            "High volatility spike detected; standing by regardless of AI (safety rule).",
        )

    proposal = propose_strategy(df, regime, atr, config)

    if proposal["strategy_type"] is None:
        return StrategyDecision(regime, Signal.HOLD, atr, f"AI: {proposal['reasoning']}")

    news_note = f" News: {proposal['news_summary']}." if proposal["news_summary"] else ""
    ai_note = (
        f" AI read: {proposal['reasoning']} (leaned {proposal['strategy_type']}, "
        f"{proposal['signal'].value})."
    )

    # The AI's proposed family is context for the reasoning text above, not
    # a restriction on the search below -- see module docstring for why.
    strategy_type, params, result, live_signal = search_best_across_types(df, config)

    if params is None:
        reason = (
            f"{ai_note}{news_note} No strategy/parameter combination showed a validated "
            f"edge (>= {config.BACKTEST_MIN_TRADES} trades, positive net return) over "
            f"the last {len(df)} candles — holding."
        )
        return StrategyDecision(regime, Signal.HOLD, atr, reason)

    backtest_note = (
        f"Best validated setup: {strategy_type} {params} — {result.trades} trades, "
        f"{result.win_rate * 100:.0f}% win rate, net {result.net_pnl_pct * 100:+.2f}%"
    )

    if live_signal is Signal.HOLD:
        reason = (
            f"{ai_note}{news_note} {backtest_note}. Passed, but isn't signaling an "
            f"entry on the current candle. Holding until it does."
        )
        return StrategyDecision(regime, Signal.HOLD, atr, reason, strategy_type)

    reason = (
        f"{ai_note}{news_note} {backtest_note}. Passed, and is signaling "
        f"{live_signal.value} on the current candle. Executing."
    )
    return StrategyDecision(regime, live_signal, atr, reason, strategy_type)
