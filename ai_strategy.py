"""AI-driven trading decision. Replaces strategy.py's fixed rules for the
trending / range-bound regimes with a two-step AI + validation pipeline:

  1. An LLM (Groq's free-tier API) looks at recent price action AND live
     headlines pulled from CNBC, Yahoo Finance, Bloomberg, WSJ, the NYT, and
     the Financial Times (news_feed.py), then proposes ONE strategy (from a
     small backtestable template set)
     plus the buy/sell/hold call that strategy makes right now.
  2. That exact strategy is replayed against recent candle history
     (backtester.run_backtest) before it's trusted. Only a strategy that
     actually shows a positive, evidenced edge over that window is allowed
     through to become a live paper trade — otherwise the bot holds, no
     matter how confident the AI sounded.

This does NOT touch risk management. The high-volatility standby rule still
overrides everything below — no new entries during a volatility spike, no
matter what the model or the backtest says. And once a signal is allowed
through, position sizing, stop distance, the circuit breaker, and the
cool-off are still entirely risk_manager.py's job, unchanged.

Uses Groq's free-tier API (no credit card required), configured via the
GROQ_API_KEY environment variable. https://console.groq.com/keys
"""

import json
import logging
import os
import re

from groq import Groq

from backtester import STRATEGY_TYPES, run_backtest
from config import Config
from news_feed import fetch_news, format_news_for_prompt
from strategy import Regime, Signal, StrategyDecision, classify_regime

logger = logging.getLogger("trading-bot")

_client = None

_GROQ_MODEL = "llama-3.3-70b-versatile"

_STRATEGY_PARAM_SPEC = (
    '  - "ma_crossover": {"fast_period": int 5-20, "slow_period": int 20-50}\n'
    '  - "support_resistance": {"lookback": int 10-40}\n'
    '  - "rsi_threshold": {"period": int 7-21, "oversold": int 20-35, "overbought": int 65-80}'
)


def _get_client():
    global _client
    if _client is None:
        _client = Groq(api_key=os.environ["GROQ_API_KEY"])
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
    propose one of STRATEGY_TYPES (with parameters) plus a right-now
    buy/sell/hold call. Fails safe to a hold on any error, block, or
    malformed response — a broken API call should never be able to force
    a trade."""
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
        "news headlines, propose ONE trading strategy from this exact set, "
        f"with parameters, that you believe fits the current market:\n{_STRATEGY_PARAM_SPEC}\n\n"
        "Then state the buy/sell/hold call that strategy makes right now.\n\n"
        'Respond with ONLY a JSON object, no other text: '
        '{"strategy_type": "ma_crossover" | "support_resistance" | "rsi_threshold", '
        '"params": {...matching the spec above...}, '
        '"signal": "buy" | "sell" | "hold", '
        '"news_summary": "one short sentence on which headline (if any) influenced this, or \'no notable news\'", '
        '"reasoning": "one short sentence"}'
    )

    try:
        response = _get_client().chat.completions.create(
            model=_GROQ_MODEL,
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
            "params": data["params"],
            "signal": Signal(data["signal"]),
            "news_summary": data.get("news_summary", ""),
            "reasoning": data["reasoning"],
        }
    except Exception as exc:
        logger.exception("AI strategy proposal failed; defaulting to hold")
        return {
            "strategy_type": None,
            "params": {},
            "signal": Signal.HOLD,
            "news_summary": "",
            "reasoning": f"AI call failed ({type(exc).__name__}: {exc}); defaulting to hold (fail-safe).",
        }


def decide_with_ai(df, config=Config) -> StrategyDecision:
    """Same return shape as strategy.decide(), but the entry signal comes
    from an LLM-proposed, news-informed strategy that must first pass a
    backtest over recent candle history (see module docstring)."""
    regime, atr = classify_regime(df, config)

    if regime is Regime.HIGH_VOL_STANDBY:
        return StrategyDecision(
            regime, Signal.HOLD, atr,
            "High volatility spike detected; standing by regardless of AI (safety rule).",
        )

    proposal = propose_strategy(df, regime, atr, config)

    if proposal["strategy_type"] is None or proposal["signal"] is Signal.HOLD:
        return StrategyDecision(regime, Signal.HOLD, atr, f"AI: {proposal['reasoning']}")

    result = run_backtest(df, proposal["strategy_type"], proposal["params"], config)
    news_note = f" News: {proposal['news_summary']}." if proposal["news_summary"] else ""
    backtest_note = (
        f"Backtest over last {len(df)} candles: {result.trades} trades, "
        f"{result.win_rate * 100:.0f}% win rate, net {result.net_pnl_pct * 100:+.2f}%"
    )

    if result.passed:
        reason = (
            f"AI proposed {proposal['strategy_type']} ({proposal['reasoning']}).{news_note} "
            f"{backtest_note} — passed, executing."
        )
        return StrategyDecision(regime, proposal["signal"], atr, reason)

    reason = (
        f"AI proposed {proposal['strategy_type']} ({proposal['reasoning']}).{news_note} "
        f"{backtest_note} — failed, holding until a validated edge appears."
    )
    return StrategyDecision(regime, Signal.HOLD, atr, reason)
