"""Public-facing AI chat assistant for the website.

Separate from ai_strategy.py's trading-decision pipeline -- this one only
answers visitor questions about the bot, using the same free-tier Groq API
(GROQ_API_KEY). It never places trades, changes config, or touches
risk_manager; it only ever sees the same data /status, /news, and /assets
already expose publicly, handed to it as a short text summary by the caller.
"""

import logging
import os

from groq import Groq

logger = logging.getLogger("trading-bot")

_client = None

_CHAT_MODEL = "llama-3.3-70b-versatile"

MAX_MESSAGE_CHARS = 500
MAX_HISTORY_TURNS = 6  # user+assistant pairs of prior context to keep

_SYSTEM_PROMPT = (
    "You are the on-site assistant for a regime-switching paper-trading bot demo. "
    "Explain how the bot works when asked: it classifies market volatility into "
    "regimes (low / moderate / high) from a 14-period ATR; in the low and moderate "
    "regimes, an AI reads recent candles plus live financial news and proposes one "
    "strategy (MA crossover, support/resistance, or RSI threshold) with parameters, "
    "plus a buy/sell/hold call. That exact strategy is then backtested over recent "
    "candle history and only allowed to trade if it shows a real, evidenced edge -- "
    "otherwise the bot holds no matter how confident the AI sounded. In the high "
    "volatility regime the bot always holds regardless of the AI, as a fixed safety "
    "rule. A separate risk manager (not the AI) handles position sizing that shrinks "
    "after losses, ATR-based stop distances, a circuit breaker that halts new trades "
    "at a 20% drawdown, and a 15-minute cool-off after 3 consecutive losses. "
    "This is a PAPER-TRADING SIMULATION ONLY -- no real money and no real orders "
    "move on any exchange, ever. Never imply otherwise, and never give financial "
    "advice or tell anyone to buy or sell for real. If asked about the bot's "
    "current status, use the CURRENT STATUS block below when present; if it's "
    "missing, say you don't have live data right now rather than guessing. Keep "
    "answers short -- a few sentences, plain text, no markdown headers or lists."
)


def _get_client():
    global _client
    if _client is None:
        _client = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _client


def _clip(text, limit) -> str:
    return (text or "").strip()[:limit]


def answer(message, history=None, status_context=None) -> str:
    """Return the assistant's reply to one chat message, given optional
    prior turns and a short text block describing current bot status.
    Fails safe to a plain apology string on any error -- a broken or
    rate-limited API call should never surface a stack trace to a visitor."""
    message = _clip(message, MAX_MESSAGE_CHARS)
    if not message:
        return "Ask me anything about how the bot works."

    system = _SYSTEM_PROMPT
    if status_context:
        system += f"\n\nCURRENT STATUS:\n{_clip(status_context, 2000)}"

    messages = [{"role": "system", "content": system}]
    for turn in (history or [])[-MAX_HISTORY_TURNS * 2:]:
        role = turn.get("role") if isinstance(turn, dict) else None
        content = _clip(turn.get("content", ""), MAX_MESSAGE_CHARS) if isinstance(turn, dict) else ""
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": message})

    try:
        response = _get_client().chat.completions.create(
            model=_CHAT_MODEL,
            messages=messages,
            temperature=0.4,
            max_tokens=300,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        logger.exception("Chat assistant call failed")
        return "Sorry, I couldn't reach the AI service just now — try again in a moment."
