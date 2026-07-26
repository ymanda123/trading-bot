"""Control-plane API for the trading bot: exposes /start, /stop, and
/status so the public dashboard can toggle the live trade loop on and off.

Runs the bot loop on a background thread inside this process. Note that
bot.py's TradingBot only *simulates* fills against live testnet prices —
it never calls exchange.create_order — so /start does not place real
exchange orders; it starts an in-memory paper-trading loop.

/start and /stop require a shared control token (CONTROL_TOKEN env var)
sent via the X-Control-Token header, so a stranger with the dashboard URL
can't flip the switch. /status is read-only and unauthenticated.
"""

import logging
import os
import threading
import time

from flask import Flask, jsonify, request
from flask_cors import CORS

from bot import TradingBot
from news_feed import fetch_news

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("trading-bot-server")

app = Flask(__name__)
ALLOWED_ORIGIN = os.getenv("ALLOWED_ORIGIN", "https://ymanda123.github.io")
CORS(app, resources={r"/*": {"origins": ALLOWED_ORIGIN}})

CONTROL_TOKEN = os.getenv("CONTROL_TOKEN", "")
POLL_SECONDS = int(os.getenv("POLL_SECONDS", "60"))

_state_lock = threading.Lock()
_bot = None
_loop_thread = None
_running = threading.Event()


def _run_loop():
    logger.info("Trade loop starting")
    while _running.is_set():
        try:
            _bot.step()
        except Exception:
            logger.exception("Error during bot step")
        for _ in range(POLL_SECONDS):
            if not _running.is_set():
                break
            time.sleep(1)
    logger.info("Trade loop stopped")


def _authorized(req) -> bool:
    return bool(CONTROL_TOKEN) and req.headers.get("X-Control-Token") == CONTROL_TOKEN


@app.post("/start")
def start():
    if not _authorized(request):
        return jsonify(error="unauthorized"), 401

    global _bot, _loop_thread
    with _state_lock:
        if _running.is_set():
            return jsonify(status="already_running")
        if _bot is None:
            _bot = TradingBot()
        _running.set()
        _loop_thread = threading.Thread(target=_run_loop, daemon=True)
        _loop_thread.start()
    return jsonify(status="started")


@app.post("/stop")
def stop():
    if not _authorized(request):
        return jsonify(error="unauthorized"), 401

    with _state_lock:
        _running.clear()
    return jsonify(status="stopped")


@app.get("/status")
def status():
    with _state_lock:
        running = _running.is_set()
        bot = _bot

    if bot is None:
        return jsonify(running=False, balance=None)

    rm = bot.risk_manager
    size_pct, atr_stop_multiple = rm.get_position_sizing()

    position = None
    if bot.position:
        position = {
            "side": bot.position["side"].value,
            "entry_price": round(bot.position["entry_price"], 2),
            "quantity": round(bot.position["quantity"], 6),
            "stop_price": round(bot.position["stop_price"], 2),
        }

    recent_trades = list(reversed(bot.trade_log[-20:]))
    last_trade = recent_trades[0] if recent_trades else None

    return jsonify(
        running=running,
        balance=round(rm.balance, 2),
        net_pnl=round(rm.balance - bot.config.INITIAL_BALANCE, 2),
        consecutive_losses=rm.consecutive_losses,
        circuit_breaker_tripped=rm.circuit_breaker_tripped,
        in_cooloff=rm.is_in_cooloff(),
        next_trade_size_pct=size_pct,
        next_trade_atr_stop_multiple=atr_stop_multiple,
        last_step_at=bot.last_step_at,
        last_price=round(bot.last_price, 2) if bot.last_price is not None else None,
        current_regime=bot.last_decision.regime.value if bot.last_decision else None,
        current_signal=bot.last_decision.signal.value if bot.last_decision else None,
        current_reason=bot.last_decision.reason if bot.last_decision else None,
        position=position,
        trade_count=len(bot.trade_log),
        last_trade=last_trade,
        recent_trades=recent_trades,
    )


@app.get("/news")
def news():
    """Read-only, unauthenticated -- same live headlines (CNBC, Yahoo
    Finance, Bloomberg) that ai_strategy.py feeds to the AI each cycle, so
    the dashboard can show what the AI is actually reading."""
    try:
        items = fetch_news()
    except Exception:
        logger.exception("News fetch failed")
        items = []

    return jsonify(items=[
        {
            "source": item["source"],
            "title": item["title"],
            "published_at": item["published_at"].isoformat(),
        }
        for item in items
    ])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
