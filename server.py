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
        rm = _bot.risk_manager if _bot else None

    if rm is None:
        return jsonify(running=False, balance=None)

    size_pct, atr_stop_multiple = rm.get_position_sizing()
    return jsonify(
        running=running,
        balance=round(rm.balance, 2),
        consecutive_losses=rm.consecutive_losses,
        circuit_breaker_tripped=rm.circuit_breaker_tripped,
        in_cooloff=rm.is_in_cooloff(),
        next_trade_size_pct=size_pct,
        next_trade_atr_stop_multiple=atr_stop_multiple,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
