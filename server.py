"""Control-plane API for the trading bot: exposes /start, /stop, /status,
/assets, /news, and /candles so the public dashboard can pick an asset,
toggle the live trade loop on and off, and see what it's doing.

Runs the bot loop on a background thread inside this process. Note that
bot.py's TradingBot only *simulates* fills against live prices -- it never
calls exchange.create_order -- so /start does not place real orders on any
exchange or broker; it starts an in-memory paper-trading loop. /start
optionally takes a JSON body {"symbol": "AAPL"} to pick which of
Config.SUPPORTED_ASSETS to trade; switching symbols always starts a fresh
bot (balance/streaks reset) since carrying state across markets wouldn't
mean anything.

/start and /stop require a shared control token (CONTROL_TOKEN env var)
sent via the X-Control-Token header, so a stranger with the dashboard URL
can't flip the switch. /status, /assets, /news, and /candles are read-only
and unauthenticated.
"""

import logging
import os
import threading
import time

from flask import Flask, jsonify, request
from flask_cors import CORS

from bot import TradingBot, fetch_ohlcv_yahoo
from config import Config
from live_tv import get_live_tv
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
    body = request.get_json(silent=True) or {}
    requested_symbol = body.get("symbol")

    with _state_lock:
        if _running.is_set():
            return jsonify(status="already_running")

        if requested_symbol:
            if requested_symbol not in Config.SUPPORTED_ASSETS:
                return jsonify(error=f"unknown symbol {requested_symbol!r}"), 400
            if _bot is None or _bot.config.SYMBOL != requested_symbol:
                # Switching assets starts a fresh bot -- balance/streaks from
                # one market shouldn't carry over to a different one.
                Config.SYMBOL = requested_symbol
                _bot = TradingBot()
        elif _bot is None:
            _bot = TradingBot()

        _running.set()
        _loop_thread = threading.Thread(target=_run_loop, daemon=True)
        _loop_thread.start()
    return jsonify(status="started", symbol=_bot.config.SYMBOL)


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
        return jsonify(running=False, balance=None, symbol=Config.SYMBOL)

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
        symbol=bot.config.SYMBOL,
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


@app.get("/assets")
def assets():
    """Read-only, unauthenticated -- the fixed set of symbols the dashboard's
    asset selector can offer, sourced from Config.SUPPORTED_ASSETS so it's
    never out of sync with what /start will actually accept."""
    return jsonify(assets=[
        {"symbol": symbol, "label": info["label"], "kind": info["kind"]}
        for symbol, info in Config.SUPPORTED_ASSETS.items()
    ])


@app.get("/news")
def news():
    """Read-only, unauthenticated -- same live headlines (CNBC, Yahoo
    Finance, Bloomberg, WSJ, NYT, Financial Times) that ai_strategy.py feeds
    to the AI each cycle, so the dashboard can show what the AI is actually
    reading. Each item's "link" points at the full article on the source's
    own site -- the dashboard opens it there rather than reproducing it."""
    try:
        items = fetch_news()
    except Exception:
        logger.exception("News fetch failed")
        items = []

    return jsonify(items=[
        {
            "source": item["source"],
            "title": item["title"],
            "link": item.get("link", ""),
            "published_at": item["published_at"].isoformat(),
        }
        for item in items
    ])


@app.get("/live-tv")
def live_tv():
    """Read-only, unauthenticated -- resolves each network's currently-live
    YouTube video ID server-side (see live_tv.py for why: browsers can't do
    this themselves, and most networks' streams are effectively permanent
    except NBC News NOW, which starts a new video ID for every day's
    broadcast). Cached for 10 minutes per network, so this is cheap to poll.
    The dashboard falls back to its own hardcoded pins if this endpoint is
    unreachable or not configured, so it's never a hard dependency."""
    try:
        networks = get_live_tv()
    except Exception:
        logger.exception("Live TV resolve failed")
        networks = {}

    return jsonify(networks=networks)


@app.get("/candles")
def candles():
    """Read-only, unauthenticated -- OHLCV candles for any non-crypto symbol
    in Config.SUPPORTED_ASSETS (?symbol=AAPL), proxying Yahoo Finance's
    public chart API server-side. The dashboard's main chart fetches crypto
    candles directly from Coinbase/Kraken/Binance in the browser -- those
    exchanges' public market-data endpoints send CORS headers -- but Yahoo's
    chart API, the only free keyless source for stocks/commodities, does
    not, so the browser can't call it itself; this is why crypto charts
    work with no backend configured at all, while stock/commodity charts
    need one."""
    symbol = request.args.get("symbol", "")
    asset = Config.SUPPORTED_ASSETS.get(symbol)
    if not asset:
        return jsonify(error=f"unknown symbol {symbol!r}"), 400
    if asset["kind"] == "crypto":
        return jsonify(error="crypto symbols are fetched directly by the browser, not via /candles"), 400

    try:
        df = fetch_ohlcv_yahoo(asset["yahoo_symbol"], Config, limit=150)
    except Exception:
        logger.exception("Candle fetch failed for %s", symbol)
        return jsonify(error="fetch failed"), 502

    return jsonify(candles=[
        {
            "time": row.timestamp.isoformat(),
            "open": row.open,
            "high": row.high,
            "low": row.low,
            "close": row.close,
        }
        for row in df.itertuples()
    ])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
