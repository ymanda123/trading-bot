"""Control-plane API for the trading bot: exposes /start, /stop, /status,
/assets, /news, and /candles so the public dashboard can pick an asset,
toggle the live trade loop on and off, and see what it's doing. Also exposes
/account/deposit, /account/withdraw, and /account/circuit-breaker for the
dashboard's balance-tile popover -- demo-only manual controls to add/remove
paper capital or raise the circuit breaker's halt floor -- and
/predictions/place for the "Predict" popup's up/down/~exact price wager.
None of this is the bot's own trading logic.

Runs the bot loop on a background thread inside this process. Note that
bot.py's TradingBot only *simulates* fills against live prices -- it never
calls exchange.create_order -- so /start does not place real orders on any
exchange or broker; it starts an in-memory paper-trading loop. /start
optionally takes a JSON body {"symbol": "AAPL"} to pick which of
Config.SUPPORTED_ASSETS to trade; switching symbols always starts a fresh
bot (balance/streaks reset) since carrying state across markets wouldn't
mean anything.

/start, /stop, /account/*, and /predictions/place require a shared control
token (CONTROL_TOKEN env var) sent via the X-Control-Token header, so a
stranger with the dashboard URL can't flip the switch or touch the balance.
/status, /assets, /news, and /candles are read-only and unauthenticated.
"""

import json
import logging
import os
import threading
import time

from flask import Flask, jsonify, request
from flask_cors import CORS

from bot import TradingBot, fetch_ohlcv_df, fetch_ohlcv_yahoo
from chat_assistant import answer as chat_answer
from config import Config
from live_tv import get_live_tv
from news_feed import fetch_news
from strategy import Signal, compute_atr

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
# Set for the ~5-10s a step() call is actually mid-flight (Gemini call +
# backtest grid search), cleared the rest of the ~POLL_SECONDS cycle. /status
# exposes this so the dashboard can show a "thinking" indicator instead of
# looking frozen between decisions. Event.is_set()/set()/clear() are already
# atomic, so this needs no lock of its own.
_stepping = threading.Event()

# --- Chat assistant rate limiting -------------------------------------------
# The chat endpoint is unauthenticated (anyone with the dashboard URL can use
# it), so it shares a free-tier Gemini quota with every other visitor. This is
# a simple per-IP sliding-window limiter, not a defense against a determined
# attacker -- just enough to stop one visitor (or one bug in the frontend
# retry logic) from burning the whole site's quota.
_chat_rate_lock = threading.Lock()
_chat_rate_log = {}  # ip -> list[float] request timestamps within the window
CHAT_RATE_LIMIT = 8
CHAT_RATE_WINDOW_SEC = 60


def _chat_rate_limited(ip: str) -> bool:
    now = time.time()
    with _chat_rate_lock:
        recent = [t for t in _chat_rate_log.get(ip, []) if now - t < CHAT_RATE_WINDOW_SEC]
        if len(recent) >= CHAT_RATE_LIMIT:
            _chat_rate_log[ip] = recent
            return True
        recent.append(now)
        _chat_rate_log[ip] = recent
        return False


# --- Persisted "should be running" intent -----------------------------------
# The trade loop only lives in memory, so a process restart -- a redeploy, or
# (on Render's free tier) the whole service spinning down after ~15 minutes
# of no incoming HTTP traffic -- silently drops back to stopped even though
# nobody clicked Stop. Persisting the user's actual intent to disk and
# replaying it on boot means a restart resumes instead of going quiet; see
# _auto_resume() below, called once at import time. This is best-effort, not
# a guarantee -- it survives the process restarting on the same disk, but a
# fresh container from a new deploy may not keep this file. Combined with
# the keep-alive workflow (.github/workflows/keep-alive.yml) that pings
# /status every 10 minutes to stop the free-tier spin-down from ever
# triggering, the common case (idle timeout) is fully covered; only an
# actual code deploy still requires manually turning it back on.
_STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".bot_state.json")


def _save_desired_state(running: bool, symbol: str | None = None) -> None:
    try:
        with open(_STATE_FILE, "w") as f:
            json.dump({"running": running, "symbol": symbol}, f)
    except Exception:
        logger.exception("Failed to persist desired bot state (non-fatal)")


def _load_desired_state() -> dict:
    try:
        with open(_STATE_FILE) as f:
            return json.load(f)
    except Exception:
        return {"running": False, "symbol": None}


def _auto_resume() -> None:
    """Called once at import time. If the last known intent (before whatever
    restarted this process) was "running", resume automatically instead of
    silently sitting stopped until someone notices and flips the toggle."""
    global _bot, _loop_thread
    state = _load_desired_state()
    if not state.get("running"):
        return

    symbol = state.get("symbol")
    with _state_lock:
        if symbol and symbol in Config.SUPPORTED_ASSETS:
            Config.SYMBOL = symbol
        _bot = TradingBot()
        _running.set()
        _loop_thread = threading.Thread(target=_run_loop, daemon=True)
        _loop_thread.start()
    logger.info("Auto-resumed trading loop on %s from persisted state", Config.SYMBOL)


def _run_loop():
    logger.info("Trade loop starting")
    while _running.is_set():
        _stepping.set()
        try:
            _bot.step()
        except Exception:
            logger.exception("Error during bot step")
        finally:
            _stepping.clear()
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
        _save_desired_state(True, _bot.config.SYMBOL)
    return jsonify(status="started", symbol=_bot.config.SYMBOL)


@app.post("/stop")
def stop():
    if not _authorized(request):
        return jsonify(error="unauthorized"), 401

    with _state_lock:
        _running.clear()
        _save_desired_state(False)
    return jsonify(status="stopped")


def _account_amount_from_request(req) -> float | None:
    """Pulls a positive numeric "amount" out of a JSON body for the
    /account/* endpoints below. Returns None if missing/non-numeric/non-
    positive so callers can 400 without duplicating the check three times."""
    body = req.get_json(silent=True) or {}
    amount = body.get("amount")
    if isinstance(amount, bool) or not isinstance(amount, (int, float)) or amount <= 0:
        return None
    return float(amount)


@app.post("/account/deposit")
def account_deposit():
    """Dashboard's balance-tile popover -- "Add Money". Demo-only manual
    account control, not part of the bot's actual trading logic. Creates a
    bot instance (without starting its trade loop) if one doesn't exist yet,
    same as /start does, so this works even before the first /start call.
    Control-token gated like /start and /stop since it mutates state."""
    if not _authorized(request):
        return jsonify(error="unauthorized"), 401

    amount = _account_amount_from_request(request)
    if amount is None:
        return jsonify(error="amount must be a positive number"), 400

    global _bot
    with _state_lock:
        if _bot is None:
            _bot = TradingBot()
        _bot.risk_manager.deposit(amount)
        balance = _bot.risk_manager.balance

    return jsonify(status="ok", balance=round(balance, 2))


@app.post("/account/withdraw")
def account_withdraw():
    """Dashboard's balance-tile popover -- "Withdraw". Same scope and gating
    as /account/deposit; fails with 400 rather than allowing the balance to
    go negative."""
    if not _authorized(request):
        return jsonify(error="unauthorized"), 401

    amount = _account_amount_from_request(request)
    if amount is None:
        return jsonify(error="amount must be a positive number"), 400

    global _bot
    with _state_lock:
        if _bot is None:
            _bot = TradingBot()
        try:
            _bot.risk_manager.withdraw(amount)
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
        balance = _bot.risk_manager.balance

    return jsonify(status="ok", balance=round(balance, 2))


@app.post("/account/circuit-breaker")
def account_raise_circuit_breaker():
    """Dashboard's balance-tile popover -- "Add to Circuit Breaker". Raises
    the halt floor (RiskManager.circuit_breaker_floor) rather than the
    balance, e.g. to lock in more of the current balance as a protected
    cushion. Same scope and gating as /account/deposit; fails with 400 if the
    new floor would reach or exceed the current balance."""
    if not _authorized(request):
        return jsonify(error="unauthorized"), 401

    amount = _account_amount_from_request(request)
    if amount is None:
        return jsonify(error="amount must be a positive number"), 400

    global _bot
    with _state_lock:
        if _bot is None:
            _bot = TradingBot()
        try:
            _bot.risk_manager.raise_circuit_breaker_floor(amount)
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
        floor = _bot.risk_manager.circuit_breaker_floor

    return jsonify(status="ok", circuit_breaker_floor=round(floor, 2))


@app.post("/predictions/place")
def place_prediction():
    """Dashboard's "Predict" popup (next to the Live Trading toggle, or the
    balance popover's last option) -- a wager on whether the current asset's
    price will be up/down/~exact by a chosen time. Not part of the bot's own
    trading logic. Fetches the current price server-side (never trusts a
    client-supplied one) and hands it to RiskManager.place_prediction, which
    deducts the wager immediately. Resolution happens lazily -- see /status
    below -- once target_time has passed. Control-token gated like
    /account/* since it mutates the balance."""
    if not _authorized(request):
        return jsonify(error="unauthorized"), 401

    body = request.get_json(silent=True) or {}
    direction = body.get("direction")
    wager = body.get("wager")
    duration_seconds = body.get("duration_seconds")
    if not isinstance(wager, (int, float)) or isinstance(wager, bool):
        return jsonify(error="wager must be a number"), 400
    if not isinstance(duration_seconds, (int, float)) or isinstance(duration_seconds, bool):
        return jsonify(error="duration_seconds must be a number"), 400

    global _bot
    with _state_lock:
        if _bot is None:
            _bot = TradingBot()
        bot = _bot

    try:
        df = fetch_ohlcv_df(bot.exchange, bot.config)
        price = float(df["close"].iloc[-1])
    except Exception:
        logger.exception("Price fetch failed for prediction placement")
        return jsonify(error="failed to fetch the current price -- try again"), 502

    with _state_lock:
        try:
            pred = bot.risk_manager.place_prediction(direction, float(wager), price, float(duration_seconds))
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
        balance = bot.risk_manager.balance

    return jsonify(
        status="ok",
        balance=round(balance, 2),
        prediction={
            "symbol": bot.config.SYMBOL,
            "direction": pred.direction,
            "wager": round(pred.wager, 2),
            "price_at_bet": round(pred.price_at_bet, 2),
            "target_time": pred.target_time,
        },
    )


@app.post("/debug/force-trade")
def force_trade():
    """TEMPORARY, test-only -- opens one real paper position immediately
    using current price/ATR and the bot's normal sizing/stop/friction logic
    (RiskManager.calculate_position_size, slippage, fees), bypassing the AI
    + backtest gate entirely. This is for manually verifying the open/track/
    close mechanics work end-to-end without waiting for a live signal --
    NOT how the bot decides real trades; every other entry still goes
    through decide_with_ai. Tagged "manual_test" as its regime so it's
    obviously distinguishable in the trade log from an AI-driven entry.
    Control-token gated like /start and /stop since it mutates state.
    Remove once verified.
    """
    if not _authorized(request):
        return jsonify(error="unauthorized"), 401

    with _state_lock:
        bot = _bot
        running = _running.is_set()

    if bot is None or not running:
        return jsonify(error="bot not running -- start it first"), 400
    if bot.position is not None:
        return jsonify(error="a position is already open"), 400

    body = request.get_json(silent=True) or {}
    side = body.get("side", "buy")
    if side not in ("buy", "sell"):
        return jsonify(error="side must be 'buy' or 'sell'"), 400

    try:
        df = fetch_ohlcv_df(bot.exchange, bot.config)
        price = df["close"].iloc[-1]
        atr = compute_atr(df, bot.config.ATR_PERIOD).iloc[-1]
        signal = Signal.BUY if side == "buy" else Signal.SELL
        bot._open_position(signal, price, atr)
        bot.position["regime"] = "manual_test"
    except Exception:
        logger.exception("Force-trade failed")
        return jsonify(error="failed to open test position -- see server logs"), 500

    return jsonify(
        status="opened",
        position={
            "side": bot.position["side"].value,
            "entry_price": round(bot.position["entry_price"], 2),
            "quantity": round(bot.position["quantity"], 6),
            "stop_price": round(bot.position["stop_price"], 2),
        },
    )


@app.get("/status")
def status():
    with _state_lock:
        running = _running.is_set()
        bot = _bot

    if bot is None:
        return jsonify(running=False, thinking=False, balance=None, symbol=Config.SYMBOL)

    rm = bot.risk_manager

    # Predictions resolve lazily -- whenever anything polls /status after the
    # target time has passed, rather than needing a dedicated scheduler/
    # background thread. Fails safe: a fetch error just means it's retried
    # on the next poll instead of leaving the prediction stuck unresolved.
    if rm.prediction_due():
        try:
            df = fetch_ohlcv_df(bot.exchange, bot.config)
            with _state_lock:
                rm.resolve_prediction(float(df["close"].iloc[-1]))
        except Exception:
            logger.exception("Prediction resolution failed; will retry next poll")

    pending_prediction = None
    if rm.pending_prediction:
        pred = rm.pending_prediction
        pending_prediction = {
            "direction": pred.direction,
            "wager": round(pred.wager, 2),
            "price_at_bet": round(pred.price_at_bet, 2),
            "target_time": pred.target_time,
            "resolved": pred.resolved,
            "outcome": pred.outcome,
            "resolution_price": round(pred.resolution_price, 2) if pred.resolution_price is not None else None,
        }

    size_pct, atr_stop_multiple = rm.get_position_sizing()

    position = None
    unrealized_pnl = None
    if bot.position:
        pos = bot.position
        position = {
            "side": pos["side"].value,
            "entry_price": round(pos["entry_price"], 2),
            "quantity": round(pos["quantity"], 6),
            "stop_price": round(pos["stop_price"], 2),
            "opened_at": pos.get("opened_at"),
        }
        if bot.last_price is not None:
            # Mirrors bot.py's _check_exit math exactly (gross price move,
            # entry friction already paid, exit friction estimated at the
            # current price) -- this is "what you'd actually net if it
            # closed right now," not just a raw price-delta estimate.
            gross = (bot.last_price - pos["entry_price"]) * pos["quantity"]
            if pos["side"] is Signal.SELL:
                gross = -gross
            estimated_exit_friction = rm.calculate_friction_cost(bot.last_price * pos["quantity"])
            unrealized_pnl = gross - pos["entry_friction"] - estimated_exit_friction

    recent_trades = list(reversed(bot.trade_log[-20:]))
    last_trade = recent_trades[0] if recent_trades else None

    return jsonify(
        running=running,
        thinking=_stepping.is_set(),
        symbol=bot.config.SYMBOL,
        balance=round(rm.balance, 2),
        net_pnl=round(rm.balance - bot.config.INITIAL_BALANCE, 2),
        consecutive_losses=rm.consecutive_losses,
        circuit_breaker_tripped=rm.circuit_breaker_tripped,
        circuit_breaker_floor=round(rm.circuit_breaker_floor, 2),
        in_cooloff=rm.is_in_cooloff(),
        next_trade_size_pct=size_pct,
        next_trade_atr_stop_multiple=atr_stop_multiple,
        last_step_at=bot.last_step_at,
        last_price=round(bot.last_price, 2) if bot.last_price is not None else None,
        current_regime=bot.last_decision.regime.value if bot.last_decision else None,
        current_signal=bot.last_decision.signal.value if bot.last_decision else None,
        current_strategy=bot.last_decision.strategy_type if bot.last_decision else None,
        current_reason=bot.last_decision.reason if bot.last_decision else None,
        position=position,
        unrealized_pnl=round(unrealized_pnl, 2) if unrealized_pnl is not None else None,
        pending_prediction=pending_prediction,
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


def _status_context_text() -> str:
    """Short plain-text summary of live bot state for the chat assistant --
    the same fields /status returns, condensed so it's cheap to include in
    every prompt."""
    with _state_lock:
        running = _running.is_set()
        bot = _bot

    if bot is None:
        return "running=False (bot has not been started yet)"

    rm = bot.risk_manager
    return (
        f"running={running} symbol={bot.config.SYMBOL} "
        f"balance=${rm.balance:.2f} net_pnl=${rm.balance - bot.config.INITIAL_BALANCE:.2f} "
        f"regime={bot.last_decision.regime.value if bot.last_decision else 'unknown'} "
        f"signal={bot.last_decision.signal.value if bot.last_decision else 'unknown'} "
        f"reason={bot.last_decision.reason if bot.last_decision else 'n/a'} "
        f"consecutive_losses={rm.consecutive_losses} "
        f"circuit_breaker_tripped={rm.circuit_breaker_tripped} "
        f"in_cooloff={rm.is_in_cooloff()} "
        f"trade_count={len(bot.trade_log)}"
    )


@app.post("/chat")
def chat():
    """Read-only, unauthenticated -- public chat assistant for the website.
    Uses the same free Gemini API as ai_strategy.py but never touches trading
    logic; it can only describe what the bot is doing, not change it. Rate
    limited per IP (see _chat_rate_limited) to protect the shared free-tier
    quota."""
    ip = request.headers.get("X-Forwarded-For", request.remote_addr or "unknown").split(",")[0].strip()
    if _chat_rate_limited(ip):
        return jsonify(error="rate_limited"), 429

    body = request.get_json(silent=True) or {}
    message = body.get("message", "")
    history = body.get("history", [])
    if not isinstance(message, str) or not message.strip():
        return jsonify(error="message required"), 400
    if not isinstance(history, list):
        history = []

    reply = chat_answer(message, history, _status_context_text())
    return jsonify(reply=reply)


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


# Runs at import time -- under gunicorn (production) the module is imported,
# never executed as __main__, so this has to live here rather than inside
# the block below to actually take effect on Render.
_auto_resume()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
