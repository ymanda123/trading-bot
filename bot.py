"""Main bot loop: pulls OHLCV from Binance Testnet via CCXT, classifies the
market regime, generates a signal, and manages a single paper position
subject to RiskManager guardrails (circuit breaker, sizing, cool-off,
friction).
"""

import logging
import time

import ccxt
import pandas as pd
import requests

from ai_strategy import decide_with_ai
from config import Config
from risk_manager import RiskManager
from strategy import Signal

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("trading-bot")

# Binance geo-blocks a meaningful share of hosting networks ("restricted
# location" per its terms) — confirmed to affect at least one deployment
# target. These give fetch_ohlcv_df somewhere to fall back to so the bot
# still sees prices even when Binance itself is unreachable from the host.
_COINBASE_GRANULARITY = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "6h": 21600, "1d": 86400}


def build_exchange(config=Config) -> ccxt.Exchange:
    exchange = ccxt.binance(
        {
            "apiKey": config.API_KEY,
            "secret": config.API_SECRET,
            "enableRateLimit": True,
        }
    )
    exchange.set_sandbox_mode(config.TESTNET)
    return exchange


def _to_coinbase_product(symbol: str) -> str:
    base, quote = symbol.split("/")
    return f"{base}-{'USD' if quote == 'USDT' else quote}"


def _to_kraken_pair(symbol: str) -> str:
    base, quote = symbol.split("/")
    base = "XBT" if base == "BTC" else base
    return f"{base}{'USD' if quote == 'USDT' else quote}"


def _fetch_ohlcv_coinbase(config, limit: int) -> pd.DataFrame:
    granularity = _COINBASE_GRANULARITY.get(config.TIMEFRAME, 3600)
    resp = requests.get(
        f"https://api.exchange.coinbase.com/products/{_to_coinbase_product(config.SYMBOL)}/candles",
        params={"granularity": granularity},
        timeout=10,
    )
    resp.raise_for_status()
    rows = sorted(resp.json(), key=lambda r: r[0])[-limit:]  # [time_s, low, high, open, close, volume]
    df = pd.DataFrame(rows, columns=["timestamp", "low", "high", "open", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    return df[["timestamp", "open", "high", "low", "close", "volume"]]


def _fetch_ohlcv_kraken(config, limit: int) -> pd.DataFrame:
    granularity = _COINBASE_GRANULARITY.get(config.TIMEFRAME, 3600)
    resp = requests.get(
        "https://api.kraken.com/0/public/OHLC",
        params={"pair": _to_kraken_pair(config.SYMBOL), "interval": max(granularity // 60, 1)},
        timeout=10,
    )
    resp.raise_for_status()
    body = resp.json()
    if body.get("error"):
        raise RuntimeError(f"Kraken error: {body['error']}")
    result_key = next(k for k in body["result"] if k != "last")
    rows = body["result"][result_key][-limit:]
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "vwap", "volume", "count"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    for col in ("open", "high", "low", "close"):
        df[col] = df[col].astype(float)
    return df[["timestamp", "open", "high", "low", "close", "volume"]]


def fetch_ohlcv_df(exchange: ccxt.Exchange, config=Config) -> pd.DataFrame:
    limit = max(config.SLOW_MA_PERIOD, config.ATR_PERIOD, config.AI_HISTORY_CANDLES) + 5
    try:
        raw = exchange.fetch_ohlcv(config.SYMBOL, timeframe=config.TIMEFRAME, limit=limit)
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df
    except Exception as binance_exc:
        logger.warning("Binance fetch failed (%s); falling back to Coinbase", binance_exc)
    try:
        return _fetch_ohlcv_coinbase(config, limit)
    except Exception as coinbase_exc:
        logger.warning("Coinbase fallback failed (%s); falling back to Kraken", coinbase_exc)
    return _fetch_ohlcv_kraken(config, limit)


class TradingBot:
    def __init__(self, config=Config, exchange: ccxt.Exchange = None, risk_manager: RiskManager = None):
        self.config = config
        self.exchange = exchange or build_exchange(config)
        self.risk_manager = risk_manager or RiskManager(config)
        self.position = None  # {"side", "entry_price", "quantity", "stop_price", "entry_friction"}
        self.last_step_at = None
        self.last_decision = None  # StrategyDecision from the most recent step
        self.last_price = None
        self.trade_log = []  # closed trades: side, entry/exit price, qty, pnl, regime, timestamps

    def step(self) -> None:
        self.last_step_at = time.time()

        if not self.risk_manager.can_trade():
            if self.risk_manager.circuit_breaker_tripped:
                logger.warning("Circuit breaker tripped at balance $%.2f. Halting.", self.risk_manager.balance)
            else:
                logger.info("In cool-off until %s. Skipping this cycle.", self.risk_manager.cooloff_until)
            return

        df = fetch_ohlcv_df(self.exchange, self.config)
        decision = decide_with_ai(df, self.config)
        price = df["close"].iloc[-1]
        self.last_decision = decision
        self.last_price = price
        logger.info(
            "Regime=%s Signal=%s ATR=%.4f Price=%.2f | %s",
            decision.regime.value, decision.signal.value, decision.atr, price, decision.reason,
        )

        if self.position is None and decision.signal in (Signal.BUY, Signal.SELL):
            self._open_position(decision.signal, price, decision.atr)
        elif self.position is not None:
            self._check_exit(price)

    def _open_position(self, signal: Signal, price: float, atr: float) -> None:
        sizing = self.risk_manager.calculate_position_size(price, atr)
        slip = self.config.SLIPPAGE_PCT
        entry_fill = price * (1 + slip) if signal is Signal.BUY else price * (1 - slip)
        friction = self.risk_manager.calculate_friction_cost(sizing["allocated_capital"])
        stop_price = sizing["stop_price_long"] if signal is Signal.BUY else sizing["stop_price_short"]

        self.position = {
            "side": signal,
            "entry_price": entry_fill,
            "quantity": sizing["quantity"],
            "stop_price": stop_price,
            "entry_friction": friction,
            "regime": self.last_decision.regime.value if self.last_decision else None,
            "opened_at": time.time(),
        }
        logger.info(
            "Opened %s position: qty=%.6f entry=%.2f stop=%.2f size=%.0f%% atr_mult=%.1fx",
            signal.value, sizing["quantity"], entry_fill, stop_price,
            sizing["size_pct"] * 100, sizing["atr_stop_multiple"],
        )

    def _check_exit(self, price: float) -> None:
        pos = self.position
        hit_stop = (
            (pos["side"] is Signal.BUY and price <= pos["stop_price"])
            or (pos["side"] is Signal.SELL and price >= pos["stop_price"])
        )
        if not hit_stop:
            return

        slip = self.config.SLIPPAGE_PCT
        exit_fill = price * (1 - slip) if pos["side"] is Signal.BUY else price * (1 + slip)

        gross_pnl = (exit_fill - pos["entry_price"]) * pos["quantity"]
        if pos["side"] is Signal.SELL:
            gross_pnl = -gross_pnl

        exit_friction = self.risk_manager.calculate_friction_cost(exit_fill * pos["quantity"])
        net_pnl = gross_pnl - pos["entry_friction"] - exit_friction

        self.risk_manager.record_trade_result(net_pnl)
        self.trade_log.append({
            "side": pos["side"].value,
            "entry_price": round(pos["entry_price"], 2),
            "exit_price": round(exit_fill, 2),
            "quantity": round(pos["quantity"], 6),
            "pnl": round(net_pnl, 2),
            "regime": pos.get("regime"),
            "opened_at": pos.get("opened_at"),
            "closed_at": time.time(),
        })
        logger.info(
            "Closed %s position: exit=%.2f net_pnl=%.2f balance=%.2f",
            pos["side"].value, exit_fill, net_pnl, self.risk_manager.balance,
        )
        self.position = None

    def run(self, poll_seconds: int = 60) -> None:
        logger.info(
            "Starting trading bot on %s (%s) — testnet=%s",
            self.config.SYMBOL, self.config.TIMEFRAME, self.config.TESTNET,
        )
        while True:
            try:
                self.step()
            except Exception:
                logger.exception("Error during bot step")
            time.sleep(poll_seconds)


if __name__ == "__main__":
    TradingBot().run()
