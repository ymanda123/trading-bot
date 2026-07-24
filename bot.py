"""Main bot loop: pulls OHLCV from Binance Testnet via CCXT, classifies the
market regime, generates a signal, and manages a single paper position
subject to RiskManager guardrails (circuit breaker, sizing, cool-off,
friction).
"""

import logging
import time

import ccxt
import pandas as pd

from config import Config
from risk_manager import RiskManager
from strategy import Signal, decide

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("trading-bot")


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


def fetch_ohlcv_df(exchange: ccxt.Exchange, config=Config) -> pd.DataFrame:
    limit = max(config.SLOW_MA_PERIOD, config.ATR_PERIOD) + 5
    raw = exchange.fetch_ohlcv(config.SYMBOL, timeframe=config.TIMEFRAME, limit=limit)
    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df


class TradingBot:
    def __init__(self, config=Config, exchange: ccxt.Exchange = None, risk_manager: RiskManager = None):
        self.config = config
        self.exchange = exchange or build_exchange(config)
        self.risk_manager = risk_manager or RiskManager(config)
        self.position = None  # {"side", "entry_price", "quantity", "stop_price", "entry_friction"}

    def step(self) -> None:
        if not self.risk_manager.can_trade():
            if self.risk_manager.circuit_breaker_tripped:
                logger.warning("Circuit breaker tripped at balance $%.2f. Halting.", self.risk_manager.balance)
            else:
                logger.info("In cool-off until %s. Skipping this cycle.", self.risk_manager.cooloff_until)
            return

        df = fetch_ohlcv_df(self.exchange, self.config)
        decision = decide(df, self.config)
        price = df["close"].iloc[-1]
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
