"""Quick historical backtest used as a gate in front of the AI's proposed
strategy (see ai_strategy.py): before a signal is allowed to become a live
paper trade, the strategy that produced it is replayed against recent
candle history. Only a strategy with a positive, evidenced track record
over that window is allowed through — otherwise the bot holds.

Each strategy template is a plain vectorized function computing a Signal
per bar; run_backtest() then walks the bars in order and simulates one
position at a time, applying the same friction assumptions as
risk_manager.py, to get a trade count / win rate / net pct return.
"""

from dataclasses import dataclass

import pandas as pd

from config import Config
from strategy import Signal

STRATEGY_TYPES = ("ma_crossover", "support_resistance", "rsi_threshold")


@dataclass
class BacktestResult:
    trades: int
    wins: int
    win_rate: float
    net_pnl_pct: float
    passed: bool


def _ma_crossover_signals(df: pd.DataFrame, params: dict) -> pd.Series:
    fast = df["close"].rolling(int(params["fast_period"])).mean()
    slow = df["close"].rolling(int(params["slow_period"])).mean()
    diff = fast - slow
    prev_diff = diff.shift(1)

    signals = pd.Series(Signal.HOLD, index=df.index)
    signals[(prev_diff <= 0) & (diff > 0)] = Signal.BUY
    signals[(prev_diff >= 0) & (diff < 0)] = Signal.SELL
    return signals


def _support_resistance_signals(df: pd.DataFrame, params: dict) -> pd.Series:
    lookback = int(params["lookback"])
    support = df["close"].rolling(lookback).min()
    resistance = df["close"].rolling(lookback).max()

    signals = pd.Series(Signal.HOLD, index=df.index)
    signals[df["close"] <= support] = Signal.BUY
    signals[df["close"] >= resistance] = Signal.SELL
    return signals


def _rsi_threshold_signals(df: pd.DataFrame, params: dict) -> pd.Series:
    period = int(params["period"])
    delta = df["close"].diff()
    avg_gain = delta.clip(lower=0).rolling(period).mean()
    avg_loss = (-delta.clip(upper=0)).rolling(period).mean()

    rs = avg_gain / avg_loss.replace(0, float("nan"))
    rsi = 100 - (100 / (1 + rs))
    rsi[(avg_loss == 0) & (avg_gain > 0)] = 100.0  # no losses in window -> fully overbought
    rsi[(avg_loss == 0) & (avg_gain == 0)] = 50.0  # flat price -> neutral
    rsi = rsi.fillna(50)  # not enough history yet -> neutral, not a signal

    signals = pd.Series(Signal.HOLD, index=df.index)
    signals[rsi <= params["oversold"]] = Signal.BUY
    signals[rsi >= params["overbought"]] = Signal.SELL
    return signals


_SIGNAL_GENERATORS = {
    "ma_crossover": _ma_crossover_signals,
    "support_resistance": _support_resistance_signals,
    "rsi_threshold": _rsi_threshold_signals,
}


def generate_signals(df: pd.DataFrame, strategy_type: str, params: dict) -> pd.Series:
    if strategy_type not in _SIGNAL_GENERATORS:
        raise ValueError(f"unknown strategy_type {strategy_type!r}")
    return _SIGNAL_GENERATORS[strategy_type](df, params)


def run_backtest(df: pd.DataFrame, strategy_type: str, params: dict, config=Config) -> BacktestResult:
    """Replay a strategy template over df bar-by-bar, one position at a
    time: open on a buy/sell signal while flat, close on the opposite
    signal (or at the final bar, to mark any still-open position). Returns
    whether the strategy showed a real, positive edge over this window."""
    signals = generate_signals(df, strategy_type, params)
    closes = df["close"].reset_index(drop=True)
    signals = signals.reset_index(drop=True)
    friction_pct = config.TAKER_FEE_PCT + config.SLIPPAGE_PCT

    position = None  # {"side": Signal, "entry_price": float}
    trade_pnls = []

    for i in range(len(closes)):
        price = closes.iloc[i]
        sig = signals.iloc[i]

        if position is None:
            if sig in (Signal.BUY, Signal.SELL):
                position = {"side": sig, "entry_price": price}
            continue

        opposite = Signal.SELL if position["side"] is Signal.BUY else Signal.BUY
        is_last_bar = i == len(closes) - 1
        if sig is opposite or is_last_bar:
            entry = position["entry_price"]
            if position["side"] is Signal.BUY:
                raw_return_pct = (price - entry) / entry
            else:
                raw_return_pct = (entry - price) / entry
            trade_pnls.append(raw_return_pct - 2 * friction_pct)
            position = {"side": sig, "entry_price": price} if (sig in (Signal.BUY, Signal.SELL) and not is_last_bar) else None

    trades = len(trade_pnls)
    wins = sum(1 for p in trade_pnls if p > 0)
    win_rate = wins / trades if trades else 0.0
    net_pnl_pct = sum(trade_pnls)
    passed = bool(trades >= config.BACKTEST_MIN_TRADES and net_pnl_pct > 0)

    return BacktestResult(trades=trades, wins=wins, win_rate=win_rate, net_pnl_pct=net_pnl_pct, passed=passed)
