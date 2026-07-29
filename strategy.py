"""Regime classification and signal generation.

Market regime is classified each bar from a 14-period ATR expressed as a
percentage of price:
  - ATR% <= LOW_VOL_ATR_PCT_THRESHOLD   -> low volatility / range-bound
  - ATR% >= HIGH_VOL_ATR_PCT_THRESHOLD  -> high volatility / spike
  - otherwise                           -> moderate volatility / trending

Each regime routes to its own signal logic:
  - Range-bound:  grid / mean-reversion (buy support, sell resistance)
  - Trending:     fast/slow moving-average crossover
  - Spike:        cash / standby (no new entries)
"""

from dataclasses import dataclass
from enum import Enum

import pandas as pd

from config import Config


class Regime(Enum):
    LOW_VOL_RANGE = "low_volatility_range_bound"
    MODERATE_VOL_TREND = "moderate_volatility_trending"
    HIGH_VOL_STANDBY = "high_volatility_standby"


class Signal(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass
class StrategyDecision:
    regime: Regime
    signal: Signal
    atr: float
    reason: str
    # Name of the validated strategy template currently in play (e.g.
    # "rsi_threshold"), or None when nothing's active (standby, no AI
    # proposal, or nothing passed the backtest gate -- see ai_strategy.py).
    # Defaults to None so this module's own fixed-rule decide() -- used by
    # app.py's Streamlit dashboard, not the AI pipeline -- doesn't need to
    # set it.
    strategy_type: str | None = None


def compute_atr(df: pd.DataFrame, period: int = Config.ATR_PERIOD) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(window=period).mean()


def classify_regime(df: pd.DataFrame, config=Config):
    atr_series = compute_atr(df, config.ATR_PERIOD)
    atr = atr_series.iloc[-1]
    price = df["close"].iloc[-1]
    atr_pct = atr / price if price else 0.0

    if atr_pct >= config.HIGH_VOL_ATR_PCT_THRESHOLD:
        return Regime.HIGH_VOL_STANDBY, atr
    if atr_pct <= config.LOW_VOL_ATR_PCT_THRESHOLD:
        return Regime.LOW_VOL_RANGE, atr
    return Regime.MODERATE_VOL_TREND, atr


def grid_mean_reversion_signal(df: pd.DataFrame, config=Config) -> Signal:
    window = df["close"].iloc[-config.GRID_LOOKBACK :]
    support, resistance = window.min(), window.max()
    price = df["close"].iloc[-1]
    if price <= support:
        return Signal.BUY
    if price >= resistance:
        return Signal.SELL
    return Signal.HOLD


def trend_following_signal(df: pd.DataFrame, config=Config) -> Signal:
    fast_ma = df["close"].rolling(config.FAST_MA_PERIOD).mean()
    slow_ma = df["close"].rolling(config.SLOW_MA_PERIOD).mean()

    if fast_ma.dropna().shape[0] < 2 or slow_ma.dropna().shape[0] < 2:
        return Signal.HOLD

    prev_diff = fast_ma.iloc[-2] - slow_ma.iloc[-2]
    curr_diff = fast_ma.iloc[-1] - slow_ma.iloc[-1]

    if prev_diff <= 0 < curr_diff:
        return Signal.BUY
    if prev_diff >= 0 > curr_diff:
        return Signal.SELL
    return Signal.HOLD


def decide(df: pd.DataFrame, config=Config) -> StrategyDecision:
    regime, atr = classify_regime(df, config)

    if regime is Regime.HIGH_VOL_STANDBY:
        return StrategyDecision(
            regime, Signal.HOLD, atr,
            "High volatility spike detected; standing by, no new entries.",
        )

    if regime is Regime.LOW_VOL_RANGE:
        signal = grid_mean_reversion_signal(df, config)
        return StrategyDecision(
            regime, signal, atr,
            "Range-bound regime; grid/mean-reversion at support/resistance.",
        )

    signal = trend_following_signal(df, config)
    return StrategyDecision(
        regime, signal, atr,
        "Trending regime; fast/slow moving-average crossover.",
    )
