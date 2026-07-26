"""Tests for backtester.py: signal generation for each strategy template,
and the trade-simulation / pass-fail gate in run_backtest."""

import pandas as pd
import pytest

from backtester import generate_signals, run_backtest
from config import Config
from strategy import Signal


def _df(prices):
    return pd.DataFrame({"close": prices})


# --- generate_signals: ma_crossover --------------------------------------------


def test_ma_crossover_signals():
    # fast_period=1 (= close itself), slow_period=2 (= avg of current+prior close)
    prices = [10, 10, 20, 20, 10, 10]
    signals = generate_signals(_df(prices), "ma_crossover", {"fast_period": 1, "slow_period": 2})
    assert list(signals) == [
        Signal.HOLD, Signal.HOLD, Signal.BUY, Signal.HOLD, Signal.SELL, Signal.HOLD,
    ]


# --- generate_signals: support_resistance --------------------------------------


def test_support_resistance_signals():
    prices = [10, 15, 20, 10, 15, 20, 10, 15, 20, 10]
    signals = generate_signals(_df(prices), "support_resistance", {"lookback": 3})
    assert list(signals) == [
        Signal.HOLD, Signal.HOLD, Signal.SELL, Signal.BUY, Signal.HOLD,
        Signal.SELL, Signal.BUY, Signal.HOLD, Signal.SELL, Signal.BUY,
    ]


# --- generate_signals: rsi_threshold --------------------------------------------


def test_rsi_threshold_signals_uptrend_is_overbought():
    prices = [10, 11, 12, 13, 14, 15]
    signals = generate_signals(_df(prices), "rsi_threshold", {"period": 2, "oversold": 30, "overbought": 70})
    assert list(signals) == [
        Signal.HOLD, Signal.HOLD, Signal.SELL, Signal.SELL, Signal.SELL, Signal.SELL,
    ]


def test_rsi_threshold_signals_downtrend_is_oversold():
    prices = [15, 14, 13, 12, 11, 10]
    signals = generate_signals(_df(prices), "rsi_threshold", {"period": 2, "oversold": 30, "overbought": 70})
    assert list(signals) == [
        Signal.HOLD, Signal.HOLD, Signal.BUY, Signal.BUY, Signal.BUY, Signal.BUY,
    ]


def test_unknown_strategy_type_raises():
    with pytest.raises(ValueError):
        generate_signals(_df([1, 2, 3]), "not_a_real_strategy", {})


# --- run_backtest ---------------------------------------------------------------


def test_run_backtest_fails_below_min_trade_count():
    # Same series as test_ma_crossover_signals: 2 trades, both losers (lag
    # makes it buy the top and sell the bottom) -- fails on trade count alone.
    prices = [10, 10, 20, 20, 10, 10]
    result = run_backtest(_df(prices), "ma_crossover", {"fast_period": 1, "slow_period": 2})
    friction = 2 * (Config.TAKER_FEE_PCT + Config.SLIPPAGE_PCT)
    assert result.trades == 2
    assert result.wins == 0
    assert result.win_rate == pytest.approx(0.0)
    assert result.net_pnl_pct == pytest.approx((-0.5 - friction) + (0.0 - friction))
    assert result.passed is False  # trades (2) < BACKTEST_MIN_TRADES (3)


def test_run_backtest_passes_with_enough_winning_trades():
    # Same series as test_support_resistance_signals: 5 buy-low/sell-high
    # trades, all winners.
    prices = [10, 15, 20, 10, 15, 20, 10, 15, 20, 10]
    result = run_backtest(_df(prices), "support_resistance", {"lookback": 3})
    assert result.trades == 5
    assert result.wins == 5
    assert result.win_rate == pytest.approx(1.0)
    assert result.net_pnl_pct > 0
    assert result.passed is True


def test_run_backtest_no_signals_means_no_trades_and_fails():
    prices = [10, 10, 10, 10, 10]
    result = run_backtest(_df(prices), "ma_crossover", {"fast_period": 2, "slow_period": 3})
    assert result.trades == 0
    assert result.win_rate == pytest.approx(0.0)
    assert result.net_pnl_pct == pytest.approx(0.0)
    assert result.passed is False
