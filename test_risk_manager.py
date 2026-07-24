"""Tests for RiskManager: circuit breaker, progressive sizing, cool-off,
and friction accounting."""

import pytest

from config import Config
from risk_manager import RiskManager


@pytest.fixture
def rm():
    return RiskManager()


# --- Circuit breaker ---------------------------------------------------------


def test_initial_balance(rm):
    assert rm.balance == Config.INITIAL_BALANCE


def test_circuit_breaker_not_tripped_initially(rm):
    assert rm.circuit_breaker_tripped is False


def test_circuit_breaker_trips_at_160(rm):
    rm.record_trade_result(-40.0)  # 200 -> 160
    assert rm.balance == pytest.approx(160.0)
    assert rm.circuit_breaker_tripped is True


def test_circuit_breaker_not_tripped_just_above_160(rm):
    rm.record_trade_result(-39.99)
    assert rm.circuit_breaker_tripped is False


def test_can_trade_blocked_after_circuit_breaker(rm):
    rm.record_trade_result(-40.0)
    assert rm.can_trade() is False


# --- Progressive position sizing ---------------------------------------------


def test_position_sizing_starts_full_size(rm):
    size_pct, atr_mult = rm.get_position_sizing()
    assert size_pct == 1.0
    assert atr_mult == 2.0


def test_position_sizing_after_one_loss(rm):
    rm.record_trade_result(-5.0)
    size_pct, atr_mult = rm.get_position_sizing()
    assert size_pct == 0.5
    assert atr_mult == 1.5


def test_position_sizing_after_two_losses(rm):
    rm.record_trade_result(-5.0)
    rm.record_trade_result(-5.0)
    size_pct, atr_mult = rm.get_position_sizing()
    assert size_pct == 0.25
    assert atr_mult == 1.0


def test_position_sizing_resets_after_a_win(rm):
    rm.record_trade_result(-5.0)
    rm.record_trade_result(10.0)
    size_pct, atr_mult = rm.get_position_sizing()
    assert size_pct == 1.0
    assert atr_mult == 2.0


def test_position_size_calculation(rm):
    result = rm.calculate_position_size(price=100.0, atr=2.0)
    assert result["size_pct"] == 1.0
    assert result["stop_distance"] == pytest.approx(4.0)  # 2.0 ATR * 2.0x multiple
    assert result["allocated_capital"] == pytest.approx(200.0)
    assert result["quantity"] == pytest.approx(2.0)
    assert result["stop_price_long"] == pytest.approx(96.0)
    assert result["stop_price_short"] == pytest.approx(104.0)


# --- Cool-off ------------------------------------------------------------------


def test_three_consecutive_losses_triggers_cooloff(rm):
    now = 1_000_000.0
    rm.record_trade_result(-5.0, now=now)
    rm.record_trade_result(-5.0, now=now + 1)
    assert rm.is_in_cooloff(now + 1) is False
    rm.record_trade_result(-5.0, now=now + 2)
    assert rm.is_in_cooloff(now + 2) is True
    assert rm.cooloff_until == pytest.approx(now + 2 + Config.COOLOFF_MINUTES * 60)


def test_cooloff_blocks_trading_until_it_expires(rm):
    now = 1_000_000.0
    for i in range(3):
        rm.record_trade_result(-5.0, now=now + i)
    assert rm.can_trade(now + 3) is False
    assert rm.can_trade(now + 3 + Config.COOLOFF_MINUTES * 60 + 1) is True


def test_cooloff_resets_size_tier_back_to_full(rm):
    now = 1_000_000.0
    for i in range(3):
        rm.record_trade_result(-5.0, now=now + i)
    size_pct, atr_mult = rm.get_position_sizing()
    assert size_pct == 1.0
    assert atr_mult == 2.0


def test_two_losses_then_a_win_does_not_trigger_cooloff(rm):
    now = 1_000_000.0
    rm.record_trade_result(-5.0, now=now)
    rm.record_trade_result(-5.0, now=now + 1)
    rm.record_trade_result(10.0, now=now + 2)
    assert rm.is_in_cooloff(now + 2) is False
    assert rm.can_trade(now + 2) is True


# --- Friction accounting -------------------------------------------------------


def test_friction_cost_applies_fee_and_slippage(rm):
    notional = 1000.0
    cost = rm.calculate_friction_cost(notional)
    expected = notional * (Config.TAKER_FEE_PCT + Config.SLIPPAGE_PCT)
    assert cost == pytest.approx(expected)
    assert cost == pytest.approx(1.25)  # 1000 * (0.00075 + 0.0005)


def test_friction_cost_scales_with_notional(rm):
    assert rm.calculate_friction_cost(2000.0) == pytest.approx(2 * rm.calculate_friction_cost(1000.0))
