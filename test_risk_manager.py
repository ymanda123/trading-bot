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


# --- Manual account actions ---------------------------------------------------


def test_deposit_increases_balance(rm):
    rm.deposit(50.0)
    assert rm.balance == pytest.approx(250.0)


def test_deposit_rejects_non_positive_amount(rm):
    with pytest.raises(ValueError):
        rm.deposit(0)


def test_withdraw_decreases_balance(rm):
    rm.withdraw(50.0)
    assert rm.balance == pytest.approx(150.0)


def test_withdraw_rejects_more_than_balance(rm):
    with pytest.raises(ValueError):
        rm.withdraw(500.0)


def test_raise_circuit_breaker_floor(rm):
    rm.raise_circuit_breaker_floor(10.0)
    assert rm.circuit_breaker_floor == pytest.approx(170.0)


def test_raise_circuit_breaker_floor_rejects_reaching_balance(rm):
    with pytest.raises(ValueError):
        rm.raise_circuit_breaker_floor(200.0)  # floor would hit the $200 balance


# --- Prediction mini-game -------------------------------------------------------


def test_place_prediction_deducts_wager(rm):
    rm.place_prediction("up", 20.0, 100.0, 60.0, now=1000.0)
    assert rm.balance == pytest.approx(180.0)


def test_place_prediction_rejects_bad_direction(rm):
    with pytest.raises(ValueError):
        rm.place_prediction("sideways", 20.0, 100.0, 60.0, now=1000.0)


def test_place_prediction_rejects_wager_over_balance(rm):
    with pytest.raises(ValueError):
        rm.place_prediction("up", 500.0, 100.0, 60.0, now=1000.0)


def test_place_prediction_rejects_second_pending(rm):
    rm.place_prediction("up", 20.0, 100.0, 60.0, now=1000.0)
    with pytest.raises(ValueError):
        rm.place_prediction("down", 10.0, 100.0, 60.0, now=1001.0)


def test_prediction_not_due_before_target_time(rm):
    rm.place_prediction("up", 20.0, 100.0, 60.0, now=1000.0)
    assert rm.prediction_due(now=1030.0) is False


def test_prediction_due_after_target_time(rm):
    rm.place_prediction("up", 20.0, 100.0, 60.0, now=1000.0)
    assert rm.prediction_due(now=1061.0) is True


def test_resolve_up_prediction_win_doubles_wager(rm):
    rm.place_prediction("up", 20.0, 100.0, 60.0, now=1000.0)  # balance 180
    pred = rm.resolve_prediction(105.0)
    assert pred.outcome == "win"
    assert rm.balance == pytest.approx(220.0)  # 180 + 2*20


def test_resolve_up_prediction_loss_forfeits_wager(rm):
    rm.place_prediction("up", 20.0, 100.0, 60.0, now=1000.0)  # balance 180
    pred = rm.resolve_prediction(95.0)
    assert pred.outcome == "loss"
    assert rm.balance == pytest.approx(180.0)  # wager already gone, no refund


def test_resolve_down_prediction_win(rm):
    rm.place_prediction("down", 20.0, 100.0, 60.0, now=1000.0)
    pred = rm.resolve_prediction(95.0)
    assert pred.outcome == "win"


def test_resolve_exact_prediction_within_tolerance_wins(rm):
    rm.place_prediction("exact", 20.0, 100.0, 60.0, now=1000.0)
    pred = rm.resolve_prediction(100.05)  # within 0.1% of 100
    assert pred.outcome == "win"


def test_resolve_exact_prediction_outside_tolerance_loses(rm):
    rm.place_prediction("exact", 20.0, 100.0, 60.0, now=1000.0)
    pred = rm.resolve_prediction(101.0)  # well outside 0.1% of 100
    assert pred.outcome == "loss"


def test_resolve_prediction_rejects_when_none_pending(rm):
    with pytest.raises(ValueError):
        rm.resolve_prediction(100.0)


def test_place_prediction_allowed_again_after_resolution(rm):
    rm.place_prediction("up", 20.0, 100.0, 60.0, now=1000.0)
    rm.resolve_prediction(105.0)
    rm.place_prediction("down", 10.0, 220.0, 30.0, now=1100.0)  # should not raise


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
