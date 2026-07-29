"""Account-state and risk-rule enforcement for the trading bot.

Implements, independent of any exchange or strategy code:
  - Global circuit breaker (20% max drawdown from initial balance)
  - Progressive position sizing tied to the current losing streak
  - Mandatory cool-off after 3 consecutive losses
  - Friction accounting (taker fee + estimated slippage)
"""

import time
from dataclasses import dataclass
from typing import Optional

from config import Config


@dataclass
class TradeResult:
    pnl: float
    timestamp: float


class RiskManager:
    def __init__(self, config=Config, initial_balance: Optional[float] = None):
        self.config = config
        self.balance = initial_balance if initial_balance is not None else config.INITIAL_BALANCE
        self.circuit_breaker_floor = config.CIRCUIT_BREAKER_BALANCE
        self.consecutive_losses = 0
        self.cooloff_until: Optional[float] = None
        self.trade_history: list[TradeResult] = []

    # --- Circuit breaker -----------------------------------------------------

    @property
    def circuit_breaker_tripped(self) -> bool:
        return self.balance <= self.circuit_breaker_floor

    # --- Manual account actions (dashboard "add money / withdraw / raise
    # circuit breaker" popover) -----------------------------------------------

    def deposit(self, amount: float) -> None:
        if amount <= 0:
            raise ValueError("deposit amount must be positive")
        self.balance += amount

    def withdraw(self, amount: float) -> None:
        if amount <= 0:
            raise ValueError("withdrawal amount must be positive")
        if amount > self.balance:
            raise ValueError("cannot withdraw more than the current balance")
        self.balance -= amount

    def raise_circuit_breaker_floor(self, amount: float) -> None:
        """Manually raises the halt floor -- e.g. to lock in more of the
        current balance as a protected cushion. Rejected if it would put the
        floor at or above the current balance, since that would trip the
        breaker immediately."""
        if amount <= 0:
            raise ValueError("amount must be positive")
        new_floor = self.circuit_breaker_floor + amount
        if new_floor >= self.balance:
            raise ValueError("circuit breaker floor cannot reach or exceed the current balance")
        self.circuit_breaker_floor = new_floor

    # --- Position sizing -------------------------------------------------------

    def get_position_sizing(self) -> tuple:
        """Return (size_pct, atr_stop_multiple) for the next trade, based on
        the current consecutive-loss streak."""
        tier = min(self.consecutive_losses, self.config.MAX_SIZE_TIER_LOSSES)
        return self.config.POSITION_SIZING_TIERS[tier]

    def calculate_position_size(self, price: float, atr: float) -> dict:
        size_pct, atr_stop_multiple = self.get_position_sizing()
        allocated_capital = self.balance * size_pct
        stop_distance = atr * atr_stop_multiple
        quantity = allocated_capital / price if price > 0 else 0.0
        return {
            "size_pct": size_pct,
            "atr_stop_multiple": atr_stop_multiple,
            "allocated_capital": allocated_capital,
            "stop_distance": stop_distance,
            "quantity": quantity,
            "stop_price_long": price - stop_distance,
            "stop_price_short": price + stop_distance,
        }

    # --- Cool-off --------------------------------------------------------------

    def is_in_cooloff(self, now: Optional[float] = None) -> bool:
        now = now if now is not None else time.time()
        return self.cooloff_until is not None and now < self.cooloff_until

    def can_trade(self, now: Optional[float] = None) -> bool:
        return not self.circuit_breaker_tripped and not self.is_in_cooloff(now)

    # --- Friction ----------------------------------------------------------------

    def calculate_friction_cost(self, notional: float) -> float:
        """Taker fee + estimated slippage, applied to one fill's notional value."""
        return notional * (self.config.TAKER_FEE_PCT + self.config.SLIPPAGE_PCT)

    # --- Recording outcomes ----------------------------------------------------

    def record_trade_result(self, pnl: float, now: Optional[float] = None) -> None:
        now = now if now is not None else time.time()
        self.balance += pnl
        self.trade_history.append(TradeResult(pnl=pnl, timestamp=now))

        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        if self.consecutive_losses >= self.config.CONSECUTIVE_LOSS_COOLOFF_THRESHOLD:
            self.cooloff_until = now + self.config.COOLOFF_MINUTES * 60
            # A forced cool-off resets the streak: the bot resumes at full size.
            self.consecutive_losses = 0
