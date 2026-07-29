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


@dataclass
class Prediction:
    """A user-placed "will the price go up/down/~exact" wager -- the
    dashboard's Predict popup, not part of the bot's own AI-driven trading.
    One at a time; resolved lazily whenever anything polls RiskManager after
    target_time has passed (see server.py's /status handler)."""
    direction: str  # "up" | "down" | "exact"
    wager: float
    price_at_bet: float
    placed_at: float
    target_time: float
    resolved: bool = False
    outcome: Optional[str] = None  # "win" | "loss", set once resolved
    resolution_price: Optional[float] = None


class RiskManager:
    def __init__(self, config=Config, initial_balance: Optional[float] = None):
        self.config = config
        self.balance = initial_balance if initial_balance is not None else config.INITIAL_BALANCE
        self.circuit_breaker_floor = config.CIRCUIT_BREAKER_BALANCE
        self.consecutive_losses = 0
        self.cooloff_until: Optional[float] = None
        self.trade_history: list[TradeResult] = []
        self.pending_prediction: Optional[Prediction] = None

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

    # --- Price-prediction mini-game (dashboard "Predict" popup) -----------------
    # Even money: correct guess doubles the wager, wrong guess forfeits it.
    # "exact" wins within a +/-0.1% tolerance band of the price at bet time,
    # since landing on a literal exact price is practically impossible.

    PREDICTION_EXACT_TOLERANCE_PCT = 0.001

    def place_prediction(
        self, direction: str, wager: float, price_at_bet: float, duration_seconds: float,
        now: Optional[float] = None,
    ) -> Prediction:
        if direction not in ("up", "down", "exact"):
            raise ValueError("direction must be 'up', 'down', or 'exact'")
        if wager <= 0:
            raise ValueError("wager must be positive")
        if wager > self.balance:
            raise ValueError("wager cannot exceed the current balance")
        if duration_seconds <= 0:
            raise ValueError("duration must be positive")
        if self.pending_prediction is not None and not self.pending_prediction.resolved:
            raise ValueError("a prediction is already pending")

        now = now if now is not None else time.time()
        self.balance -= wager
        self.pending_prediction = Prediction(
            direction=direction, wager=wager, price_at_bet=price_at_bet,
            placed_at=now, target_time=now + duration_seconds,
        )
        return self.pending_prediction

    def prediction_due(self, now: Optional[float] = None) -> bool:
        now = now if now is not None else time.time()
        pred = self.pending_prediction
        return pred is not None and not pred.resolved and now >= pred.target_time

    def resolve_prediction(self, current_price: float) -> Prediction:
        pred = self.pending_prediction
        if pred is None or pred.resolved:
            raise ValueError("no pending prediction to resolve")

        tolerance = pred.price_at_bet * self.PREDICTION_EXACT_TOLERANCE_PCT
        if pred.direction == "exact":
            won = abs(current_price - pred.price_at_bet) <= tolerance
        elif pred.direction == "up":
            won = current_price > pred.price_at_bet
        else:
            won = current_price < pred.price_at_bet

        pred.resolution_price = current_price
        pred.resolved = True
        pred.outcome = "win" if won else "loss"
        if won:
            self.balance += pred.wager * 2
        return pred

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
