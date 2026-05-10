"""Risk + halal-trading guards.

Most of these are *negative* rules: things we refuse to do. They apply
before any order leaves the engine.
"""

from dataclasses import dataclass
from decimal import Decimal


class RiskViolation(ValueError):
    """Raised when a proposed order violates a risk or halal trading rule."""


@dataclass(frozen=True)
class RiskLimits:
    max_position_pct_of_equity: Decimal = Decimal("0.10")  # 10% per name
    max_order_pct_of_equity: Decimal = Decimal("0.05")     # 5% per single order
    min_cash_buffer_pct: Decimal = Decimal("0.02")          # keep 2% cash idle
    max_open_positions: int = 15


@dataclass(frozen=True)
class AccountSnapshot:
    equity: Decimal
    cash: Decimal
    is_margin_account: bool
    open_position_count: int


@dataclass(frozen=True)
class ProposedOrder:
    symbol: str
    side: str            # "BUY" or "SELL"
    quantity: Decimal
    estimated_price: Decimal
    current_position: Decimal  # signed; negative would be a short

    @property
    def notional(self) -> Decimal:
        return self.quantity * self.estimated_price


def check_order(order: ProposedOrder, account: AccountSnapshot, limits: RiskLimits) -> None:
    """Raise RiskViolation if the order breaks any rule. Halal rules first."""

    # --- Halal-trading hard rules ---
    if account.is_margin_account:
        raise RiskViolation(
            "Account is a margin account. Halal trading requires a cash account "
            "(no interest-bearing borrowing). Refusing to trade."
        )

    if order.side == "SELL":
        # No shorting: can only sell shares we currently own.
        if order.current_position < order.quantity:
            raise RiskViolation(
                f"Refusing SELL {order.quantity} {order.symbol}: current position is "
                f"{order.current_position}. Short selling is not permitted."
            )

    if order.quantity <= 0:
        raise RiskViolation("Order quantity must be positive.")

    # --- Position-sizing rules ---
    if account.equity <= 0:
        raise RiskViolation("Account equity is non-positive; cannot size orders.")

    order_pct = order.notional / account.equity
    if order_pct > limits.max_order_pct_of_equity:
        raise RiskViolation(
            f"Order notional {order.notional} is {order_pct:.1%} of equity, "
            f"exceeds max_order_pct_of_equity {limits.max_order_pct_of_equity:.1%}."
        )

    if order.side == "BUY":
        # Cash must cover notional plus the configured buffer.
        required_cash = order.notional + (account.equity * limits.min_cash_buffer_pct)
        if account.cash < required_cash:
            raise RiskViolation(
                f"Insufficient cash for BUY: need {required_cash} (incl. buffer), "
                f"have {account.cash}."
            )

        if account.open_position_count >= limits.max_open_positions:
            # Allow adding to an existing position, but not opening a new one.
            if order.current_position == 0:
                raise RiskViolation(
                    f"Already at max_open_positions ({limits.max_open_positions}); "
                    f"cannot open new position in {order.symbol}."
                )
