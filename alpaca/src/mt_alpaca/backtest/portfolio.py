"""Backtest portfolio: cash account, no shorting, conservation invariant.

The single most important file in the backtester. If accounting is wrong
here, every metric downstream is wrong.

Invariants enforced:
  - cash >= 0 (cash account, no margin)
  - position_qty >= 0 for every symbol (no shorting)
  - equity == cash + sum(qty * mark_price)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


class PortfolioError(RuntimeError):
    """Raised when a trade or mark-to-market would break an invariant."""


@dataclass
class Fill:
    date: pd.Timestamp
    symbol: str
    side: str        # "BUY" or "SELL"
    quantity: float
    price: float
    commission: float

    @property
    def notional(self) -> float:
        return self.quantity * self.price


@dataclass
class Portfolio:
    starting_cash: float
    cash: float = field(init=False)
    positions: dict[str, float] = field(default_factory=dict)
    fills: list[Fill] = field(default_factory=list)
    equity_curve: list[tuple[pd.Timestamp, float]] = field(default_factory=list)

    # Fill model parameters
    slippage_bps: float = 5.0
    commission_per_share: float = 0.0035   # IBKR Tiered
    min_commission: float = 1.0             # IBKR per-order minimum

    def __post_init__(self) -> None:
        self.cash = float(self.starting_cash)

    def qty(self, symbol: str) -> float:
        return self.positions.get(symbol.upper(), 0.0)

    def open_position_count(self) -> int:
        return sum(1 for q in self.positions.values() if q > 0)

    def commission_for(self, quantity: float) -> float:
        return max(self.min_commission, self.commission_per_share * quantity)

    def slipped_price(self, side: str, mid: float) -> float:
        bps = self.slippage_bps / 10_000.0
        return mid * (1 + bps) if side == "BUY" else mid * (1 - bps)

    def buy(self, date: pd.Timestamp, symbol: str, quantity: float, mid_price: float) -> Fill:
        if quantity <= 0:
            raise PortfolioError(f"BUY quantity must be positive (got {quantity})")
        price = self.slipped_price("BUY", mid_price)
        commission = self.commission_for(quantity)
        cost = price * quantity + commission
        if cost > self.cash + 1e-9:
            raise PortfolioError(
                f"BUY {quantity} {symbol} @ {price:.2f} costs {cost:.2f} but cash is {self.cash:.2f}"
            )
        self.cash -= cost
        self.positions[symbol] = self.qty(symbol) + quantity
        fill = Fill(date, symbol, "BUY", quantity, price, commission)
        self.fills.append(fill)
        return fill

    def sell(self, date: pd.Timestamp, symbol: str, quantity: float, mid_price: float) -> Fill:
        if quantity <= 0:
            raise PortfolioError(f"SELL quantity must be positive (got {quantity})")
        held = self.qty(symbol)
        if quantity > held + 1e-9:
            raise PortfolioError(
                f"SELL {quantity} {symbol} but only hold {held} (no shorting in cash account)"
            )
        price = self.slipped_price("SELL", mid_price)
        commission = self.commission_for(quantity)
        proceeds = price * quantity - commission
        self.cash += proceeds
        new_qty = held - quantity
        if new_qty <= 1e-9:
            self.positions.pop(symbol, None)
        else:
            self.positions[symbol] = new_qty
        fill = Fill(date, symbol, "SELL", quantity, price, commission)
        self.fills.append(fill)
        return fill

    def equity(self, mark_prices: dict[str, float]) -> float:
        positions_value = sum(qty * mark_prices.get(sym, 0.0)
                              for sym, qty in self.positions.items())
        return self.cash + positions_value

    def record_equity(self, date: pd.Timestamp, mark_prices: dict[str, float]) -> float:
        eq = self.equity(mark_prices)
        if self.cash < -1e-9:
            raise PortfolioError(f"Cash invariant broken on {date}: cash={self.cash}")
        for sym, q in self.positions.items():
            if q < -1e-9:
                raise PortfolioError(f"Position invariant broken on {date}: {sym}={q}")
        self.equity_curve.append((date, eq))
        return eq

    def equity_series(self) -> pd.Series:
        if not self.equity_curve:
            return pd.Series(dtype=float)
        idx, vals = zip(*self.equity_curve)
        return pd.Series(vals, index=pd.DatetimeIndex(idx), name="equity")

    def fills_df(self) -> pd.DataFrame:
        if not self.fills:
            return pd.DataFrame(columns=["date", "symbol", "side", "quantity",
                                         "price", "commission", "notional"])
        return pd.DataFrame([{
            "date": f.date, "symbol": f.symbol, "side": f.side,
            "quantity": f.quantity, "price": f.price,
            "commission": f.commission, "notional": f.notional,
        } for f in self.fills])
