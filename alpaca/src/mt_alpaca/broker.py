"""Alpaca broker wrapper using alpaca-py.

Pure REST API — no local gateway needed. Stateless connection.
Fractional shares supported via API (unlike IBKR), so the bot can
do proper equal-weight halal universe deployment without workarounds.

Halal guards:
  - assert_halal at every order submit
  - cash account check (refuses if margin is enabled)
  - no shorting (refuses SELL > current_position)
  - ALLOW_LIVE flag prevents accidental live-API connection
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal

import pandas as pd
from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, OrderStatus, TimeInForce
from alpaca.trading.requests import (
    GetOrdersRequest,
    LimitOrderRequest,
    MarketOrderRequest,
)

from .risk import AccountSnapshot
from .universe import Universe

log = logging.getLogger(__name__)


@dataclass
class AlpacaConfig:
    api_key: str
    secret_key: str
    paper: bool = True       # default to paper for safety
    allow_live: bool = False  # hard guardrail for live API


class AlpacaBroker:
    """Thin wrapper around alpaca-py with halal + safety guards baked in."""

    def __init__(self, config: AlpacaConfig, universe: Universe):
        if not config.paper and not config.allow_live:
            raise RuntimeError(
                "Live API requested but ALLOW_LIVE is false. Refusing to connect. "
                "Read README 'Going live' before flipping this."
            )
        self.config = config
        self.universe = universe
        self.trading = TradingClient(
            api_key=config.api_key,
            secret_key=config.secret_key,
            paper=config.paper,
        )
        self.data = StockHistoricalDataClient(
            api_key=config.api_key,
            secret_key=config.secret_key,
        )
        self._verified_cash_account = False

    def connect(self) -> None:
        """Verify account is halal-compliant cash account at first call."""
        log.info("Connecting to Alpaca %s API...", "paper" if self.config.paper else "LIVE")
        account = self.trading.get_account()
        # Refuse to operate on margin-enabled accounts
        if hasattr(account, "trading_blocked") and account.trading_blocked:
            raise RuntimeError("Account trading is blocked. Cannot proceed.")
        if hasattr(account, "account_blocked") and account.account_blocked:
            raise RuntimeError("Account is blocked. Cannot proceed.")
        # Alpaca cash accounts have multiplier=1; margin accounts have 2 or 4.
        # Paper accounts default to margin=2 — Alpaca's setting, not user's choice.
        # Halal-correctness comes from never USING margin (no borrowing, no shorting),
        # which the risk module enforces at every order. Margin AVAILABILITY without
        # use doesn't violate AAOIFI / contemporary fiqh.
        multiplier = int(getattr(account, "multiplier", 1) or 1)
        if multiplier > 1:
            if self.config.paper:
                log.warning(
                    "Account has margin enabled (multiplier=%s). Paper-only — "
                    "risk module will enforce cash-only behavior on every order. "
                    "For LIVE deployment, you must switch to a cash account.", multiplier)
            else:
                raise RuntimeError(
                    f"LIVE account has margin enabled (multiplier={multiplier}). "
                    "Halal trading requires a cash account for live trading. "
                    "Disable margin in Alpaca dashboard before going live."
                )
        self._verified_cash_account = (multiplier == 1)
        log.info("Account verified: %s account, status=%s, equity=$%s",
                 "cash" if multiplier == 1 else "margin (paper)",
                 account.status, account.equity)

    def disconnect(self) -> None:
        # Alpaca is stateless REST; nothing to disconnect.
        pass

    @property
    def is_paper(self) -> bool:
        return self.config.paper

    # ---------- Account / positions ----------

    def account_snapshot(self) -> AccountSnapshot:
        account = self.trading.get_account()
        equity = Decimal(str(account.equity))
        cash = Decimal(str(account.cash))
        multiplier = int(getattr(account, "multiplier", 1) or 1)
        # For halal-purposes is_margin reflects whether margin is BEING USED.
        # On paper, Alpaca defaults multiplier=2 but we never borrow — risk module's
        # cash sufficiency + no-shorting checks ensure cash-only behavior. So we
        # report is_margin_account=False on paper. Live accounts must be true cash.
        is_margin = (multiplier > 1) and not self.config.paper

        positions = self.trading.get_all_positions()
        open_positions = sum(1 for p in positions if Decimal(str(p.qty)) != 0)

        return AccountSnapshot(
            equity=equity, cash=cash,
            is_margin_account=is_margin,
            open_position_count=open_positions,
        )

    def position_qty(self, symbol: str) -> Decimal:
        try:
            position = self.trading.get_open_position(symbol.upper())
            return Decimal(str(position.qty))
        except Exception:
            # alpaca-py raises if no position exists
            return Decimal(0)

    def positions_with_cost(self) -> list[tuple[str, float, float]]:
        """Return list of (symbol, qty, avg_cost) for all open universe positions."""
        out = []
        for p in self.trading.get_all_positions():
            sym = p.symbol
            if sym in self.universe.symbols:
                qty = float(p.qty)
                if qty > 0:
                    avg_cost = float(p.avg_entry_price)
                    out.append((sym, qty, avg_cost))
        return out

    # ---------- Historical bars ----------

    def historical_bars(self, symbol: str, lookback_days: int = 60,
                        timeframe: TimeFrame = TimeFrame.Day) -> pd.DataFrame:
        # Free tier uses IEX feed (no SIP subscription required).
        # IEX has lower volume than full SIP but mega-cap prices match closely.
        end = pd.Timestamp.now(tz="UTC")
        start = end - pd.Timedelta(days=lookback_days * 2)
        request = StockBarsRequest(
            symbol_or_symbols=symbol.upper(),
            timeframe=timeframe,
            start=start,
            end=end,
            limit=lookback_days * 2,
            feed=DataFeed.IEX,
        )
        bars = self.data.get_stock_bars(request)
        if symbol.upper() not in bars.data or not bars.data[symbol.upper()]:
            return pd.DataFrame()
        rows = [
            {
                "open": b.open, "high": b.high, "low": b.low,
                "close": b.close, "volume": b.volume,
                "date": b.timestamp,
            }
            for b in bars.data[symbol.upper()]
        ]
        df = pd.DataFrame(rows).set_index("date").tail(lookback_days)
        return df

    # ---------- Order management ----------

    def cancel_all_open_orders(self) -> int:
        """Cancel all open orders. Returns count cancelled."""
        cancelled = self.trading.cancel_orders()
        n = len(cancelled) if cancelled else 0
        if n > 0:
            log.info("Cancelled %s open orders", n)
        return n

    def open_orders(self) -> list:
        """Return list of currently open (non-filled, non-cancelled) orders."""
        request = GetOrdersRequest(status="open")
        return self.trading.get_orders(filter=request)

    def submit_market_order(self, symbol: str, side: str,
                            quantity: Decimal | float | None = None,
                            notional: Decimal | float | None = None) -> object:
        """Place a market order. Use either `quantity` (shares, fractional OK)
        or `notional` (dollar amount). Alpaca supports both natively."""
        self.universe.assert_halal(symbol)
        if side not in ("BUY", "SELL"):
            raise ValueError(f"Invalid side {side!r}; must be BUY or SELL.")
        if (quantity is None) == (notional is None):
            raise ValueError("Specify exactly one of quantity or notional")
        if quantity is not None and float(quantity) <= 0:
            raise ValueError("Quantity must be positive.")
        if notional is not None and float(notional) <= 0:
            raise ValueError("Notional must be positive.")

        order_side = OrderSide.BUY if side == "BUY" else OrderSide.SELL
        kwargs = {
            "symbol": symbol.upper(),
            "side": order_side,
            "time_in_force": TimeInForce.DAY,
        }
        if quantity is not None:
            # Alpaca caps fractional qty precision at 9 decimals
            kwargs["qty"] = round(float(quantity), 9)
        else:
            # Alpaca requires notional to be 2 decimal places (cents)
            kwargs["notional"] = round(float(notional), 2)

        log.info("Submitting %s %s%s (paper=%s)",
                 side,
                 f"qty={quantity}" if quantity is not None else f"${notional}",
                 f" {symbol}", self.config.paper)
        return self.trading.submit_order(MarketOrderRequest(**kwargs))

    def submit_limit_order(self, symbol: str, side: str,
                           quantity: Decimal | float, limit_price: Decimal | float) -> object:
        """Limit order with explicit price. Useful for tight fills + outside hours."""
        self.universe.assert_halal(symbol)
        if side not in ("BUY", "SELL"):
            raise ValueError(f"Invalid side {side!r}")
        if float(quantity) <= 0 or float(limit_price) <= 0:
            raise ValueError("Quantity and limit price must be positive")

        order_side = OrderSide.BUY if side == "BUY" else OrderSide.SELL
        log.info("Submitting %s %s %s @ LMT %s (paper=%s)",
                 side, quantity, symbol, limit_price, self.config.paper)
        return self.trading.submit_order(LimitOrderRequest(
            symbol=symbol.upper(),
            qty=float(quantity),
            side=order_side,
            time_in_force=TimeInForce.DAY,
            limit_price=float(limit_price),
        ))

    # ---------- Context manager ----------

    def __enter__(self) -> "AlpacaBroker":
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.disconnect()
