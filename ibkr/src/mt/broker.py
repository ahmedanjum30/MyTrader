"""IBKR broker wrapper using ib_async.

Connects to a running TWS or IB Gateway. Exposes only the operations the
engine needs: account snapshot, positions, historical bars, and submitting
equity orders for symbols already cleared by the halal guard.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

import pandas as pd
from ib_async import IB, LimitOrder, MarketOrder, Stock, util

from .risk import AccountSnapshot
from .universe import Universe

log = logging.getLogger(__name__)


@dataclass
class IBKRConfig:
    host: str
    port: int
    client_id: int
    account: str | None = None
    allow_live: bool = False


class IBKRBroker:
    """Thin wrapper around ib_async with halal + safety guards baked in."""

    PAPER_PORTS = {7497, 4002}  # TWS-paper, Gateway-paper

    def __init__(self, config: IBKRConfig, universe: Universe):
        self.config = config
        self.universe = universe
        self.ib = IB()
        self._is_paper = config.port in self.PAPER_PORTS

        if not self._is_paper and not config.allow_live:
            raise RuntimeError(
                f"Port {config.port} is a live-trading port but ALLOW_LIVE is false. "
                "Refusing to connect. Read README 'Going live' before flipping this."
            )

    def connect(self) -> None:
        log.info("Connecting to IBKR %s:%s (client_id=%s, paper=%s)",
                 self.config.host, self.config.port, self.config.client_id, self._is_paper)
        self.ib.connect(self.config.host, self.config.port, clientId=self.config.client_id)

        # Verify cash account; refuse to operate on a margin account.
        for v in self.ib.accountValues(self.config.account):
            if v.tag == "AccountType" and "margin" in v.value.lower():
                self.disconnect()
                raise RuntimeError(
                    f"IBKR account type is '{v.value}'. Halal trading requires a "
                    "cash account. Change account type or use a different account."
                )

    def disconnect(self) -> None:
        if self.ib.isConnected():
            self.ib.disconnect()

    def account_snapshot(self) -> AccountSnapshot:
        values = {v.tag: v for v in self.ib.accountValues(self.config.account)}
        equity = Decimal(values["NetLiquidation"].value) if "NetLiquidation" in values else Decimal(0)
        cash = Decimal(values["TotalCashValue"].value) if "TotalCashValue" in values else Decimal(0)
        account_type = values.get("AccountType")
        is_margin = bool(account_type and "margin" in account_type.value.lower())

        positions = self.ib.positions(self.config.account)
        open_positions = sum(1 for p in positions if p.position != 0)

        return AccountSnapshot(
            equity=equity,
            cash=cash,
            is_margin_account=is_margin,
            open_position_count=open_positions,
        )

    def position_qty(self, symbol: str) -> Decimal:
        for p in self.ib.positions(self.config.account):
            if p.contract.symbol == symbol.upper():
                return Decimal(str(p.position))
        return Decimal(0)

    def historical_bars(self, symbol: str, lookback: str = "60 D",
                        bar_size: str = "1 day") -> pd.DataFrame:
        contract = Stock(symbol.upper(), "SMART", "USD")
        self.ib.qualifyContracts(contract)
        bars = self.ib.reqHistoricalData(
            contract,
            endDateTime="",
            durationStr=lookback,
            barSizeSetting=bar_size,
            whatToShow="TRADES",
            useRTH=True,
            formatDate=1,
        )
        return util.df(bars) if bars else pd.DataFrame()

    def cancel_all_open_orders(self) -> int:
        """Cancel any open / pending orders. Returns count cancelled."""
        n = 0
        for trade in list(self.ib.openTrades()):
            try:
                self.ib.cancelOrder(trade.order)
                n += 1
            except Exception:
                log.exception("Could not cancel order %s", trade.order)
        if n > 0:
            log.info("Cancelled %s open orders", n)
        return n

    def submit_limit_order(self, symbol: str, side: str, quantity: Decimal,
                           limit_price: Decimal) -> object:
        """Limit order that doesn't require a real-time data subscription."""
        self.universe.assert_halal(symbol)
        if side not in ("BUY", "SELL"):
            raise ValueError(f"Invalid side {side!r}; must be BUY or SELL.")
        if quantity <= 0:
            raise ValueError("Quantity must be positive.")
        if limit_price <= 0:
            raise ValueError("Limit price must be positive.")

        contract = Stock(symbol.upper(), "SMART", "USD")
        self.ib.qualifyContracts(contract)
        order = LimitOrder(side, float(quantity), float(limit_price), tif="DAY")
        if self.config.account:
            order.account = self.config.account

        log.info("Submitting %s %s %s @ LMT %s (paper=%s)", side, quantity,
                 symbol, limit_price, self._is_paper)
        return self.ib.placeOrder(contract, order)

    def submit_market_order(self, symbol: str, side: str, quantity: Decimal) -> object:
        # Last-line halal check at the broker boundary. Belt + suspenders.
        self.universe.assert_halal(symbol)

        if side not in ("BUY", "SELL"):
            raise ValueError(f"Invalid side {side!r}; must be BUY or SELL.")
        if quantity <= 0:
            raise ValueError("Quantity must be positive.")

        contract = Stock(symbol.upper(), "SMART", "USD")
        self.ib.qualifyContracts(contract)
        order = MarketOrder(side, float(quantity), tif="DAY")
        if self.config.account:
            order.account = self.config.account

        log.info("Submitting %s %s %s @ MKT (paper=%s)", side, quantity, symbol, self._is_paper)
        trade = self.ib.placeOrder(contract, order)
        return trade

    @property
    def is_paper(self) -> bool:
        return self._is_paper

    def __enter__(self) -> "IBKRBroker":
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.disconnect()
