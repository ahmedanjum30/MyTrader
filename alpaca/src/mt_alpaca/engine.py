"""Alpaca live engine — same architecture as IBKR engine but uses
fractional shares natively (since Alpaca's API supports them).

Dispatches based on strategy interface:
  - target_weights(panel, today, extra) for portfolio-level strategies
  - signal(symbol, bars, qty) for legacy per-symbol strategies

`budget` lets you cap deployable dollars regardless of paper account size.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

import pandas as pd

from .broker import AlpacaBroker
from .risk import ProposedOrder, RiskLimits, RiskViolation, check_order
from .universe import NotHalalError, Universe

log = logging.getLogger(__name__)

REBALANCE_THRESHOLD = 0.005   # don't trade for <0.5% drift


@dataclass
class Engine:
    broker: AlpacaBroker
    universe: Universe
    strategy: object
    limits: RiskLimits
    budget: float | None = None

    def run_once(self) -> None:
        account = self.broker.account_snapshot()
        effective_equity = (Decimal(str(self.budget))
                            if self.budget is not None and self.budget > 0
                            else account.equity)
        log.info("Account: equity=%s cash=%s open=%s | effective_equity=%s",
                 account.equity, account.cash, account.open_position_count,
                 effective_equity)

        if hasattr(self.strategy, "target_weights"):
            self._run_weights(account, effective_equity)
        else:
            self._run_signals(account, effective_equity)

    # ---------- per-symbol signal path (legacy) ----------

    def _run_signals(self, account, effective_equity: Decimal) -> None:
        for symbol in sorted(self.universe.symbols):
            try:
                self._step_signal(symbol, account, effective_equity)
            except (NotHalalError, RiskViolation) as e:
                log.warning("Skip %s: %s", symbol, e)
            except Exception:
                log.exception("Unhandled error processing %s", symbol)

    def _step_signal(self, symbol: str, account, effective_equity: Decimal) -> None:
        self.universe.assert_halal(symbol)
        bars = self.broker.historical_bars(symbol)
        if bars.empty:
            log.info("No bars for %s; skipping", symbol)
            return

        current_qty = self.broker.position_qty(symbol)
        signal = self.strategy.signal(symbol, bars, float(current_qty))
        log.info("%s: %s (%s)", symbol, signal.action, signal.reason)
        if signal.action == "HOLD":
            return

        last_close = Decimal(str(bars["close"].iloc[-1]))
        target_notional = effective_equity * self.limits.max_order_pct_of_equity
        # Alpaca supports fractional, so we keep decimals (no integer rounding)
        qty = (target_notional / last_close).quantize(Decimal("0.0001"))
        if qty <= 0:
            return

        if signal.action == "SELL":
            qty = min(qty, current_qty)
            if qty <= 0:
                return

        order = ProposedOrder(symbol=symbol, side=signal.action, quantity=qty,
                              estimated_price=last_close, current_position=current_qty)
        check_order(order, account, self.limits)
        self.broker.submit_market_order(symbol, signal.action, quantity=qty)

    # ---------- target_weights path ----------

    def _run_weights(self, account, effective_equity: Decimal) -> None:
        log.info("Fetching bars for %s symbols...", len(self.universe.symbols))
        panel: dict[str, pd.DataFrame] = {}
        for symbol in sorted(self.universe.symbols):
            try:
                bars = self.broker.historical_bars(symbol, lookback_days=250)
            except Exception as e:
                log.warning("Could not fetch bars for %s: %s", symbol, e)
                continue
            if not bars.empty:
                panel[symbol] = bars

        if not panel:
            log.error("No bars available; aborting")
            return

        today = pd.Timestamp(datetime.now()).normalize()
        targets = self.strategy.target_weights(panel, today, extra=None)
        if targets is None:
            log.info("Strategy says no rebalance today.")
            return

        marks = {sym: float(panel[sym]["close"].iloc[-1]) for sym in panel}
        current_qtys = {sym: self.broker.position_qty(sym) for sym in panel}
        current_value = {sym: float(current_qtys[sym]) * marks[sym] for sym in panel}
        eq_for_weights = float(effective_equity) if effective_equity > 0 else 1.0
        current_weights = {sym: v / eq_for_weights for sym, v in current_value.items()}

        sells = []
        buys = []
        for sym in sorted(set(targets) | set(current_qtys)):
            target_w = targets.get(sym, 0.0)
            current_w = current_weights.get(sym, 0.0)
            delta_w = target_w - current_w
            if abs(delta_w) < REBALANCE_THRESHOLD:
                continue
            price = marks.get(sym)
            if price is None or price <= 0:
                continue
            delta_dollars = delta_w * float(effective_equity)
            # Alpaca: use notional (dollar amount) on BUYs for clean fractional handling
            if delta_dollars > 0:
                buys.append((sym, delta_dollars, price))
            else:
                sell_qty = min(abs(delta_dollars) / price, float(current_qtys[sym]))
                if sell_qty > 0:
                    sells.append((sym, sell_qty, price))

        log.info("Rebalance plan: %d sells, %d buys", len(sells), len(buys))

        # Sells first to free up cash, then buys
        for sym, sell_qty, price in sells:
            try:
                self.universe.assert_halal(sym)
                order = ProposedOrder(
                    symbol=sym, side="SELL",
                    quantity=Decimal(str(sell_qty)),
                    estimated_price=Decimal(str(price)),
                    current_position=current_qtys.get(sym, Decimal(0)),
                )
                check_order(order, account, self.limits)
                self.broker.submit_market_order(sym, "SELL", quantity=sell_qty)
            except (NotHalalError, RiskViolation) as e:
                log.warning("Skip SELL %s: %s", sym, e)
            except Exception:
                log.exception("SELL failure for %s", sym)

        for sym, buy_dollars, price in buys:
            try:
                self.universe.assert_halal(sym)
                qty_estimate = Decimal(str(buy_dollars / price))
                order = ProposedOrder(
                    symbol=sym, side="BUY",
                    quantity=qty_estimate,
                    estimated_price=Decimal(str(price)),
                    current_position=current_qtys.get(sym, Decimal(0)),
                )
                check_order(order, account, self.limits)
                self.broker.submit_market_order(sym, "BUY", notional=buy_dollars)
            except (NotHalalError, RiskViolation) as e:
                log.warning("Skip BUY %s: %s", sym, e)
            except Exception:
                log.exception("BUY failure for %s", sym)
