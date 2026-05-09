"""Engine: ties data → strategy → risk → halal-screen → broker together.

One pass = one rebalance. Designed to be invoked on a schedule (cron, or
just by running `mt run` after market open).

Dispatches on the strategy's interface:
  - `target_weights(panel, today, extra)` → portfolio-level rebalance
  - `signal(symbol, bars, qty)` → per-symbol decision

`budget` lets you cap deployable dollars regardless of broker account size.
With `budget=100`, a $1M paper account is treated as $100 for sizing —
the rest of the cash sits idle.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

import pandas as pd

from .broker import IBKRBroker
from .risk import ProposedOrder, RiskLimits, RiskViolation, check_order
from .universe import NotHalalError, Universe

log = logging.getLogger(__name__)

REBALANCE_THRESHOLD = 0.005   # don't trade for <0.5% drift


@dataclass
class Engine:
    broker: IBKRBroker
    universe: Universe
    strategy: object
    limits: RiskLimits
    budget: float | None = None    # if set, cap deployable dollars at this

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
        qty = self._round_qty(target_notional / last_close)
        if qty <= 0:
            return

        if signal.action == "SELL":
            qty = min(qty, current_qty)
            if qty <= 0:
                return

        order = ProposedOrder(symbol=symbol, side=signal.action, quantity=qty,
                              estimated_price=last_close, current_position=current_qty)
        check_order(order, account, self.limits)
        self.broker.submit_market_order(symbol, signal.action, qty)

    # ---------- portfolio-level target_weights path ----------

    def _run_weights(self, account, effective_equity: Decimal) -> None:
        log.info("Fetching bars for %s symbols...", len(self.universe.symbols))
        panel: dict[str, pd.DataFrame] = {}
        for symbol in sorted(self.universe.symbols):
            try:
                bars = self.broker.historical_bars(symbol, lookback="1 Y")
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

        # Compute current weights from broker positions and last prices.
        marks = {sym: float(panel[sym]["close"].iloc[-1]) for sym in panel}
        current_qtys = {sym: self.broker.position_qty(sym) for sym in panel}
        current_value = {sym: float(current_qtys[sym]) * marks[sym] for sym in panel}
        eq_for_weights = float(effective_equity) if effective_equity > 0 else 1.0
        current_weights = {sym: v / eq_for_weights for sym, v in current_value.items()}

        # Order: SELLs first (free cash), then BUYs.
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
            delta_qty = self._round_qty(Decimal(str(delta_dollars / price)))
            if delta_qty == 0:
                continue
            if delta_qty > 0:
                buys.append((sym, delta_qty, price))
            else:
                sell_qty = min(-delta_qty, current_qtys[sym])
                if sell_qty > 0:
                    sells.append((sym, sell_qty, price))

        log.info("Rebalance plan: %d sells, %d buys", len(sells), len(buys))
        for sym, qty, price in sells + buys:
            try:
                self.universe.assert_halal(sym)
                side = "BUY" if qty > 0 else "SELL"
                abs_qty = abs(qty)
                order = ProposedOrder(
                    symbol=sym, side=side, quantity=abs_qty,
                    estimated_price=Decimal(str(price)),
                    current_position=current_qtys.get(sym, Decimal(0)),
                )
                check_order(order, account, self.limits)
                self.broker.submit_market_order(sym, side, abs_qty)
            except (NotHalalError, RiskViolation) as e:
                log.warning("Skip %s %s: %s", side, sym, e)
            except Exception:
                log.exception("Order failure for %s", sym)

    @staticmethod
    def _round_qty(qty: Decimal, decimals: int = 4) -> Decimal:
        """Round to fractional-share precision (IBKR supports 4 decimals)."""
        return qty.quantize(Decimal(10) ** -decimals)
