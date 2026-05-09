"""Walk-forward backtest engine.

Loop:
  for each trading day t:
    1. Fill any orders queued at t-1's close at today's OPEN.
    2. Mark the portfolio to today's CLOSE.
    3. Generate signals on bars-up-to-today; queue orders for tomorrow's open.

This avoids look-ahead: a strategy that uses today's close decides at
end-of-day, fills tomorrow at the open. Standard EOD-strategy convention.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from ..risk import RiskLimits
from ..strategy import Signal
from ..universe import Universe
from .portfolio import Portfolio

log = logging.getLogger(__name__)


@dataclass
class QueuedOrder:
    symbol: str
    side: str        # "BUY" or "SELL"
    quantity: float


@dataclass
class BacktestResult:
    portfolio: Portfolio
    benchmark_equity: pd.Series
    trade_count: int
    skipped_no_data: int
    skipped_risk: int


def _bars_through(panel: dict[str, pd.DataFrame], symbol: str,
                  date: pd.Timestamp) -> pd.DataFrame:
    if symbol not in panel:
        return pd.DataFrame()
    df = panel[symbol]
    return df.loc[:date]


def run_backtest(panel: dict[str, pd.DataFrame], universe: Universe,
                 strategy, starting_cash: float, max_order_pct: float = 0.05,
                 max_positions: int = 15,
                 extra: dict[str, pd.DataFrame] | None = None,
                 rebalance_threshold: float = 0.005) -> BacktestResult:
    """Run a backtest. Strategy must expose either:
       - `target_weights(panel, today, extra)` (preferred, portfolio-level), or
       - `signal(symbol, bars, qty)` (per-symbol)."""

    portfolio = Portfolio(starting_cash=starting_cash)
    benchmark = _equal_weight_buy_and_hold(panel, universe, starting_cash)

    universe_panel = {s: panel[s] for s in panel if s in universe.symbols}
    if not universe_panel:
        raise RuntimeError("Panel contains no symbols in the halal universe.")
    calendar = sorted(set().union(*(df.index for df in universe_panel.values())))

    queued: list[QueuedOrder] = []
    skipped_no_data = 0
    skipped_risk = 0
    limits = RiskLimits()

    is_weights = hasattr(strategy, "target_weights")

    for today in calendar:
        # 1. Fill queued orders at today's open.
        for order in queued:
            df = universe_panel.get(order.symbol)
            if df is None or today not in df.index:
                continue
            open_px = float(df.loc[today, "open"])
            try:
                if order.side == "BUY":
                    qty = min(order.quantity, _max_buyable_qty(
                        portfolio, open_px, limits.min_cash_buffer_pct))
                    if qty > 0:
                        portfolio.buy(today, order.symbol, qty, open_px)
                else:
                    qty = min(order.quantity, portfolio.qty(order.symbol))
                    if qty > 0:
                        portfolio.sell(today, order.symbol, qty, open_px)
            except Exception as e:
                log.warning("%s fill skipped (%s %s): %s", today.date(),
                            order.side, order.symbol, e)
                skipped_risk += 1
        queued = []

        # 2. Mark to today's close.
        marks = {s: float(df.loc[today, "close"])
                 for s, df in universe_panel.items() if today in df.index}
        equity_today = portfolio.record_equity(today, marks)

        # 3. Generate orders for tomorrow's open.
        if is_weights:
            queued, dropped = _orders_from_weights(
                strategy, universe_panel, portfolio, equity_today, marks, today,
                extra=extra, threshold=rebalance_threshold,
                max_positions=max_positions,
            )
            skipped_risk += dropped
        else:
            new_queue, no_data, risk = _orders_from_signals(
                strategy, universe_panel, universe, portfolio, equity_today, today,
                max_order_pct=max_order_pct, max_positions=max_positions,
            )
            queued.extend(new_queue)
            skipped_no_data += no_data
            skipped_risk += risk

    return BacktestResult(
        portfolio=portfolio,
        benchmark_equity=benchmark,
        trade_count=len(portfolio.fills),
        skipped_no_data=skipped_no_data,
        skipped_risk=skipped_risk,
    )


def _orders_from_signals(strategy, universe_panel, universe, portfolio, equity_today,
                         today, *, max_order_pct: float, max_positions: int):
    queue: list[QueuedOrder] = []
    no_data = 0
    risk = 0
    for symbol in sorted(universe.symbols):
        bars = _bars_through(universe_panel, symbol, today)
        if bars.empty or today not in bars.index:
            no_data += 1
            continue
        current_qty = portfolio.qty(symbol)
        signal = strategy.signal(symbol, bars, current_qty)
        if signal.action == "HOLD":
            continue
        close_px = float(bars["close"].iloc[-1])
        qty = (equity_today * max_order_pct) / close_px
        if qty <= 0:
            continue
        if signal.action == "BUY":
            if portfolio.open_position_count() >= max_positions and current_qty == 0:
                risk += 1
                continue
            queue.append(QueuedOrder(symbol, "BUY", qty))
        elif signal.action == "SELL":
            qty = min(qty, current_qty)
            if qty > 0:
                queue.append(QueuedOrder(symbol, "SELL", qty))
    return queue, no_data, risk


def _orders_from_weights(strategy, universe_panel, portfolio, equity_today, marks,
                          today, *, extra, threshold: float, max_positions: int):
    """Diff target weights against current weights; queue trades for the deltas."""
    targets = strategy.target_weights(universe_panel, today, extra=extra)
    if targets is None:
        return [], 0
    queue: list[QueuedOrder] = []
    dropped = 0

    current_weights = {sym: (qty * marks.get(sym, 0)) / equity_today
                       for sym, qty in portfolio.positions.items()
                       if equity_today > 0}

    # Symbols that need to be reduced/closed (sell first to free cash).
    all_syms = set(targets) | set(portfolio.positions)
    sells: list[QueuedOrder] = []
    buys: list[QueuedOrder] = []
    for sym in all_syms:
        target_w = targets.get(sym, 0.0)
        current_w = current_weights.get(sym, 0.0)
        delta_w = target_w - current_w
        if abs(delta_w) < threshold:
            continue
        price = marks.get(sym)
        if price is None or price <= 0:
            continue
        delta_qty = (delta_w * equity_today) / price
        if delta_qty > 0:
            buys.append(QueuedOrder(sym, "BUY", delta_qty))
        else:
            sell_qty = min(-delta_qty, portfolio.qty(sym))
            if sell_qty > 0:
                sells.append(QueuedOrder(sym, "SELL", sell_qty))

    # Cap simultaneous new opens at max_positions.
    held = {s for s, q in portfolio.positions.items() if q > 0}
    new_opens = [b for b in buys if b.symbol not in held]
    if len(held) + len(new_opens) > max_positions:
        keep = max_positions - len(held)
        if keep < len(new_opens):
            dropped = len(new_opens) - keep
            new_opens = new_opens[:keep]
        buys = [b for b in buys if b.symbol in held] + new_opens

    return sells + buys, dropped


def _max_buyable_qty(portfolio: Portfolio, price: float, buffer_pct) -> float:
    """How many shares we can afford at `price` without breaking the cash buffer."""
    buffer_pct = float(buffer_pct)
    available = portfolio.cash * (1.0 - buffer_pct) - portfolio.min_commission
    if available <= 0:
        return 0.0
    return available / price


def _equal_weight_buy_and_hold(panel: dict[str, pd.DataFrame], universe: Universe,
                                starting_cash: float) -> pd.Series:
    """Benchmark: split starting_cash equally across the universe, hold to end."""
    universe_panel = {s: panel[s] for s in panel if s in universe.symbols}
    if not universe_panel:
        return pd.Series(dtype=float)

    calendar = sorted(set().union(*(df.index for df in universe_panel.values())))
    if not calendar:
        return pd.Series(dtype=float)

    first_day = calendar[0]
    eligible = {s: df for s, df in universe_panel.items() if first_day in df.index}
    if not eligible:
        return pd.Series(dtype=float)

    per_name_cash = starting_cash / len(eligible)
    qtys = {s: per_name_cash / float(df.loc[first_day, "open"]) for s, df in eligible.items()}

    equity = []
    for day in calendar:
        value = 0.0
        for s, q in qtys.items():
            df = universe_panel[s]
            if day in df.index:
                value += q * float(df.loc[day, "close"])
            else:
                # Last known close before today.
                slice_ = df.loc[:day, "close"]
                if not slice_.empty:
                    value += q * float(slice_.iloc[-1])
        equity.append((day, value))
    idx, vals = zip(*equity)
    return pd.Series(vals, index=pd.DatetimeIndex(idx), name="benchmark")
