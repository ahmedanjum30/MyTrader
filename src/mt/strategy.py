"""Strategies — both per-symbol (Signal) and portfolio-level (target_weights).

Two interfaces live here:
  - Signal-based (legacy, per-symbol): SmaCrossover. Engine calls
    `signal(symbol, bars, qty) -> Signal`. Good for indicator-driven
    strategies that don't need to see the full universe at once.
  - Weights-based (portfolio-level): TrendFilter, Momentum,
    QuarterlyEqualWeight. Engine calls
    `target_weights(panel_through_today, today, extra) -> dict[str,float] | None`.
    Returning None means "no rebalance today, leave positions alone."
    Required for any strategy that ranks names against each other,
    gates on a non-universe market signal (SPY), or schedules by calendar.

The engine dispatches on whichever method the strategy implements.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class Signal:
    symbol: str
    action: str   # "BUY", "SELL", or "HOLD"
    reason: str


@dataclass(frozen=True)
class SmaCrossover:
    fast: int = 10
    slow: int = 30

    def signal(self, symbol: str, bars: pd.DataFrame, current_qty: float) -> Signal:
        if len(bars) < self.slow + 1:
            return Signal(symbol, "HOLD", f"need >= {self.slow + 1} bars, have {len(bars)}")

        close = bars["close"]
        fast = close.rolling(self.fast).mean()
        slow = close.rolling(self.slow).mean()

        # Crossover: fast was <= slow yesterday, fast > slow today → BUY (and vice-versa).
        crossed_up = fast.iloc[-2] <= slow.iloc[-2] and fast.iloc[-1] > slow.iloc[-1]
        crossed_down = fast.iloc[-2] >= slow.iloc[-2] and fast.iloc[-1] < slow.iloc[-1]

        if crossed_up and current_qty == 0:
            return Signal(symbol, "BUY", f"SMA{self.fast} crossed above SMA{self.slow}")

        if crossed_down and current_qty > 0:
            return Signal(symbol, "SELL", f"SMA{self.fast} crossed below SMA{self.slow}")

        return Signal(symbol, "HOLD", "no crossover")


@dataclass
class BuyAndHold:
    """Buy each universe name equal-weight once, then never trade.

    Useful as benchmark — any active strategy should at least beat
    this risk-adjusted to justify trading costs. Implements both
    interfaces; the engine prefers `target_weights` (clean equal-weight
    sizing) when available.
    """
    _initialized: bool = field(default=False, init=False, repr=False)

    def target_weights(self, panel: dict[str, pd.DataFrame],
                       today: pd.Timestamp,
                       extra: dict[str, pd.DataFrame] | None = None
                       ) -> dict[str, float] | None:
        if self._initialized:
            return None
        self._initialized = True
        eligible = [s for s, df in panel.items() if not df.empty]
        return _equal_weights(eligible) if eligible else None

    def signal(self, symbol: str, bars: pd.DataFrame, current_qty: float) -> Signal:
        # Kept for backward compat with signal-based engine path.
        if current_qty > 0:
            return Signal(symbol, "HOLD", "already held")
        if bars.empty:
            return Signal(symbol, "HOLD", "no bars")
        return Signal(symbol, "BUY", "initial buy-and-hold position")


# ---------- Portfolio-level (target_weights) strategies ----------


def _equal_weights(symbols: list[str]) -> dict[str, float]:
    if not symbols:
        return {}
    w = 1.0 / len(symbols)
    return {s: w for s in symbols}


@dataclass
class TrendFilter:
    """Risk-on / risk-off gate driven by SPY's 200-day SMA.

    When SPY > SPY-200d-SMA: target equal weight across all available
    universe names. When SPY ≤ SPY-200d-SMA: target 0 weight (cash).

    Faber 2007 (`A Quantitative Approach to Tactical Asset Allocation`):
    a 10-month-SMA gate on SPY raised Sharpe and cut drawdowns vs
    buy-and-hold across ~80 years of data. We use 200-day (≈10-month)
    for the same idea on daily bars.
    """
    sma_window: int = 200
    market_symbol: str = "SPY"
    _last_state: bool | None = field(default=None, init=False, repr=False)

    def target_weights(self, panel: dict[str, pd.DataFrame],
                       today: pd.Timestamp,
                       extra: dict[str, pd.DataFrame] | None = None
                       ) -> dict[str, float] | None:
        if extra is None or self.market_symbol not in extra:
            return None  # no SPY data available; do nothing
        spy = extra[self.market_symbol]
        spy_so_far = spy.loc[:today]
        if len(spy_so_far) < self.sma_window:
            return None  # not enough history yet
        sma = spy_so_far["close"].rolling(self.sma_window).mean().iloc[-1]
        last_close = spy_so_far["close"].iloc[-1]
        risk_on = bool(last_close > sma)

        # Only emit weights on regime change to avoid daily churn.
        if self._last_state is not None and risk_on == self._last_state:
            return None
        self._last_state = risk_on

        if not risk_on:
            return {s: 0.0 for s in panel}  # exit everything → cash
        # Equal weight across symbols with data through today.
        eligible = [s for s, df in panel.items() if today in df.index]
        return _equal_weights(eligible)


@dataclass
class Momentum:
    """Top-N by trailing 12-month return, rebalanced monthly.

    Standard cross-sectional momentum (Jegadeesh-Titman 1993).
    Rank universe by total return from t-252 to today (252 trading
    days ≈ 12 months). Hold top N equal-weight until next month-end.
    """
    top_n: int = 8
    lookback: int = 252
    _last_month: int | None = field(default=None, init=False, repr=False)

    def target_weights(self, panel: dict[str, pd.DataFrame],
                       today: pd.Timestamp,
                       extra: dict[str, pd.DataFrame] | None = None
                       ) -> dict[str, float] | None:
        # Rebalance only on month-end (the last day we see in each month).
        # Approximation: rebalance when the next bar's month differs from today's.
        if today.month == self._last_month:
            return None
        self._last_month = today.month

        scores: list[tuple[str, float]] = []
        for sym, df in panel.items():
            df_so_far = df.loc[:today]
            if len(df_so_far) < self.lookback + 1:
                continue
            ret = (df_so_far["close"].iloc[-1] / df_so_far["close"].iloc[-self.lookback]) - 1
            scores.append((sym, float(ret)))
        if not scores:
            return None
        scores.sort(key=lambda x: x[1], reverse=True)
        winners = [s for s, _ in scores[:self.top_n]]
        return _equal_weights(winners)


def _inverse_vol_weights(panel: dict[str, pd.DataFrame], today: pd.Timestamp,
                         window: int = 60) -> dict[str, float]:
    """Weight each symbol inversely proportional to its trailing realized vol.

    Symbols with insufficient history are excluded. Resulting weights
    sum to 1.0 (fully invested across eligible names; remainder is 0).
    """
    import numpy as np
    inv: dict[str, float] = {}
    for sym, df in panel.items():
        df_so_far = df.loc[:today]
        if len(df_so_far) < window + 2:
            continue
        rets = df_so_far["close"].pct_change().dropna().tail(window)
        vol = float(rets.std()) * np.sqrt(252)
        if vol > 0:
            inv[sym] = 1.0 / vol
    total = sum(inv.values())
    if total <= 0:
        return {}
    return {s: w / total for s, w in inv.items()}


@dataclass
class InverseVolWeighted:
    """Quarterly-rebalanced portfolio with inverse-volatility position sizing.

    Each name's weight is proportional to 1/(60-day realized vol),
    normalized so weights sum to 1. Low-vol names get bigger positions;
    high-vol names get smaller. Captures the low-vol anomaly
    (Frazzini-Pedersen 2014, `Betting Against Beta`) without leverage —
    halal-compatible because we never short or lever.
    """
    vol_window: int = 60
    _last_quarter: tuple[int, int] | None = field(default=None, init=False, repr=False)

    def target_weights(self, panel: dict[str, pd.DataFrame],
                       today: pd.Timestamp,
                       extra: dict[str, pd.DataFrame] | None = None
                       ) -> dict[str, float] | None:
        quarter = (today.year, (today.month - 1) // 3)
        if quarter == self._last_quarter:
            return None
        self._last_quarter = quarter
        weights = _inverse_vol_weights(panel, today, self.vol_window)
        return weights or None


@dataclass
class InverseVolTrendGated:
    """Inverse-vol weights when SPY > SPY-200d-SMA, all-cash otherwise.

    Combines drawdown protection (trend gate) with risk-parity sizing.
    Rebalances at quarter-ends OR on regime flips. Halal-friendly: cash
    is the defensive asset, no shorting, no leverage.
    """
    vol_window: int = 60
    sma_window: int = 200
    market_symbol: str = "SPY"
    _last_quarter: tuple[int, int] | None = field(default=None, init=False, repr=False)
    _last_state: bool | None = field(default=None, init=False, repr=False)

    def target_weights(self, panel: dict[str, pd.DataFrame],
                       today: pd.Timestamp,
                       extra: dict[str, pd.DataFrame] | None = None
                       ) -> dict[str, float] | None:
        if extra is None or self.market_symbol not in extra:
            return None
        spy = extra[self.market_symbol].loc[:today]
        if len(spy) < self.sma_window:
            return None
        sma = spy["close"].rolling(self.sma_window).mean().iloc[-1]
        risk_on = bool(spy["close"].iloc[-1] > sma)

        quarter = (today.year, (today.month - 1) // 3)
        regime_changed = (self._last_state is not None
                          and risk_on != self._last_state)
        quarterly_due = quarter != self._last_quarter

        if not (regime_changed or quarterly_due):
            return None
        self._last_state = risk_on
        self._last_quarter = quarter

        if not risk_on:
            return {s: 0.0 for s in panel}
        weights = _inverse_vol_weights(panel, today, self.vol_window)
        return weights or None


def _rsi(close: pd.Series, period: int) -> pd.Series:
    """Wilder-smoothed RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    # Use simple rolling for short windows (RSI(2)); Wilder for longer.
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))


@dataclass
class SwingMeanReversion:
    """Connors-style RSI(2) mean reversion in uptrending names.

    Entry: close > 200-day SMA (uptrend filter) AND RSI(2) < 10 (extreme oversold).
    Exit: close > 5-day SMA OR RSI(2) > 70 OR -3% stop OR held 10 days.
    Sizing: 10% of equity per name; with up to ~10 concurrent, near 100% invested.

    The trend filter is critical — RSI mean reversion outside a confirmed
    uptrend frequently catches falling knives. Connors' research showed
    this version held edge across 1990s-2010s on US equities.
    """
    rsi_period: int = 2
    oversold: float = 10.0
    overbought: float = 70.0
    trend_window: int = 200
    exit_sma_window: int = 5
    max_hold_days: int = 10
    stop_loss_pct: float = 0.03
    position_weight: float = 0.10
    # Internal state — entry date + entry price per held symbol.
    _entries: dict[str, tuple[pd.Timestamp, float]] = field(
        default_factory=dict, init=False, repr=False)

    def target_weights(self, panel: dict[str, pd.DataFrame],
                       today: pd.Timestamp,
                       extra: dict[str, pd.DataFrame] | None = None
                       ) -> dict[str, float] | None:
        weights: dict[str, float] = {}
        min_bars = max(self.trend_window, self.exit_sma_window) + self.rsi_period + 1

        for sym, df in panel.items():
            df_so_far = df.loc[:today]
            if len(df_so_far) < min_bars:
                weights[sym] = 0.0
                continue
            close = df_so_far["close"]
            current_close = float(close.iloc[-1])
            trend_sma = float(close.rolling(self.trend_window).mean().iloc[-1])
            exit_sma = float(close.rolling(self.exit_sma_window).mean().iloc[-1])
            rsi_now = float(_rsi(close, self.rsi_period).iloc[-1])

            if sym in self._entries:
                entry_date, entry_price = self._entries[sym]
                days_held = (today - entry_date).days
                exit_now = (
                    current_close < entry_price * (1 - self.stop_loss_pct) or
                    days_held >= self.max_hold_days or
                    current_close > exit_sma or
                    rsi_now > self.overbought
                )
                if exit_now:
                    weights[sym] = 0.0
                    del self._entries[sym]
                else:
                    weights[sym] = self.position_weight
            else:
                if current_close > trend_sma and rsi_now < self.oversold:
                    weights[sym] = self.position_weight
                    self._entries[sym] = (today, current_close)
                else:
                    weights[sym] = 0.0
        return weights


@dataclass
class SwingBreakout:
    """Donchian N-day-high breakout, swing-trade exits.

    Entry: today's close > prior N-day high (excludes today's bar).
    Exit: +10% profit target OR -5% stop OR 10-day time stop OR
          break of 10-day low (trail).
    Sizing: 10% of equity per name.

    Captures momentum impulses. Best in trending environments;
    whipsaws in choppy markets. Combine with a market regime gate
    in production.
    """
    breakout_window: int = 20
    trail_window: int = 10
    profit_target_pct: float = 0.10
    stop_loss_pct: float = 0.05
    max_hold_days: int = 10
    position_weight: float = 0.10
    _entries: dict[str, tuple[pd.Timestamp, float]] = field(
        default_factory=dict, init=False, repr=False)

    def target_weights(self, panel: dict[str, pd.DataFrame],
                       today: pd.Timestamp,
                       extra: dict[str, pd.DataFrame] | None = None
                       ) -> dict[str, float] | None:
        weights: dict[str, float] = {}
        min_bars = max(self.breakout_window, self.trail_window) + 2

        for sym, df in panel.items():
            df_so_far = df.loc[:today]
            if len(df_so_far) < min_bars:
                weights[sym] = 0.0
                continue
            close = df_so_far["close"]
            high = df_so_far["high"]
            low = df_so_far["low"]
            current_close = float(close.iloc[-1])
            # Prior N-day high, excluding today.
            n_high = float(high.iloc[-self.breakout_window - 1:-1].max())
            n_low = float(low.iloc[-self.trail_window - 1:-1].min())

            if sym in self._entries:
                entry_date, entry_price = self._entries[sym]
                days_held = (today - entry_date).days
                exit_now = (
                    current_close < entry_price * (1 - self.stop_loss_pct) or
                    current_close > entry_price * (1 + self.profit_target_pct) or
                    days_held >= self.max_hold_days or
                    current_close < n_low
                )
                if exit_now:
                    weights[sym] = 0.0
                    del self._entries[sym]
                else:
                    weights[sym] = self.position_weight
            else:
                if current_close > n_high:
                    weights[sym] = self.position_weight
                    self._entries[sym] = (today, current_close)
                else:
                    weights[sym] = 0.0
        return weights


@dataclass
class QuarterlyEqualWeight:
    """Hold the universe equal-weight, rebalance to equal weight quarterly.

    Equal-weight indices have historically outperformed cap-weighted
    indices by 1-2%/year due to the rebalancing premium (forced
    selling of recent winners + buying of recent losers). Almost-passive.
    """
    _last_quarter: tuple[int, int] | None = field(default=None, init=False, repr=False)

    def target_weights(self, panel: dict[str, pd.DataFrame],
                       today: pd.Timestamp,
                       extra: dict[str, pd.DataFrame] | None = None
                       ) -> dict[str, float] | None:
        quarter = (today.year, (today.month - 1) // 3)
        if quarter == self._last_quarter:
            return None
        self._last_quarter = quarter
        eligible = [s for s, df in panel.items() if today in df.index]
        return _equal_weights(eligible)
