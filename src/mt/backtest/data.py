"""Historical bar loader with on-disk parquet cache.

Data source: yfinance. Pulled once per (symbol, range), cached to
data/cache/{symbol}.parquet. Subsequent backtests read straight from
disk in milliseconds.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "cache"


def _cache_path(symbol: str) -> Path:
    return CACHE_DIR / f"{symbol.upper()}.parquet"


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase columns, ensure tz-naive DatetimeIndex, drop NaN rows."""
    df = df.copy()
    df.columns = [str(c).lower() for c in df.columns]
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index.name = "date"
    keep = ["open", "high", "low", "close", "volume"]
    df = df[[c for c in keep if c in df.columns]].dropna(how="any")
    return df


def load_symbol(symbol: str, start: str | date, end: str | date,
                refresh: bool = False) -> pd.DataFrame:
    """Return daily OHLCV bars for `symbol` between start and end (inclusive)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(symbol)

    if path.exists() and not refresh:
        cached = pd.read_parquet(path)
        cached.index = pd.to_datetime(cached.index)
        if cached.index.min() <= pd.Timestamp(start) and cached.index.max() >= pd.Timestamp(end):
            return cached.loc[str(start):str(end)]

    log.info("Fetching %s from yfinance (%s → %s)", symbol, start, end)
    raw = yf.download(symbol, start=start, end=end, auto_adjust=True,
                      progress=False, threads=False)
    if raw is None or raw.empty:
        raise RuntimeError(f"yfinance returned no data for {symbol} ({start}→{end})")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = _normalize(raw)
    df.to_parquet(path)
    return df


def load_panel(symbols: list[str], start: str | date, end: str | date,
               refresh: bool = False) -> dict[str, pd.DataFrame]:
    """Load bars for many symbols. Returns {symbol -> bars}."""
    out: dict[str, pd.DataFrame] = {}
    for s in symbols:
        try:
            out[s] = load_symbol(s, start, end, refresh=refresh)
        except Exception as e:
            log.warning("Skipping %s: %s", s, e)
    return out


def trading_days(panel: dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    """Union of all dates across the panel — the master backtest calendar."""
    if not panel:
        return pd.DatetimeIndex([])
    idx = panel[next(iter(panel))].index
    for df in panel.values():
        idx = idx.union(df.index)
    return idx.sort_values()


def survivorship_warnings(panel: dict[str, pd.DataFrame], requested_start: str | date,
                          requested_end: str | date) -> list[str]:
    """Detect symbols whose first bar is after the requested start.

    These are survivorship-bias landmines: the symbol is in *today's*
    halal universe but didn't exist (or wasn't yet listed/spun-off) at
    the requested start. Treating its later inclusion as if it were a
    contemporaneous decision overstates the strategy's hindsight skill.
    """
    requested_start = pd.Timestamp(requested_start)
    requested_end = pd.Timestamp(requested_end)
    warnings = []
    for sym, df in panel.items():
        if df.empty:
            warnings.append(f"{sym}: no bars in requested range")
            continue
        first = df.index.min()
        last = df.index.max()
        # Tolerate up to 5 days of missing leading data (holidays, weekends).
        if first > requested_start + pd.Timedelta(days=5):
            warnings.append(
                f"{sym}: first bar {first.date()}, "
                f"{(first - requested_start).days} days after requested start"
            )
        if last < requested_end - pd.Timedelta(days=5):
            warnings.append(
                f"{sym}: last bar {last.date()}, "
                f"{(requested_end - last).days} days before requested end"
            )
    return warnings


def split_walkforward(panel: dict[str, pd.DataFrame], in_sample_pct: float = 0.6
                      ) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Split each symbol's bars into IS / OOS by date.

    in_sample_pct=0.6 means the first 60% of dates are IS, last 40% are OOS.
    Strategy params should only ever be tuned on IS; OOS is the test.
    """
    cal = trading_days(panel)
    if cal.empty:
        return {}, {}
    cutoff = cal[int(len(cal) * in_sample_pct)]
    is_panel = {s: df.loc[df.index < cutoff] for s, df in panel.items()}
    oos_panel = {s: df.loc[df.index >= cutoff] for s, df in panel.items()}
    return is_panel, oos_panel
