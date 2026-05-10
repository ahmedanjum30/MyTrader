"""Performance statistics from an equity curve."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Metrics:
    starting_equity: float
    ending_equity: float
    total_return: float
    cagr: float
    annual_vol: float
    sharpe: float
    max_drawdown: float
    trading_days: int


def compute(equity: pd.Series) -> Metrics:
    if equity.empty or len(equity) < 2:
        return Metrics(0, 0, 0, 0, 0, 0, 0, 0)

    starting = float(equity.iloc[0])
    ending = float(equity.iloc[-1])
    total_return = (ending / starting) - 1.0 if starting > 0 else 0.0

    days = (equity.index[-1] - equity.index[0]).days
    years = days / 365.25 if days > 0 else 0.0
    cagr = (ending / starting) ** (1 / years) - 1 if (years > 0 and starting > 0) else 0.0

    daily_returns = equity.pct_change().dropna()
    annual_vol = float(daily_returns.std() * np.sqrt(252)) if not daily_returns.empty else 0.0
    sharpe = float(daily_returns.mean() / daily_returns.std() * np.sqrt(252)) \
        if (not daily_returns.empty and daily_returns.std() > 0) else 0.0

    running_max = equity.cummax()
    drawdown = (equity / running_max) - 1.0
    max_dd = float(drawdown.min())

    return Metrics(
        starting_equity=starting,
        ending_equity=ending,
        total_return=total_return,
        cagr=cagr,
        annual_vol=annual_vol,
        sharpe=sharpe,
        max_drawdown=max_dd,
        trading_days=len(equity),
    )


def fills_summary(fills: pd.DataFrame) -> dict:
    if fills.empty:
        return {"trades": 0, "buys": 0, "sells": 0, "win_rate": float("nan"),
                "total_commission": 0.0}

    buys = fills[fills["side"] == "BUY"]
    sells = fills[fills["side"] == "SELL"]

    # Pair buys and sells per-symbol FIFO to compute round-trip P&L.
    pnls: list[float] = []
    for symbol in fills["symbol"].unique():
        sym_fills = fills[fills["symbol"] == symbol].sort_values("date")
        lots: list[tuple[float, float]] = []  # (qty, cost-per-share-incl-commission)
        for _, f in sym_fills.iterrows():
            if f["side"] == "BUY":
                lots.append((f["quantity"], f["price"] + f["commission"] / f["quantity"]))
            else:
                qty_to_sell = f["quantity"]
                proceeds_per_share = f["price"] - f["commission"] / f["quantity"]
                while qty_to_sell > 0 and lots:
                    lot_qty, lot_cost = lots[0]
                    matched = min(qty_to_sell, lot_qty)
                    pnls.append(matched * (proceeds_per_share - lot_cost))
                    qty_to_sell -= matched
                    if matched >= lot_qty:
                        lots.pop(0)
                    else:
                        lots[0] = (lot_qty - matched, lot_cost)

    wins = sum(1 for p in pnls if p > 0)
    win_rate = wins / len(pnls) if pnls else float("nan")

    return {
        "trades": int(len(fills)),
        "buys": int(len(buys)),
        "sells": int(len(sells)),
        "round_trips": int(len(pnls)),
        "win_rate": float(win_rate),
        "total_commission": float(fills["commission"].sum()),
    }
