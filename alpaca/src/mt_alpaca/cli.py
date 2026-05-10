"""CLI for Alpaca-side bot. Entry point: `mta`.

Mirrors the IBKR `mt` command set but uses Alpaca-py.
Key differences from IBKR CLI:
  - No `mt deploy` greedy-cheapest (we have fractional, so direct buyhold works)
  - `mta run --strategy buyhold --budget 1000` does the equivalent properly
  - Status, account, list-universe behave the same
"""

import logging
import os
from pathlib import Path

import click
from dotenv import load_dotenv

from .backtest import data as bt_data
from .backtest.engine import run_backtest
from .backtest.metrics import compute as compute_metrics, fills_summary
from .broker import AlpacaBroker, AlpacaConfig
from .engine import Engine
from .risk import RiskLimits
from .strategy import (
    BuyAndHold,
    InverseVolTrendGated,
    InverseVolWeighted,
    Momentum,
    QuarterlyEqualWeight,
    SmaCrossover,
    SwingBreakout,
    SwingMeanReversion,
    TrendFilter,
)
from .universe import load_universe

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _config() -> AlpacaConfig:
    api_key = os.environ.get("ALPACA_API_KEY", "").strip()
    secret = os.environ.get("ALPACA_SECRET_KEY", "").strip()
    if not api_key or not secret:
        raise click.UsageError(
            "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in .env"
        )
    paper = os.environ.get("ALPACA_PAPER", "true").lower() == "true"
    allow_live = os.environ.get("ALLOW_LIVE", "false").lower() == "true"
    return AlpacaConfig(
        api_key=api_key,
        secret_key=secret,
        paper=paper,
        allow_live=allow_live,
    )


STRATEGY_FACTORY = {
    "sma": SmaCrossover,
    "buyhold": BuyAndHold,
    "trend": TrendFilter,
    "momentum": Momentum,
    "quarterly_ew": QuarterlyEqualWeight,
    "invvol": InverseVolWeighted,
    "invvol_trend": InverseVolTrendGated,
    "swing_meanrev": SwingMeanReversion,
    "swing_breakout": SwingBreakout,
}


def _strategy_for(name: str):
    return STRATEGY_FACTORY[name]()


def _needs_extra(name: str) -> list[str]:
    return ["SPY"] if name in {"trend", "invvol_trend"} else []


@click.group()
def cli() -> None:
    """Halal-screened equities trading bot on Alpaca (mta CLI)."""


@cli.command("list-universe")
@click.option("--config", default=None, help="Path to halal_universe.yaml")
def list_universe(config: str | None) -> None:
    """Print the loaded halal allowlist."""
    u = load_universe(Path(config)) if config else load_universe()
    click.echo(f"{len(u.symbols)} screened symbols:")
    for s in sorted(u.symbols):
        click.echo(f"  {s}")
    if u.review_required:
        click.echo("\nReview-required (not tradable until re-screened):")
        for s in sorted(u.review_required):
            click.echo(f"  {s}")


@cli.command("account")
def account_cmd() -> None:
    """Connect to Alpaca and print the account snapshot."""
    universe = load_universe()
    broker = AlpacaBroker(_config(), universe)
    broker.connect()
    snap = broker.account_snapshot()
    click.echo(f"Paper:           {broker.is_paper}")
    click.echo(f"Equity:          ${snap.equity}")
    click.echo(f"Cash:            ${snap.cash}")
    click.echo(f"Margin account:  {snap.is_margin_account}")
    click.echo(f"Open positions:  {snap.open_position_count}")


@cli.command("run")
@click.option("--dry-run", is_flag=True, help="Compute orders but submit none.")
@click.option("--strategy", "strategy_name", default="buyhold",
              type=click.Choice(list(STRATEGY_FACTORY.keys())),
              help="Strategy to run live")
@click.option("--budget", default=None, type=float,
              help="Cap deployable dollars (e.g. 100). If unset, uses full account.")
def run_cmd(dry_run: bool, strategy_name: str, budget: float | None) -> None:
    """Run one rebalance pass against Alpaca."""
    universe = load_universe()
    broker = AlpacaBroker(_config(), universe)
    broker.connect()

    if dry_run:
        broker.submit_market_order = lambda symbol, side, quantity=None, notional=None: click.echo(
            f"[DRY-RUN] would submit {side} {symbol} "
            f"{f'qty={quantity}' if quantity else f'${notional}'}"
        )

    engine = Engine(broker=broker, universe=universe,
                    strategy=_strategy_for(strategy_name),
                    limits=RiskLimits(),
                    budget=budget)
    click.echo(f"Strategy: {strategy_name}, "
               f"budget: {f'${budget:.2f}' if budget else 'full account'}, "
               f"dry_run: {dry_run}")
    engine.run_once()


@cli.command("status")
@click.option("--budget", default=None, type=float,
              help="Reference budget for weight calculations")
def status_cmd(budget: float | None) -> None:
    """Show current positions, cost basis, current value, and P&L."""
    universe = load_universe()
    broker = AlpacaBroker(_config(), universe)
    broker.connect()
    snap = broker.account_snapshot()
    click.echo(f"Account:        Paper={broker.is_paper}  "
               f"Cash account={not snap.is_margin_account}")
    click.echo(f"Equity:         ${snap.equity:,.2f}")
    click.echo(f"Cash:           ${snap.cash:,.2f}")
    click.echo(f"Open positions: {snap.open_position_count}")
    if budget:
        click.echo(f"Budget cap:     ${budget:,.2f}")

    positions = broker.positions_with_cost()
    open_orders = broker.open_orders()
    pending = [(o.symbol, o.status.value if hasattr(o.status, "value") else str(o.status),
                float(o.qty or 0), float(o.limit_price or 0))
               for o in open_orders if o.symbol in universe.symbols]

    if not positions and not pending:
        click.echo("\nNo positions or pending orders.")
        return

    if positions:
        click.echo(f"\n{'Symbol':<8}  {'Qty':>8}  {'Avg cost':>10}  "
                   f"{'Last':>10}  {'Cost basis':>12}  {'Value':>12}  "
                   f"{'P&L $':>11}  {'P&L %':>7}")
        click.echo("-" * 95)
        total_cost = 0.0
        total_value = 0.0
        for sym, qty, avg_cost in sorted(positions):
            try:
                bars = broker.historical_bars(sym, lookback_days=5)
                last = float(bars["close"].iloc[-1]) if not bars.empty else avg_cost
            except Exception:
                last = avg_cost
            cost = qty * avg_cost
            value = qty * last
            pnl = value - cost
            pnl_pct = (pnl / cost * 100) if cost > 0 else 0.0
            total_cost += cost
            total_value += value
            color = "green" if pnl >= 0 else "red"
            line = (f"{sym:<8}  {qty:>8.4f}  ${avg_cost:>9.2f}  "
                    f"${last:>9.2f}  ${cost:>11.2f}  ${value:>11.2f}  "
                    f"${pnl:>+10.2f}  {pnl_pct:>+6.2f}%")
            click.echo(click.style(line, fg=color))
        click.echo("-" * 95)
        total_pnl = total_value - total_cost
        total_pnl_pct = (total_pnl / total_cost * 100) if total_cost > 0 else 0.0
        click.echo(f"{'TOTAL':<8}  {'':>8}  {'':>10}  {'':>10}  "
                   f"${total_cost:>11.2f}  ${total_value:>11.2f}  "
                   f"${total_pnl:>+10.2f}  {total_pnl_pct:>+6.2f}%")

    if pending:
        click.echo(f"\nPending orders ({len(pending)}):")
        for sym, status, qty, lmt in sorted(pending):
            click.echo(f"  {sym:<8}  {status:>14}  qty={qty:>6.4f}  "
                       f"{f'${lmt:.2f}' if lmt > 0 else 'MKT'}")


@cli.command("deploy")
@click.option("--budget", default=1000.0, type=float, help="Total dollar budget")
@click.option("--strategy", "strategy_name", default="buyhold",
              type=click.Choice(list(STRATEGY_FACTORY.keys())))
@click.option("--dry-run", is_flag=True, help="Show plan, submit nothing")
@click.option("--cancel-pending", is_flag=True,
              help="Cancel any open orders before deploying")
def deploy_cmd(budget: float, strategy_name: str, dry_run: bool,
               cancel_pending: bool) -> None:
    """Deploy strategy with fractional shares (Alpaca-native).

    Unlike the IBKR `mt deploy` greedy-cheapest workaround, Alpaca supports
    fractional via API, so this just runs the strategy with the budget cap.
    Result: equal-weight buyhold of the FULL halal universe at $1000 budget,
    not just the cheapest 8 names.
    """
    universe = load_universe()
    broker = AlpacaBroker(_config(), universe)
    broker.connect()

    if cancel_pending:
        n = broker.cancel_all_open_orders()
        click.echo(f"Cancelled {n} pending orders.")

    if dry_run:
        broker.submit_market_order = lambda symbol, side, quantity=None, notional=None: click.echo(
            f"[DRY-RUN] would submit {side} {symbol} "
            f"{f'qty={quantity}' if quantity else f'${notional}'}"
        )

    engine = Engine(broker=broker, universe=universe,
                    strategy=_strategy_for(strategy_name),
                    limits=RiskLimits(),
                    budget=budget)
    click.echo(f"Deploying {strategy_name} with ${budget:.2f} budget on Alpaca {'paper' if broker.is_paper else 'LIVE'}")
    engine.run_once()
    click.echo("Done. Run `mta status --budget {:.0f}` to inspect.".format(budget))


# Backtest commands (same as IBKR — uses yfinance, broker-agnostic)

def _row(label: str, m) -> str:
    return (f"  {label:<22}  end=${m.ending_equity:>11,.2f}  "
            f"return={m.total_return*100:>7.2f}%  "
            f"CAGR={m.cagr*100:>6.2f}%  "
            f"Sharpe={m.sharpe:>5.2f}  "
            f"maxDD={m.max_drawdown*100:>6.2f}%")


def _run_segment(panel, universe, strategy, cash: float, label: str,
                 extra: dict | None = None) -> tuple:
    result = run_backtest(panel=panel, universe=universe,
                          strategy=strategy, starting_cash=cash, extra=extra)
    eq = result.portfolio.equity_series()
    metrics = compute_metrics(eq)
    fills_stats = fills_summary(result.portfolio.fills_df())
    return result, metrics, fills_stats


def _load_extra(symbols: list[str], start: str, end: str, refresh: bool = False) -> dict:
    if not symbols:
        return {}
    extra: dict = {}
    for s in symbols:
        try:
            extra[s] = bt_data.load_symbol(s, start, end, refresh=refresh)
        except Exception as e:
            click.echo(f"  WARNING: could not load {s}: {e}")
    return extra


@cli.command("backtest")
@click.option("--start", default="2020-01-01")
@click.option("--end", default=None)
@click.option("--cash", default=10_000.0, type=float)
@click.option("--strategy", "strategy_name", default="buyhold",
              type=click.Choice(list(STRATEGY_FACTORY.keys())))
@click.option("--refresh", is_flag=True)
def backtest_cmd(start: str, end: str | None, cash: float, strategy_name: str,
                 refresh: bool) -> None:
    """Walk-forward backtest using yfinance (broker-independent)."""
    import pandas as pd

    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    universe = load_universe()
    symbols = sorted(universe.symbols)

    click.echo(f"Loading bars for {len(symbols)} symbols ({start} → {end})...")
    panel = bt_data.load_panel(symbols, start, end, refresh=refresh)
    extra = _load_extra(_needs_extra(strategy_name), start, end, refresh=refresh)
    click.echo(f"Loaded {len(panel)} symbols.\n")

    result, metrics, fills = _run_segment(panel, universe,
                                          _strategy_for(strategy_name), cash,
                                          strategy_name, extra=extra if _needs_extra(strategy_name) else None)
    bh_result, bh_metrics, _ = _run_segment(panel, universe, BuyAndHold(), cash, "B&H")
    click.echo("=" * 110)
    click.echo(_row(strategy_name, metrics))
    click.echo(_row("buy&hold", bh_metrics))
    click.echo("=" * 110)
    edge = ((metrics.ending_equity - bh_metrics.ending_equity)
            / bh_metrics.ending_equity if bh_metrics.ending_equity else 0)
    click.echo(f"\nStrategy vs buy-and-hold: {edge*100:+.2f}% in final equity")
    click.echo(f"Trades: {fills['trades']}, win rate: {fills['win_rate']:.1%}, "
               f"commission: ${fills['total_commission']:,.2f}")


if __name__ == "__main__":
    cli()
