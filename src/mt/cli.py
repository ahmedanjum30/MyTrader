"""CLI entry points: `mt list-universe`, `mt account`, `mt run`."""

import logging
import os
from pathlib import Path

import click
from dotenv import load_dotenv

from .backtest import data as bt_data
from .backtest.engine import run_backtest
from .backtest.metrics import compute as compute_metrics, fills_summary
from .broker import IBKRBroker, IBKRConfig
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


def _config() -> IBKRConfig:
    return IBKRConfig(
        host=os.environ.get("IBKR_HOST", "127.0.0.1"),
        port=int(os.environ.get("IBKR_PORT", "7497")),
        client_id=int(os.environ.get("IBKR_CLIENT_ID", "17")),
        account=os.environ.get("IBKR_ACCOUNT") or None,
        allow_live=os.environ.get("ALLOW_LIVE", "false").lower() == "true",
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
    """Symbols this strategy needs in `extra` (non-universe market data)."""
    return ["SPY"] if name in {"trend", "invvol_trend"} else []


@click.group()
def cli() -> None:
    """Halal-screened equities trading bot on IBKR."""


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
    """Connect to IBKR and print the account snapshot."""
    universe = load_universe()
    with IBKRBroker(_config(), universe) as broker:
        snap = broker.account_snapshot()
        click.echo(f"Paper:           {broker.is_paper}")
        click.echo(f"Equity:          {snap.equity}")
        click.echo(f"Cash:            {snap.cash}")
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
    """Run one rebalance pass against the IBKR broker."""
    universe = load_universe()
    with IBKRBroker(_config(), universe) as broker:
        if dry_run:
            real_submit = broker.submit_market_order
            broker.submit_market_order = lambda symbol, side, quantity: click.echo(
                f"[DRY-RUN] would submit {side} {quantity} {symbol}"
            )
        engine = Engine(broker=broker, universe=universe,
                        strategy=_strategy_for(strategy_name),
                        limits=RiskLimits(),
                        budget=budget)
        click.echo(f"Strategy: {strategy_name}, "
                   f"budget: {f'${budget:.2f}' if budget else 'full account'}, "
                   f"dry_run: {dry_run}")
        engine.run_once()


@cli.command("deploy")
@click.option("--budget", default=1000.0, type=float, help="Total dollar budget")
@click.option("--dry-run", is_flag=True, help="Show plan, submit nothing")
@click.option("--limit-buffer", default=0.005, type=float,
              help="Limit price buffer above last close (default 0.5%%)")
@click.option("--cancel-pending", is_flag=True,
              help="Cancel any open orders before deploying")
def deploy_cmd(budget: float, dry_run: bool, limit_buffer: float,
               cancel_pending: bool) -> None:
    """One-shot whole-share deployment.

    Greedy-cheapest rule: sort universe by current price ascending; buy 1 share
    of each in that order until adding the next would exceed `budget`.
    Uses LIMIT orders (last_close × (1 + buffer)) — works reliably without
    real-time market data subscriptions and outside market hours.
    """
    from decimal import Decimal

    universe = load_universe()
    with IBKRBroker(_config(), universe) as broker:
        if cancel_pending:
            n = broker.cancel_all_open_orders()
            click.echo(f"Cancelled {n} pending orders.")

        held = []
        for sym in sorted(universe.symbols):
            qty = broker.position_qty(sym)
            if qty > 0:
                held.append((sym, qty))
        if held and not dry_run:
            click.echo("Already holding positions in halal universe:")
            for sym, qty in held:
                click.echo(f"  {sym}: {float(qty)} shares")
            click.echo("`mt deploy` is one-shot. Inspect or close positions first.")
            return

        click.echo(f"Fetching current prices for {len(universe.symbols)} symbols...")
        prices: dict[str, float] = {}
        for sym in sorted(universe.symbols):
            try:
                bars = broker.historical_bars(sym, lookback="5 D")
                if not bars.empty:
                    prices[sym] = float(bars["close"].iloc[-1])
            except Exception as e:
                click.echo(f"  WARNING: no price for {sym} ({e})")

        ranked = sorted(prices.items(), key=lambda x: x[1])
        plan: list[tuple[str, float, float]] = []
        cum = 0.0
        for sym, price in ranked:
            limit_px = price * (1.0 + limit_buffer)
            if cum + limit_px > budget:
                break
            plan.append((sym, price, limit_px))
            cum += limit_px

        click.echo(f"\nGreedy-cheapest LIMIT order plan (budget ${budget:.0f}, "
                   f"buffer +{limit_buffer*100:.2f}%):")
        click.echo(f"  {'Symbol':<8}  {'Last':>9}  {'Limit':>9}  {'Cumulative':>12}")
        click.echo("-" * 50)
        running = 0.0
        for sym, last, limit_px in plan:
            running += limit_px
            click.echo(f"  {sym:<8}  ${last:>8.2f}  ${limit_px:>8.2f}  ${running:>11.2f}")
        click.echo("-" * 50)
        click.echo(f"  {len(plan)} positions, max ${cum:.2f} (worst-case fill at limits), "
                   f"${budget - cum:.2f} cash buffer")

        if dry_run:
            click.echo("\n[DRY-RUN] no orders submitted.")
            return

        click.echo("\nSubmitting limit orders...")
        for sym, _last, limit_px in plan:
            try:
                broker.submit_limit_order(sym, "BUY", Decimal("1"),
                                           Decimal(f"{limit_px:.2f}"))
            except Exception as e:
                click.echo(f"  FAILED {sym}: {e}")
        click.echo("Done. Orders are DAY limits — they'll work the next session if "
                   "market is closed. Run `mt status` to inspect.")


@cli.command("status")
@click.option("--budget", default=None, type=float,
              help="Reference budget for weight calculations")
def status_cmd(budget: float | None) -> None:
    """Show current positions, cost basis, current value, and P&L."""
    universe = load_universe()
    with IBKRBroker(_config(), universe) as broker:
        snap = broker.account_snapshot()
        click.echo(f"Account:        Paper={broker.is_paper}  "
                   f"Cash account={not snap.is_margin_account}")
        click.echo(f"Equity:         ${snap.equity:,.2f}")
        click.echo(f"Cash:           ${snap.cash:,.2f}")
        click.echo(f"Open positions: {snap.open_position_count}")
        if budget:
            click.echo(f"Budget cap:     ${budget:,.2f}")

        # Pull positions WITH avg cost in a single call.
        positions = []
        for p in broker.ib.positions(broker.config.account):
            sym = p.contract.symbol
            if sym in universe.symbols and p.position > 0:
                positions.append((sym, float(p.position), float(p.avgCost)))

        # Also list pending open orders (not yet filled).
        pending = []
        for trade in broker.ib.openTrades():
            sym = trade.contract.symbol
            if sym in universe.symbols and trade.orderStatus.status not in (
                    "Filled", "Cancelled", "ApiCancelled"):
                pending.append((sym, trade.orderStatus.status,
                                float(trade.order.totalQuantity),
                                float(trade.order.lmtPrice or 0)))

        if not positions and not pending:
            click.echo("\nNo positions or pending orders in halal universe yet.")
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
                    bars = broker.historical_bars(sym, lookback="5 D")
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
            click.echo(f"  {'Symbol':<8}  {'Status':>14}  {'Qty':>6}  {'Limit':>10}")
            click.echo("  " + "-" * 45)
            for sym, status, qty, lmt in sorted(pending):
                click.echo(f"  {sym:<8}  {status:>14}  {qty:>6.0f}  ${lmt:>9.2f}")


def _row(label: str, m) -> str:
    return (f"  {label:<22}  end=${m.ending_equity:>11,.2f}  "
            f"return={m.total_return*100:>7.2f}%  "
            f"CAGR={m.cagr*100:>6.2f}%  "
            f"Sharpe={m.sharpe:>5.2f}  "
            f"maxDD={m.max_drawdown*100:>6.2f}%")


def _print_survivorship(panel, start, end) -> None:
    warnings = bt_data.survivorship_warnings(panel, start, end)
    if warnings:
        click.echo(click.style(
            "\n  Survivorship-bias notice — universe was not contemporaneous:",
            fg="yellow"))
        for w in warnings:
            click.echo(f"    - {w}")
        click.echo()


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
@click.option("--start", default="2020-01-01", help="Start date YYYY-MM-DD")
@click.option("--end", default=None, help="End date YYYY-MM-DD (default: today)")
@click.option("--cash", default=10_000.0, type=float, help="Starting equity")
@click.option("--strategy", "strategy_name", default="sma",
              type=click.Choice(list(STRATEGY_FACTORY.keys())),
              help="Strategy to test")
@click.option("--refresh", is_flag=True, help="Force re-pull bars from yfinance")
@click.option("--save-equity", default=None, type=click.Path(),
              help="Save equity curve CSV to this path")
@click.option("--walkforward/--no-walkforward", default=True,
              help="Split into IS (60%%) and OOS (40%%) and report both")
def backtest_cmd(start: str, end: str | None, cash: float, strategy_name: str,
                 refresh: bool, save_equity: str | None, walkforward: bool) -> None:
    """Replay strategy over historical bars. OOS is the only number that counts."""
    import pandas as pd

    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    universe = load_universe()
    symbols = sorted(universe.symbols)

    click.echo(f"Loading bars for {len(symbols)} symbols ({start} → {end})...")
    panel = bt_data.load_panel(symbols, start, end, refresh=refresh)
    click.echo(f"Loaded {len(panel)} symbols ({sum(len(d) for d in panel.values())} total bars).")
    _print_survivorship(panel, start, end)

    extra = _load_extra(_needs_extra(strategy_name), start, end, refresh=refresh)

    click.echo(f"Strategy: {strategy_name}, starting_cash=${cash:,.2f}, "
               f"walkforward={walkforward}\n")

    def _segment(p, label: str, strat=None):
        return _run_segment(p, universe, strat or _strategy_for(strategy_name),
                            cash, label, extra=extra if strat is None else None)

    if walkforward:
        is_panel, oos_panel = bt_data.split_walkforward(panel, in_sample_pct=0.6)
        click.echo("=" * 110)
        is_result, is_metrics, _ = _segment(is_panel, "in-sample")
        oos_result, oos_metrics, oos_fills = _segment(oos_panel, "OOS")
        click.echo(_row(f"{strategy_name} IS  (train)", is_metrics))
        click.echo(_row(f"{strategy_name} OOS (test)", oos_metrics))
        click.echo("-" * 110)
        bh_is_result, bh_is_metrics, _ = _segment(is_panel, "B&H IS", BuyAndHold())
        bh_oos_result, bh_oos_metrics, _ = _segment(oos_panel, "B&H OOS", BuyAndHold())
        click.echo(_row("buyhold IS  (train)", bh_is_metrics))
        click.echo(_row("buyhold OOS (test)", bh_oos_metrics))
        click.echo("=" * 110)

        oos_edge = ((oos_metrics.ending_equity - bh_oos_metrics.ending_equity)
                    / bh_oos_metrics.ending_equity if bh_oos_metrics.ending_equity else 0)
        click.echo(f"\nOOS edge over buy-and-hold: {oos_edge*100:+.2f}%")
        click.echo(f"OOS trades: {oos_fills['trades']}, "
                   f"win rate: {oos_fills['win_rate']:.1%}, "
                   f"commission: ${oos_fills['total_commission']:,.2f}")

        if save_equity:
            df = pd.DataFrame({
                "strat_is": is_result.portfolio.equity_series(),
                "strat_oos": oos_result.portfolio.equity_series(),
                "bh_is": bh_is_result.portfolio.equity_series(),
                "bh_oos": bh_oos_result.portfolio.equity_series(),
            })
            df.to_csv(save_equity)
            click.echo(f"\nEquity curves saved to {save_equity}")
        return

    # Single-window mode
    result, metrics, fills_stats = _segment(panel, strategy_name)
    bh_result, bh_metrics, _ = _segment(panel, "B&H", BuyAndHold())
    click.echo("=" * 110)
    click.echo(_row(strategy_name, metrics))
    click.echo(_row("buy&hold", bh_metrics))
    click.echo("=" * 110)
    edge = ((metrics.ending_equity - bh_metrics.ending_equity)
            / bh_metrics.ending_equity if bh_metrics.ending_equity else 0)
    click.echo(f"\nStrategy vs buy-and-hold: {edge*100:+.2f}% in final equity")
    click.echo(f"Trades: {fills_stats['trades']}, "
               f"win rate: {fills_stats['win_rate']:.1%}, "
               f"commission: ${fills_stats['total_commission']:,.2f}")

    if save_equity:
        df = pd.DataFrame({"strategy": result.portfolio.equity_series(),
                           "buyhold": bh_result.portfolio.equity_series()})
        df.to_csv(save_equity)
        click.echo(f"\nEquity curve saved to {save_equity}")


# Known historical regimes for stress-testing. Pre-2010 names that didn't exist
# yet are dropped automatically by survivorship-bias detection.
DEFAULT_REGIMES = [
    ("2008-01-01", "2010-12-31", "GFC + recovery"),
    ("2015-01-01", "2018-12-31", "Mid-cycle bull"),
    ("2020-01-01", "2022-06-30", "COVID + meme rally"),
    ("2022-01-01", "2022-12-31", "Inflation drawdown"),
    ("2023-01-01", None, "AI bull (current)"),
]


@cli.command("regimes")
@click.option("--cash", default=10_000.0, type=float)
@click.option("--strategy", "strategy_name", default="sma",
              type=click.Choice(list(STRATEGY_FACTORY.keys())))
@click.option("--refresh", is_flag=True)
def regimes_cmd(cash: float, strategy_name: str, refresh: bool) -> None:
    """Run the strategy across distinct historical market regimes."""
    import pandas as pd

    universe = load_universe()
    symbols = sorted(universe.symbols)
    extra_syms = _needs_extra(strategy_name)

    click.echo(f"Strategy: {strategy_name}, starting_cash=${cash:,.2f} per regime\n")
    rows = []
    for start, end, label in DEFAULT_REGIMES:
        end_eff = end or pd.Timestamp.today().strftime("%Y-%m-%d")
        click.echo(click.style(f"\n=== {label}: {start} → {end_eff} ===", fg="cyan"))
        panel = bt_data.load_panel(symbols, start, end_eff, refresh=refresh)
        extra = _load_extra(extra_syms, start, end_eff, refresh=refresh)
        if not panel:
            click.echo("  (no data)")
            continue

        _, m, fills = _run_segment(panel, universe, _strategy_for(strategy_name),
                                    cash, label, extra=extra)
        _, bh_m, _ = _run_segment(panel, universe, BuyAndHold(), cash, label)
        click.echo(_row(strategy_name, m))
        click.echo(_row("buy&hold", bh_m))

        edge = ((m.ending_equity - bh_m.ending_equity) / bh_m.ending_equity
                if bh_m.ending_equity else 0)
        rows.append({
            "regime": label, "period": f"{start}→{end_eff}",
            "strat_cagr": m.cagr, "bh_cagr": bh_m.cagr,
            "strat_sharpe": m.sharpe, "bh_sharpe": bh_m.sharpe,
            "strat_maxdd": m.max_drawdown, "bh_maxdd": bh_m.max_drawdown,
            "edge_pct": edge, "trades": fills["trades"],
        })

    click.echo("\n" + "=" * 110)
    click.echo("Summary across regimes")
    click.echo("=" * 110)
    click.echo(f"  {'Regime':<22}  {'Period':<24}  "
               f"{'Strat CAGR':>11}  {'B&H CAGR':>10}  {'Edge':>9}  {'Trades':>7}")
    for r in rows:
        click.echo(f"  {r['regime']:<22}  {r['period']:<24}  "
                   f"{r['strat_cagr']*100:>10.2f}%  {r['bh_cagr']*100:>9.2f}%  "
                   f"{r['edge_pct']*100:>+8.2f}%  {r['trades']:>7}")


@cli.command("yearly")
@click.option("--start", default="2020-01-01")
@click.option("--end", default=None)
@click.option("--cash", default=10_000.0, type=float)
@click.option("--strategies", default="sma,buyhold,trend,momentum,quarterly_ew,invvol,invvol_trend")
@click.option("--refresh", is_flag=True)
def yearly_cmd(start: str, end: str | None, cash: float, strategies: str,
                refresh: bool) -> None:
    """Run each strategy once from `start` to `end`; tabulate per-year P&L."""
    import pandas as pd

    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    universe = load_universe()
    symbols = sorted(universe.symbols)
    strategy_names = [s.strip() for s in strategies.split(",") if s.strip()]
    for n in strategy_names:
        if n not in STRATEGY_FACTORY:
            raise click.UsageError(f"Unknown strategy: {n}")

    click.echo(f"Loading bars for {len(symbols)} symbols ({start} → {end})...")
    panel = bt_data.load_panel(symbols, start, end, refresh=refresh)
    extra_syms = sorted({s for n in strategy_names for s in _needs_extra(n)})
    extra = _load_extra(extra_syms, start, end, refresh=refresh)
    click.echo(f"Loaded {len(panel)} symbols\n")

    # Strategy → (equity_series, fills_df)
    runs: dict[str, tuple] = {}
    for name in strategy_names:
        ex = extra if _needs_extra(name) else None
        click.echo(f"Running {name}...")
        result, _, _ = _run_segment(panel, universe, _strategy_for(name),
                                     cash, name, extra=ex)
        runs[name] = (result.portfolio.equity_series(),
                      result.portfolio.fills_df())

    # Strategy descriptions
    descriptions = {
        "sma": "SMA(10/30) per-symbol crossover; sell on cross-down, buy on cross-up.",
        "buyhold": "Buy each name once, never sell.",
        "trend": "Hold equal-weight when SPY > SPY-200d-SMA; cash otherwise.",
        "momentum": "Top 8 by trailing 12m return; rebalance monthly.",
        "quarterly_ew": "Equal weight across universe; rebalance every quarter.",
        "invvol": "Inverse-volatility weighted (1/60d-vol); rebalance quarterly.",
        "invvol_trend": "Inverse-vol weights when SPY > SPY-200d-SMA; cash otherwise.",
        "swing_meanrev": "Connors RSI(2): buy oversold in uptrend, exit on rebound (1-10 day swing).",
        "swing_breakout": "Donchian 20-day breakout: buy on new high, exit on +10/-5 or 10d (1-10 day swing).",
    }

    click.echo("\n" + "=" * 110)
    click.echo("Strategies tested")
    click.echo("=" * 110)
    for name in strategy_names:
        click.echo(f"  {name:<14} — {descriptions.get(name, '')}")

    click.echo("\n" + "=" * 110)
    click.echo(f"Per-year performance, starting cash ${cash:,.0f} (each strategy run independently)")
    click.echo("=" * 110)

    # For each strategy, build year-end equities + per-year fills
    years_seen: set[int] = set()
    for eq, _ in runs.values():
        if not eq.empty:
            years_seen.update(eq.index.year.unique().tolist())
    years = sorted(years_seen)

    for name in strategy_names:
        eq, fills = runs[name]
        if eq.empty:
            click.echo(f"\n{name}: no data")
            continue
        click.echo(f"\n{name}")
        click.echo(f"  {'Year':<6}  {'Start $':>10}  {'End $':>10}  "
                   f"{'P&L $':>10}  {'P&L %':>8}  {'Trades':>7}  {'Days':>5}  {'Trades/day':>11}")
        # Year-end snapshots: first/last equity value of each year
        prev_year_end_equity: float | None = None
        for year in years:
            year_slice = eq[eq.index.year == year]
            if year_slice.empty:
                continue
            start_eq = float(prev_year_end_equity) if prev_year_end_equity is not None \
                else float(year_slice.iloc[0])
            end_eq = float(year_slice.iloc[-1])
            pnl_dollar = end_eq - start_eq
            pnl_pct = (end_eq / start_eq - 1.0) * 100 if start_eq > 0 else 0.0
            year_fills = fills[pd.to_datetime(fills["date"]).dt.year == year] \
                if not fills.empty else fills
            n_fills = len(year_fills)
            n_days = len(year_slice)
            tpd = n_fills / n_days if n_days > 0 else 0.0
            click.echo(f"  {year:<6}  ${start_eq:>9,.2f}  ${end_eq:>9,.2f}  "
                       f"${pnl_dollar:>+9,.2f}  {pnl_pct:>+7.2f}%  "
                       f"{n_fills:>7}  {n_days:>5}  {tpd:>11.2f}")
            prev_year_end_equity = end_eq

        total_fills = len(fills)
        total_days = len(eq)
        total_pnl = float(eq.iloc[-1]) - float(eq.iloc[0])
        total_pct = (float(eq.iloc[-1]) / float(eq.iloc[0]) - 1.0) * 100
        click.echo(f"  {'TOTAL':<6}  ${float(eq.iloc[0]):>9,.2f}  ${float(eq.iloc[-1]):>9,.2f}  "
                   f"${total_pnl:>+9,.2f}  {total_pct:>+7.2f}%  "
                   f"{total_fills:>7}  {total_days:>5}  "
                   f"{(total_fills/total_days if total_days else 0):>11.2f}")


@cli.command("compare")
@click.option("--cash", default=10_000.0, type=float)
@click.option("--strategies", default="trend,momentum,quarterly_ew,buyhold",
              help="Comma-separated strategy names to compare")
@click.option("--refresh", is_flag=True)
def compare_cmd(cash: float, strategies: str, refresh: bool) -> None:
    """Run multiple strategies across all regimes; tabulate side by side."""
    import pandas as pd

    universe = load_universe()
    symbols = sorted(universe.symbols)
    strategy_names = [s.strip() for s in strategies.split(",") if s.strip()]
    for n in strategy_names:
        if n not in STRATEGY_FACTORY:
            raise click.UsageError(f"Unknown strategy: {n}")

    click.echo(f"Strategies: {strategy_names}, starting_cash=${cash:,.2f}\n")

    # CAGR matrix: rows = regimes, cols = strategies
    cagr: dict[tuple[str, str], float] = {}
    sharpe: dict[tuple[str, str], float] = {}
    maxdd: dict[tuple[str, str], float] = {}
    trades: dict[tuple[str, str], int] = {}

    for start, end, label in DEFAULT_REGIMES:
        end_eff = end or pd.Timestamp.today().strftime("%Y-%m-%d")
        click.echo(click.style(f"\n=== {label}: {start} → {end_eff} ===", fg="cyan"))
        panel = bt_data.load_panel(symbols, start, end_eff, refresh=refresh)
        if not panel:
            continue
        # Pre-load any extras any strategy in the set may need.
        extra_syms = sorted({s for n in strategy_names for s in _needs_extra(n)})
        extra = _load_extra(extra_syms, start, end_eff, refresh=refresh)

        for name in strategy_names:
            ex = extra if _needs_extra(name) else None
            _, m, fs = _run_segment(panel, universe, _strategy_for(name),
                                     cash, name, extra=ex)
            click.echo(_row(name, m))
            cagr[(label, name)] = m.cagr
            sharpe[(label, name)] = m.sharpe
            maxdd[(label, name)] = m.max_drawdown
            trades[(label, name)] = fs["trades"]

    click.echo("\n" + "=" * 130)
    click.echo("CAGR by regime × strategy")
    click.echo("=" * 130)
    header = f"  {'Regime':<22}" + "".join(f"{n:>14}" for n in strategy_names)
    click.echo(header)
    for _, _, label in DEFAULT_REGIMES:
        row = f"  {label:<22}"
        for name in strategy_names:
            v = cagr.get((label, name))
            row += f"{(v*100 if v is not None else 0):>13.2f}%"
        click.echo(row)

    click.echo("\nSharpe by regime × strategy")
    click.echo("-" * 130)
    click.echo(header)
    for _, _, label in DEFAULT_REGIMES:
        row = f"  {label:<22}"
        for name in strategy_names:
            v = sharpe.get((label, name))
            row += f"{v if v is not None else 0:>14.2f}"
        click.echo(row)

    click.echo("\nMax drawdown by regime × strategy")
    click.echo("-" * 130)
    click.echo(header)
    for _, _, label in DEFAULT_REGIMES:
        row = f"  {label:<22}"
        for name in strategy_names:
            v = maxdd.get((label, name))
            row += f"{(v*100 if v is not None else 0):>13.2f}%"
        click.echo(row)


if __name__ == "__main__":
    cli()
