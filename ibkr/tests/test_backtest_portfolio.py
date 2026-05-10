"""The most important backtest tests: portfolio accounting invariants.

If these break, every metric downstream is wrong.
"""

import pandas as pd
import pytest

from mt.backtest.portfolio import Portfolio, PortfolioError


def ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s)


def test_starting_state():
    p = Portfolio(starting_cash=10_000)
    assert p.cash == 10_000
    assert p.positions == {}
    assert p.equity({}) == 10_000


def test_buy_then_sell_conserves_value_minus_costs():
    p = Portfolio(starting_cash=10_000, slippage_bps=0,
                  commission_per_share=0, min_commission=0)
    p.buy(ts("2024-01-01"), "AAPL", 10, mid_price=100)
    assert p.cash == 9_000
    assert p.qty("AAPL") == 10
    assert p.equity({"AAPL": 100}) == 10_000  # zero costs, perfect conservation

    p.sell(ts("2024-01-02"), "AAPL", 10, mid_price=110)
    assert p.cash == pytest.approx(10_100)
    assert p.qty("AAPL") == 0


def test_buy_with_slippage_costs_more():
    p = Portfolio(starting_cash=10_000, slippage_bps=10,
                  commission_per_share=0, min_commission=0)
    p.buy(ts("2024-01-01"), "AAPL", 10, mid_price=100)
    # 10 bps slippage on BUY → fill at 100.10
    assert p.fills[-1].price == pytest.approx(100.10)
    assert p.cash == pytest.approx(10_000 - 10 * 100.10)


def test_buy_with_commission_minimum_applies():
    p = Portfolio(starting_cash=10_000, slippage_bps=0,
                  commission_per_share=0.01, min_commission=1.0)
    p.buy(ts("2024-01-01"), "AAPL", 10, mid_price=100)
    # 10 shares * $0.01 = $0.10, but min commission is $1
    assert p.fills[-1].commission == 1.0
    assert p.cash == pytest.approx(10_000 - 1000 - 1)


def test_short_sale_rejected():
    p = Portfolio(starting_cash=10_000)
    with pytest.raises(PortfolioError, match="no shorting"):
        p.sell(ts("2024-01-01"), "AAPL", 10, mid_price=100)


def test_partial_short_rejected():
    p = Portfolio(starting_cash=10_000, slippage_bps=0,
                  commission_per_share=0, min_commission=0)
    p.buy(ts("2024-01-01"), "AAPL", 5, mid_price=100)
    with pytest.raises(PortfolioError, match="no shorting"):
        p.sell(ts("2024-01-02"), "AAPL", 10, mid_price=100)


def test_buy_more_than_cash_rejected():
    p = Portfolio(starting_cash=1_000)
    with pytest.raises(PortfolioError, match="cash is"):
        p.buy(ts("2024-01-01"), "AAPL", 100, mid_price=100)


def test_negative_quantity_rejected():
    p = Portfolio(starting_cash=10_000)
    with pytest.raises(PortfolioError, match="positive"):
        p.buy(ts("2024-01-01"), "AAPL", -10, mid_price=100)
    with pytest.raises(PortfolioError, match="positive"):
        p.sell(ts("2024-01-01"), "AAPL", 0, mid_price=100)


def test_equity_curve_records_correctly():
    p = Portfolio(starting_cash=10_000, slippage_bps=0,
                  commission_per_share=0, min_commission=0)
    p.record_equity(ts("2024-01-01"), {})
    p.buy(ts("2024-01-02"), "AAPL", 10, mid_price=100)
    p.record_equity(ts("2024-01-02"), {"AAPL": 100})
    p.record_equity(ts("2024-01-03"), {"AAPL": 110})  # AAPL went up, equity should rise

    s = p.equity_series()
    assert len(s) == 3
    assert s.iloc[0] == 10_000
    assert s.iloc[1] == 10_000   # cost-basis price
    assert s.iloc[2] == 10_100   # +10 * (110-100)


def test_record_equity_catches_negative_cash_invariant():
    """Test the safety net even though buy() should prevent negative cash."""
    p = Portfolio(starting_cash=10_000)
    # Force-corrupt cash to simulate a bug
    p.cash = -100
    with pytest.raises(PortfolioError, match="Cash invariant"):
        p.record_equity(ts("2024-01-01"), {})


def test_partial_sell_keeps_remaining_position():
    p = Portfolio(starting_cash=10_000, slippage_bps=0,
                  commission_per_share=0, min_commission=0)
    p.buy(ts("2024-01-01"), "AAPL", 10, mid_price=100)
    p.sell(ts("2024-01-02"), "AAPL", 4, mid_price=100)
    assert p.qty("AAPL") == 6
