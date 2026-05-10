from decimal import Decimal

import pytest

from mt_alpaca.risk import AccountSnapshot, ProposedOrder, RiskLimits, RiskViolation, check_order


def cash_account(equity="100000", cash="100000", positions=0):
    return AccountSnapshot(
        equity=Decimal(equity),
        cash=Decimal(cash),
        is_margin_account=False,
        open_position_count=positions,
    )


def buy(symbol="AAPL", qty="10", price="200", current="0"):
    return ProposedOrder(
        symbol=symbol, side="BUY",
        quantity=Decimal(qty), estimated_price=Decimal(price),
        current_position=Decimal(current),
    )


def sell(symbol="AAPL", qty="10", price="200", current="10"):
    return ProposedOrder(
        symbol=symbol, side="SELL",
        quantity=Decimal(qty), estimated_price=Decimal(price),
        current_position=Decimal(current),
    )


def test_margin_account_rejected():
    account = AccountSnapshot(
        equity=Decimal("100000"), cash=Decimal("100000"),
        is_margin_account=True, open_position_count=0,
    )
    with pytest.raises(RiskViolation, match="cash account"):
        check_order(buy(), account, RiskLimits())


def test_short_sell_rejected():
    # Trying to sell more than we own.
    with pytest.raises(RiskViolation, match="Short selling is not permitted"):
        check_order(sell(qty="10", current="0"), cash_account(), RiskLimits())


def test_short_sell_partial_position_rejected():
    with pytest.raises(RiskViolation, match="Short selling is not permitted"):
        check_order(sell(qty="10", current="3"), cash_account(), RiskLimits())


def test_sell_within_position_allowed():
    check_order(sell(qty="10", current="10"), cash_account(), RiskLimits())
    check_order(sell(qty="5", current="10"), cash_account(), RiskLimits())


def test_buy_within_limits_allowed():
    check_order(buy(qty="10", price="200"), cash_account(), RiskLimits())


def test_oversized_order_rejected():
    # 100 * 200 = 20000 = 20% of 100k equity, exceeds 5% limit.
    with pytest.raises(RiskViolation, match="exceeds max_order_pct_of_equity"):
        check_order(buy(qty="100", price="200"), cash_account(), RiskLimits())


def test_insufficient_cash_rejected():
    # Equity 100k but only 100 in cash.
    account = cash_account(equity="100000", cash="100")
    with pytest.raises(RiskViolation, match="Insufficient cash"):
        check_order(buy(qty="10", price="200"), account, RiskLimits())


def test_max_open_positions_blocks_new_position():
    account = cash_account(positions=15)
    with pytest.raises(RiskViolation, match="max_open_positions"):
        check_order(buy(qty="10", price="200", current="0"), account, RiskLimits())


def test_max_open_positions_allows_adding_to_existing():
    account = cash_account(positions=15)
    # current_position > 0 means we're adding to an existing position.
    check_order(buy(qty="10", price="200", current="5"), account, RiskLimits())


def test_zero_quantity_rejected():
    with pytest.raises(RiskViolation, match="positive"):
        check_order(buy(qty="0"), cash_account(), RiskLimits())
