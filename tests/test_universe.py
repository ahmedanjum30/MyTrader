from pathlib import Path

import pytest
import yaml

from mt.universe import NotHalalError, Universe, load_universe


def make_universe(symbols=("AAPL", "MSFT"), review=("TSLA",)):
    return Universe(
        symbols=frozenset(symbols),
        review_required=frozenset(review),
        forbidden_sectors=frozenset(),
    )


def test_assert_halal_accepts_known_symbol():
    u = make_universe()
    u.assert_halal("AAPL")
    u.assert_halal("aapl")  # case-insensitive
    u.assert_halal(" MSFT ")  # whitespace tolerant


def test_assert_halal_rejects_unknown_symbol():
    u = make_universe()
    with pytest.raises(NotHalalError, match="not in the halal-screened universe"):
        u.assert_halal("JPM")


def test_assert_halal_rejects_review_required_with_specific_message():
    u = make_universe()
    with pytest.raises(NotHalalError, match="review-required watchlist"):
        u.assert_halal("TSLA")


def test_contains_check():
    u = make_universe()
    assert "AAPL" in u
    assert "JPM" not in u
    assert 42 not in u


def test_load_universe_from_default_config():
    u = load_universe()
    assert len(u.symbols) > 0
    # Spot-check that obvious haram sectors aren't accidentally present.
    forbidden_examples = {"JPM", "BAC", "C", "WFC", "GS", "MS"}  # major banks
    assert not (u.symbols & forbidden_examples), (
        f"Bank tickers found in halal universe: {u.symbols & forbidden_examples}"
    )


def test_yaml_structure_is_valid():
    config = Path(__file__).resolve().parents[1] / "config" / "halal_universe.yaml"
    data = yaml.safe_load(config.read_text())
    assert "universe" in data
    assert isinstance(data["universe"], list)
    assert all(isinstance(s, str) and s.isupper() for s in data["universe"])
