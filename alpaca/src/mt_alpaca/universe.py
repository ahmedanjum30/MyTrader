"""Halal universe: load the allowlist and enforce it on every order.

This module is the last line of defense. If a symbol is not in the loaded
allowlist, `assert_halal` raises and the engine refuses to send the order.
The guard runs in two places (engine pre-flight and broker submit) so a
single bug elsewhere cannot route a non-screened symbol to market.
"""

from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "halal_universe.yaml"


class NotHalalError(ValueError):
    """Raised when an order targets a symbol outside the screened universe."""


@dataclass(frozen=True)
class Universe:
    symbols: frozenset[str]
    review_required: frozenset[str]
    forbidden_sectors: frozenset[str]

    def assert_halal(self, symbol: str) -> None:
        s = symbol.upper().strip()
        if s in self.symbols:
            return
        if s in self.review_required:
            raise NotHalalError(
                f"{s} is on the review-required watchlist. Re-screen current "
                "AAOIFI ratios before adding it to `universe`."
            )
        raise NotHalalError(
            f"{s} is not in the halal-screened universe. Refusing to trade."
        )

    def __contains__(self, symbol: object) -> bool:
        return isinstance(symbol, str) and symbol.upper().strip() in self.symbols


def load_universe(path: Path | str = DEFAULT_CONFIG) -> Universe:
    data = yaml.safe_load(Path(path).read_text())
    return Universe(
        symbols=frozenset(s.upper() for s in data.get("universe", [])),
        review_required=frozenset(s.upper() for s in data.get("watchlist_review_required", [])),
        forbidden_sectors=frozenset(data.get("forbidden_sectors", [])),
    )
