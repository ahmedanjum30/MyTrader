# MyTrader — Halal trading bot, dual-broker

Two parallel broker implementations sharing the same halal universe, risk model, strategies, and backtester. Pick the broker that fits your country.

```
MT/
├── ibkr/                 ← Interactive Brokers implementation (mt CLI)
│   ├── src/mt/
│   ├── tests/
│   ├── config/halal_universe.yaml
│   └── pyproject.toml
└── alpaca/               ← Alpaca implementation (mta CLI)
    ├── src/mt_alpaca/
    ├── tests/
    ├── config/halal_universe.yaml
    └── pyproject.toml
```

## Which one to use

| If you're in… | Use |
|---|---|
| Pakistan / most non-US countries (until verified Alpaca eligibility) | **IBKR** |
| US / Alpaca-supported countries | **Alpaca** (cheaper, fractional via API, modern REST) |
| Both available | **Alpaca** is simpler; IBKR has wider international order routing |

## Halal trading rules (enforced in both implementations)

These are baked into both `ibkr/` and `alpaca/`:

1. **Universe allowlist** — only screened symbols can be traded; `assert_halal()` runs at engine pre-flight AND broker submit
2. **Cash account only** — engine refuses to start if account has margin enabled
3. **No short selling** — `SELL` orders rejected if quantity > current long position
4. **No options/futures/forex** — broker layer exposes equities only
5. **No interest on idle cash** — for IBKR, opt out manually; Alpaca cash accounts don't pay interest by default
6. **Live-port guardrail** — `ALLOW_LIVE=true` required to permit any live-API call

## Setup — IBKR side

```bash
cd ibkr
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env
# fill in IBKR_ACCOUNT (your paper account ID, e.g. DUQ123456)

# Make sure IB Gateway is running and logged into paper:
mt account               # confirm connection + cash account
mt list-universe         # print 28 screened symbols
mt deploy --budget 1000  # one-shot whole-share buyhold (greedy-cheapest)
mt status --budget 1000  # color-coded P&L
```

See [ibkr/README.md](ibkr/README.md) for full IBKR setup details.

## Setup — Alpaca side

```bash
cd alpaca
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env
# fill in ALPACA_API_KEY and ALPACA_SECRET_KEY from your dashboard

# No local gateway needed — pure REST API:
mta account              # confirm connection + cash account
mta list-universe        # same 28 symbols (shared config)
mta deploy --budget 1000 --strategy buyhold  # fractional buyhold of full universe
mta status --budget 1000 # color-coded P&L
```

## Key differences between the two

| | IBKR (`mt`) | Alpaca (`mta`) |
|---|---|---|
| Country availability | Most countries (incl. Pakistan) | US + selected (verify) |
| Connection | Local Gateway / TWS via socket | Pure HTTPS REST |
| Setup overhead | Run IB Gateway, log in, configure API | API key + secret, that's it |
| Commission | $0.0035/share, $0.35 min | $0 |
| Fractional shares (API) | ❌ Blocked | ✅ Native |
| Mobile app | TWS Mobile separate | Trade from web/mobile |
| Universe approach | Whole-share greedy-cheapest workaround | Equal-weight full universe |
| Real-time data | Subscription tiers | Free for paper |

## What's shared between them

Both implementations contain identical copies of:
- `config/halal_universe.yaml` — 28-symbol AAOIFI-screened allowlist
- `universe.py` — `assert_halal()` guard
- `risk.py` — RiskLimits, halal trading rules
- `strategy.py` — 9 strategies (BuyAndHold, SmaCrossover, TrendFilter, Momentum, QuarterlyEqualWeight, InverseVolWeighted, InverseVolTrendGated, SwingMeanReversion, SwingBreakout)
- `backtest/` — yfinance-based walk-forward backtester (broker-independent)
- 27 unit tests (universe, risk, portfolio invariants)

This redundancy is intentional for v1 — keeps each side self-contained and broker-agnostic refactoring is non-trivial. Future work: extract shared core into a `halal_core` package both depend on.

## Backtest results

We tested 9 strategies across 5 historical regimes (GFC, mid-cycle bull, COVID, inflation drawdown, AI bull). Headline finding: **buy-and-hold beats every active strategy** at retail size, after costs.

- $10k → $53k over 6 years on buyhold (28% CAGR — abnormally good 2020–2026)
- All active strategies underperformed; some lost money
- Realistic forward expectation: 8–12% CAGR long-run

See backtest commands: `mt backtest`, `mt regimes`, `mt compare`, `mt yearly`.

## Going live

Default is paper. To enable live API:

1. **IBKR**: set `ALLOW_LIVE=true` and `IBKR_PORT=4001` (Gateway-live) in `ibkr/.env`
2. **Alpaca**: set `ALLOW_LIVE=true` and `ALPACA_PAPER=false` in `alpaca/.env`

For both, **read each side's README first.** Re-screen the universe quarterly. Lower position sizes for first live runs.

## License / disclaimer

Personal project. Not financial advice. Halal screening is mine + ETF-derived; cross-check with your scholar before live trading.
