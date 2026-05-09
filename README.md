# MT — Halal-screened equities trading bot

A small Python bot that trades a hand-screened universe of US equities through
Interactive Brokers (paper or live). All orders pass through a halal allowlist
guard *and* a cash-account / no-shorting risk check before they leave the engine.

## Halal trading rules baked in

These are enforced in code, not just convention. Bypassing them requires
editing the source.

1. **Universe allowlist** — only symbols in [config/halal_universe.yaml](config/halal_universe.yaml) can be traded. Seeded from the intersection of SPUS and HLAL ETF holdings.
2. **Cash account only** — engine refuses to start on a margin account (no interest-bearing borrowing).
3. **No short selling** — `SELL` orders rejected if quantity exceeds current long position.
4. **No options / futures / forex** — broker layer only exposes equities.
5. **No interest on idle cash** — see "Required IBKR setup" below; you must opt out manually in IBKR.

The first guard runs in [src/mt/engine.py](src/mt/engine.py) before any data is fetched, and again at the broker boundary in [src/mt/broker.py:submit_market_order](src/mt/broker.py).

## Required IBKR setup

You need an Interactive Brokers account (works from Pakistan). Paper trading
is included with any account.

**Critical halal step — opt out of interest on cash:**
1. Log into IBKR Account Management
2. Settings → Account Configuration → "Interest Paid on Cash"
3. Set to **No** / opt out
4. IBKR pays interest on USD balances over $10,000 by default. That income is riba.

**Account type:** make sure the account is **Cash**, not **Margin**. The bot
will refuse to connect to a margin account, but you should choose Cash at
sign-up regardless.

**Run TWS or IB Gateway:**
- Download Trader Workstation or IB Gateway from interactivebrokers.com
- Log into the **paper** trading account
- Enable API: Configure → Settings → API → Settings:
  - Check "Enable ActiveX and Socket Clients"
  - Uncheck "Read-Only API"
  - Socket port: `4002` (Gateway paper, recommended) or `7497` (TWS paper)
  - Add `127.0.0.1` to "Trusted IPs"
- Restart Gateway/TWS for settings to fully apply

## Setup

```bash
# Python 3.11+
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env
# fill in IBKR_ACCOUNT (your paper account ID, e.g. DU1234567)

# Verify the universe loads
mt list-universe

# With TWS/Gateway running on the paper port:
mt account                 # confirm connection + cash account
mt run --dry-run           # compute signals, submit nothing
mt run                     # actually place paper orders
```

## Tests

```bash
pytest
```

The universe and risk tests do not require a running IBKR connection.

## Going live

The bot will refuse to connect to a live-trading port (7496 or 4001) unless
`ALLOW_LIVE=true` is set in `.env`. Before flipping that flag:

1. Run `mt run` against paper for **at least several weeks**, then read the trade log line by line.
2. Re-screen every symbol in the universe against current AAOIFI ratios. Compliance changes quarterly.
3. Confirm the IBKR account is **Cash** (not Margin) and that interest-on-cash is opted out.
4. Lower `RiskLimits.max_order_pct_of_equity` for the first live run; the default 5% is fine for paper but aggressive for new live capital.

## Layout

```
src/mt/
  universe.py    # halal allowlist + assert_halal guard
  risk.py        # cash-account, no-shorting, position-sizing checks
  broker.py      # IBKR connection via ib_async; halal guard at submit
  strategy.py    # SMA crossover starter; replace as you like
  engine.py      # data → signal → risk → broker, one pass per call
  cli.py         # `mt list-universe`, `mt account`, `mt run`
config/
  halal_universe.yaml   # the screened symbols
tests/
  test_universe.py   # allowlist enforcement
  test_risk.py       # halal + risk rules
```

## Scholarly notes (worth knowing)

- **Settlement (T+1):** US equities now settle T+1 (since May 2024). Most contemporary fatwas accept this.
- **Brokerage commissions:** universally accepted as a fee-for-service, not riba.
- **Compliance is dynamic:** AAOIFI ratios are recomputed each quarter. A stock that passes today may fail next quarter. **Re-screen the universe quarterly** — at minimum check debt-to-market-cap, cash-and-receivables-to-market-cap, and impermissible-income ratios via Zoya, Islamicly, Musaffa, or your scholar.
- **Tesla, Procter & Gamble, Coca-Cola, Home Depot, NextEra:** historically borderline on debt ratios — listed under `watchlist_review_required` in the YAML and **not tradable** until you re-verify them yourself.
- **Pure-play sectors that are out:** conventional banks, conventional insurance, alcohol, tobacco, gambling, adult, primary defense contractors, pork, conventional asset management, interest-based lending. The screen is sector-then-ratios.

This is not a fatwa. Cross-check anything significant with a scholar you trust.
