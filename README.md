# AI Paper Trader

Hybrid AI paper-trading system powered by **Claude** (LLM weekly thesis) + **XGBoost** (ML daily signals) on top of **OpenBB** market data.

Trades with real market data, no real money.

## Architecture

```
OpenBB (data) → Brain (LLM + ML) → Gate → PaperEngine → SQLite
                                                ↓
                                         Streamlit Dashboard
```

- **LLM layer** — Claude generates a weekly bullish/bearish/neutral thesis per asset
- **ML layer** — XGBoost trained on technical features, predicts buy/hold/sell daily
- **Gate** — Both must agree (LLM confidence ≥ 0.60 + ML signal match) to trade
- **Portfolio** — 10% max allocation per position, 5% stop-loss, dynamic take-profit

## Assets

AAPL, MSFT, NVDA, SPY, QQQ, BTC-USD, ETH-USD, EURUSD

## Setup

```bash
# 1. Install dependencies
pip install -r trader/requirements.txt

# 2. Set your Anthropic API key
export ANTHROPIC_API_KEY=sk-ant-...

# 3. Start the live scheduler (runs 24/7)
python -m trader.scheduler

# 4. Launch the dashboard
streamlit run trader/dashboard.py

# 5. Run the test suite
cd trader && pytest
```

## Running a Backtest

Open the dashboard → **Backtest** tab → choose date range → click **Run Backtest**.

Or run programmatically:

```python
from datetime import date
from trader.backtest.runner import run_backtest
from trader.config import ASSETS

metrics = run_backtest(
    assets=ASSETS,
    start_date=date(2023, 1, 1),
    end_date=date(2024, 1, 1),
    starting_capital=100_000,
    run_id="my_run_001",
)
print(metrics["total_return_pct"], metrics["sharpe_ratio"])
```

## Project Structure

```
trader/
├── config.py              # All tuneable parameters
├── scheduler.py           # APScheduler — weekly + daily jobs
├── dashboard.py           # Streamlit UI
├── state/
│   ├── store.py           # SQLite CRUD
│   └── migrations/
│       └── 001_init.sql   # Schema
├── data/
│   ├── price.py           # OHLCV via OpenBB
│   ├── news.py            # Headlines via Biztoc
│   ├── fundamentals.py    # PE, revenue growth, D/E
│   └── macro.py           # Unemployment, CPI, rates
├── brain/
│   ├── features.py        # Shared feature engineering (RSI, MACD, BB, ATR…)
│   ├── train.py           # XGBoost training with atomic model writes
│   ├── ml.py              # Inference with in-memory model cache
│   ├── llm.py             # Claude thesis generation + disk cache
│   └── gate.py            # Combines LLM + ML → ENTER / CLOSE / HOLD
├── execution/
│   ├── portfolio.py       # Position sizing, P&L, cash management
│   └── paper.py           # Paper trade execution + stop enforcement
├── backtest/
│   └── runner.py          # Time-step simulation, Sharpe, drawdown metrics
└── tests/
    ├── conftest.py
    ├── test_store.py
    ├── test_features.py
    ├── test_gate.py
    ├── test_portfolio.py
    └── test_paper.py
```

## Configuration (`trader/config.py`)

| Parameter | Default | Description |
|---|---|---|
| `STARTING_CAPITAL` | $100,000 | Paper trading starting capital |
| `MAX_POSITIONS` | 20 | Max simultaneous positions |
| `MAX_ALLOCATION` | 10% | Max per-position size |
| `STOP_LOSS_PCT` | 5% | Hard stop-loss |
| `MIN_LLM_CONFIDENCE` | 0.60 | Minimum LLM confidence to act |
| `LABEL_THRESHOLD` | 1.5% | ML label threshold (buy/sell vs hold) |
| `LABEL_HORIZON` | 5 days | Forward-return window for ML labels |
| `ANTHROPIC_MODEL` | claude-haiku-4-5 | LLM model (~$0.002/week for 8 assets) |
