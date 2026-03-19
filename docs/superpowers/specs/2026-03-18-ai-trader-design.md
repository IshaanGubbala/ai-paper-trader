# AI Paper Trader — Design Spec
**Date:** 2026-03-18
**Status:** Approved

## Overview

A paper trading system that uses real market data from OpenBB to drive AI-powered trade decisions. No real money is at risk. The system backtests strategies on historical data and runs live paper trading with real-time data.

**Core idea:** Claude (via local MCP) sets a weekly directional thesis per asset. A local XGBoost ML model generates daily entry/exit signals. A trade only executes when both agree.

---

## Architecture

```
trader/
├── data/           # OpenBB data fetchers (price, news, fundamentals, macro)
├── brain/
│   ├── llm.py      # Claude MCP client — weekly thesis generation
│   └── ml.py       # XGBoost model — daily entry/exit signals
├── execution/
│   ├── paper.py    # Paper trading engine (positions, orders, cash)
│   └── portfolio.py # P&L tracking, position sizing
├── backtest/
│   └── runner.py   # Replay historical data through brain + execution
├── scheduler.py    # Orchestrates weekly LLM + daily ML runs
└── dashboard.py    # Streamlit UI
```

### Data Flow

1. **Weekly:** OpenBB fetches macro + fundamentals + news → Claude MCP reads context → outputs JSON thesis per asset
2. **Daily:** OpenBB fetches OHLCV + technicals → XGBoost scores entry/exit → paper engine executes if thesis agrees
3. **Backtest:** Same pipeline, time-stepped through historical data with no live calls

---

## Markets

All assets supported by OpenBB:
- US equities (stocks, ETFs)
- Crypto
- Forex
- Options (signals only, no complex Greeks modeling)

Data providers in use: yfinance (price), Biztoc (news), Alpha Vantage (news/macro), BLS (labor data), Congress.gov (congressional trades).

---

## Brain Layer

### Weekly LLM Thesis (Claude via MCP at port 8001)

**Input context per asset:**
- 1-month price performance vs sector
- Key news headlines (last 7 days)
- Macro indicators: jobs, rates, GDP
- Fundamentals: P/E, revenue growth, debt-to-equity (equities only)

**Output (JSON):**
```json
{
  "asset": "AAPL",
  "stance": "bullish" | "bearish" | "neutral",
  "confidence": 0.0–1.0,
  "reasoning": "...",
  "horizon": "1-2 weeks"
}
```

### Daily ML Signals (XGBoost, local)

**Features:**
- Technical: RSI, MACD, Bollinger Bands, volume ratio
- Momentum: 5/10/20-day returns
- Volatility: ATR, realized volatility

**Output:** `buy` | `sell` | `hold` + probability score

### Gate Logic

A trade executes only when **both signals agree**:
- LLM stance = bullish AND ML signal = buy → enter long
- LLM stance = bearish AND ML signal = sell → exit / short
- Any mismatch → hold

---

## Execution Layer

### Paper Trading Engine

- Starting capital: configurable (default $100,000 virtual)
- Position sizing: fixed fractional, risk 2% of portfolio per trade
- Order type: market orders (realistic for liquid assets)
- Max open positions: 20
- Max single asset allocation: 10% of portfolio
- Stop-loss: -5% per position (auto-close)
- Take-profit: scales with LLM confidence (higher confidence = wider target)

### Portfolio Tracker

- Tracks cash, open positions, unrealized/realized P&L
- Trade history log with entry/exit reasons
- Daily snapshot for charting

---

## Backtest Runner

- Input: asset list, date range, starting capital
- Steps through time day-by-day, week-by-week
- Weekly boundary: re-runs LLM thesis on historical data snapshot
- Daily step: re-runs ML model, executes paper trades
- Output report:
  - Total return
  - Sharpe ratio
  - Max drawdown
  - Win rate
  - Trade log

---

## Dashboard (Streamlit)

- Portfolio overview: cash, positions, total P&L
- Active thesis per asset (Claude's reasoning visible)
- Trade history table
- Backtest results chart (equity curve)
- Run controls: start/stop live paper trading, launch backtest

---

## Constraints & Decisions

| Decision | Choice | Reason |
|---|---|---|
| LLM | Claude via local MCP (port 8001) | Already running, no API cost |
| ML framework | XGBoost | Fast, interpretable, no GPU needed |
| Broker | None (paper only) | No real money risk |
| Data | OpenBB API (port 6900) | Already running with keys configured |
| UI | Streamlit | Fast to build, sufficient for monitoring |
| Language | Python | OpenBB ecosystem is Python-native |

---

## Out of Scope

- Real money execution (no broker API integration)
- Options Greeks modeling
- High-frequency / intraday trading
- Portfolio optimization (MPT, etc.)
- Multi-agent debate between LLMs
