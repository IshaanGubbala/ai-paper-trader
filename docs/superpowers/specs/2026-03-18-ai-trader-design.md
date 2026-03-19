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
│   ├── ml.py       # XGBoost model — daily entry/exit signals
│   └── train.py    # Model training pipeline
├── execution/
│   ├── paper.py    # Paper trading engine (positions, orders, cash)
│   └── portfolio.py # P&L tracking, position sizing
├── backtest/
│   └── runner.py   # Replay historical data through brain + execution
├── state/
│   ├── store.py        # SQLite persistence (portfolio, thesis, trade log)
│   └── migrations/     # Numbered SQL migration scripts (001_init.sql, ...)
├── scheduler.py    # Orchestrates weekly LLM + daily ML runs
└── dashboard.py    # Streamlit UI
```

### Data Flow

1. **Weekly (Monday 8:00 AM ET):** OpenBB fetches macro + fundamentals + news → Claude MCP reads context → outputs JSON thesis per asset → saved to SQLite
2. **Daily (after market close, 4:30 PM ET):** OpenBB fetches OHLCV + technicals → XGBoost scores entry/exit → paper engine executes if thesis agrees → state persisted
3. **Backtest:** Same pipeline, time-stepped through historical data with no live calls; model trained on pre-backtest window

---

## Markets

All assets supported by OpenBB:
- US equities (stocks, ETFs) — long only (no shorting in paper engine v1)
- Crypto — fractional units supported, traded 24/7 (daily run still at 4:30 PM ET for consistency)
- Forex — treated as price ratio pairs, lot-size of 1 unit for paper purposes
- Options — LLM/ML signals generated, but paper engine does NOT execute options orders (signals logged only)

Data providers in use: yfinance (price), Biztoc (news), Alpha Vantage (news/macro), BLS (labor data), Congress.gov (congressional trades).

---

## Brain Layer

### Weekly LLM Thesis — MCP Integration

**Transport:** HTTP POST to `http://127.0.0.1:8001/mcp` using the streamable-http MCP protocol. The request sends a structured prompt via the `tools/call` MCP method.

**Prompt template:** Claude receives a per-asset context block:
```
You are a financial analyst. Given the following data for {asset}, output a JSON trading thesis.

Price (1mo return): {price_return}%
News headlines: {headlines}
Macro: {macro_summary}
Fundamentals: {fundamentals}

Respond ONLY with valid JSON matching this schema:
{"asset": str, "stance": "bullish"|"bearish"|"neutral", "confidence": float 0-1, "reasoning": str, "horizon": str}
```

**JSON enforcement:** Response is parsed with `json.loads()`. If parsing fails, retry once with an explicit correction prompt. If it fails again, stance defaults to `neutral` with confidence `0.0` and the raw response is logged for inspection.

**Timeout:** 30 seconds per asset. On timeout, default to `neutral`.

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
- Technical: RSI(14), MACD(12,26,9), Bollinger Band %B, volume ratio (day/20d avg)
- Momentum: 5/10/20-day returns
- Volatility: ATR(14), 20-day realized volatility

**Output:** `buy` | `sell` | `hold` + probability score

### Model Training

**Target label:** Next-5-day forward return direction. Label = `buy` if return > +1.5%, `sell` if return < -1.5%, `hold` otherwise. This 1.5% threshold filters noise.

**Training data:** 2 years of daily OHLCV history per asset, fetched via OpenBB at system startup and before each backtest.

**Training trigger:**
- Initial: trained once at first launch using 2 years of history
- Live: retrained weekly (Sunday night) on rolling 2-year window
- Backtest: trained on data strictly before the backtest start date (no look-ahead)

**Persistence:** Model saved per asset as `models/{asset}_xgb.joblib`. Feature scaler saved as `models/{asset}_scaler.joblib`. Loaded at startup; reloaded after weekly retraining. Retraining writes to a temp file (`{asset}_xgb.joblib.tmp`) and atomically renames it on completion — preventing partial-write reads by the inference job.

**Feature engineering:** Computed in a shared `features.py` module used by both training and inference to guarantee identical transformation. If an asset has insufficient history for a feature (e.g., fewer than 26 bars for MACD), that asset is skipped for the current run and logged as a warning — it is never traded with incomplete features.

### Gate Logic

Minimum confidence threshold: **0.6** — LLM stances with confidence below 0.6 are treated as neutral regardless of stated stance.

| LLM Stance | ML Signal | Action |
|---|---|---|
| bullish (conf ≥ 0.6) | buy | Enter long |
| bearish (conf ≥ 0.6) | sell | Close long if held; log signal (no shorting v1) |
| neutral OR conf < 0.6 | any | Hold |
| bullish | sell or hold | Hold |
| bearish | buy or hold | Hold |

**Existing position management:**
- If LLM flips to neutral or bearish on a held long position AND ML signal = sell → close position
- If LLM flips to neutral or bearish AND ML signal = hold/buy → keep position, flag for review in dashboard
- Stop-loss and take-profit override all gate logic

---

## Execution Layer

### Paper Trading Engine

- Starting capital: configurable (default $100,000 virtual)
- **Position sizing:** The formula `shares = (portfolio_value × 0.02) / (entry_price × 0.05)` always produces a position worth ~40% of portfolio before capping. The 10% max allocation cap is always binding. **Effective rule: position size = 10% of portfolio value.** Effective stop-loss risk per position = 0.5% of portfolio (not 2%). The formula is retained for documentation of the original risk-based intent.
- Order type: market orders simulated at next-day open price
- Max open positions: 20 (limits UI/cognitive load, not a risk constraint — cash availability is the binding risk constraint)
- Max single asset allocation: 10% of portfolio (hard cap, applied after sizing formula)
- Stop-loss: -5% per position (auto-close at next-day open)
- Take-profit: `entry_price × (1 + 0.05 + (confidence × 0.10))` — ranges from 5% to 15% depending on LLM confidence
- No shorting in v1; no options execution; crypto uses fractional units; forex uses 1-unit lot

### Portfolio Tracker

- Tracks cash, open positions, unrealized/realized P&L
- Trade history log with entry/exit reasons (including which gate condition triggered)
- Daily snapshot saved to SQLite for charting

### State Persistence (SQLite)

All state saved to `trader/state/trader.db`:
- `portfolio` — current cash, positions, P&L
- `thesis` — latest LLM thesis per asset (timestamp, stance, confidence, reasoning)
- `trades` — full trade log
- `snapshots` — daily portfolio value for equity curve. Snapshot is written immediately after all daily orders are applied (4:30 PM ET), representing end-of-day portfolio value.

On restart, state is loaded from SQLite. No data loss on process restart. Schema migrations are applied automatically at startup via numbered scripts in `state/migrations/`. Breaking schema changes in future versions require running a migration script; v1 schema is fixed.

---

## Backtest Runner

- Input: asset list, date range, starting capital
- **Point-in-time data:** Fundamentals (P/E, revenue, debt) are fetched as reported values using only data available before each simulated date. Financials updated quarterly — backtest uses the most recent filing date prior to simulation date. News filtered by publication timestamp.
- ML model trained strictly on data before `backtest_start_date` before the run begins
- **LLM in backtest:** The live Claude MCP endpoint IS called during backtest (one call per asset per simulated week). Responses are cached to `backtest/llm_cache/{run_id}/{asset}_{week}.json` where `week` is the ISO date string of the Monday that started that simulated week (e.g., `2024-01-08`). Re-runs of the same backtest are deterministic and fast via cache. A backtest over 2 years × 10 assets = ~1,040 LLM calls on first run; subsequent runs use cache.
- Steps through time day-by-day; weekly boundary triggers LLM thesis re-run on historical snapshot
- **Output metrics:**
  - Total return (%)
  - Sharpe ratio (annualized, using period-appropriate risk-free rate fetched from FRED series DTB3 for the backtest date range; falls back to 4.5% if FRED unavailable)
  - Max drawdown (computed on daily close equity curve)
  - Win rate (% of closed trades that were profitable)
  - Trade log (entry/exit dates, prices, P&L per trade)

---

## Scheduler

All times in **US/Eastern** timezone:
- **Sunday 11:00 PM ET** — weekly ML model retraining (runs regardless of holidays)
- **Monday 8:00 AM ET** — weekly LLM thesis run. If Monday is a US market holiday, the thesis run is skipped; the previous week's thesis remains active until the next Monday.
- **Daily 4:30 PM ET** — the daily job runs **seven days a week**. On weekends and equity holidays, only crypto assets are processed. On weekdays (non-holidays), all asset classes are processed.
- **Stop-loss enforcement** is checked once daily at 4:30 PM ET using closing prices. Intraday gaps (including overnight crypto moves) are accepted as a known limitation — positions may gap through the stop-loss level.
- **Missed run:** If a scheduled run is missed (process was down), it runs immediately on next startup; does not attempt to replay missed days

---

## Error Handling

- **OpenBB API unreachable:** Log error, skip run, retry next scheduled cycle
- **Incomplete data from provider:** Skip the affected asset for that run; log warning; do not trade with partial data
- **Claude MCP timeout/parse failure:** Default thesis to neutral (see Brain Layer section)
- **Logging:** Structured logs to stdout + `trader/logs/trader.log`, rotating daily, level INFO (DEBUG in backtest mode)

---

## Dashboard (Streamlit)

- Portfolio overview: cash, positions, total P&L
- Active thesis per asset (Claude's reasoning visible, confidence score, timestamp). The `horizon` field from the LLM thesis is displayed here as informational context only — it does not affect trade logic.
- Trade history table with entry/exit reasons
- Backtest results: equity curve chart, metrics table. Backtest runs in a background thread; a progress bar shows simulated date / total range. On completion, results are displayed without blocking the UI.
- Run controls: start/stop live paper trading, launch backtest with date range picker

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
| Persistence | SQLite | Zero-config, sufficient for single-machine use |
| Shorting | Disabled in v1 | Simplifies paper engine; adds complexity for minimal benefit |
| Options execution | Signals only, not executed | Greeks modeling out of scope |
| Timezone | US/Eastern for all scheduling | Aligns with US equity market hours |

---

## Out of Scope

- Real money execution (no broker API integration)
- Options Greeks modeling or execution
- High-frequency / intraday trading
- Portfolio optimization (MPT, etc.)
- Multi-agent debate between LLMs
- Short selling (v1)
- Walk-forward retraining during backtest (model trained once pre-backtest)
