# AI Paper Trader Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a paper trading system that uses Claude (via local MCP) for weekly directional thesis and XGBoost for daily entry/exit signals, executing paper trades against real market data from OpenBB.

**Architecture:** Data is fetched via the OpenBB Python SDK (`from openbb import obb`). Claude is called via the OpenBB MCP server (port 8001, streamable-http protocol) after an initialize handshake. XGBoost models are trained per-asset and persisted as joblib files. State is stored in SQLite. APScheduler runs the weekly LLM and daily ML jobs. Streamlit provides the dashboard.

**Tech Stack:** Python 3.13, OpenBB SDK, XGBoost 3.2, scikit-learn 1.8, APScheduler 3.11, Streamlit 1.55, SQLite (stdlib), httpx, joblib, pandas 2.3, pandas_market_calendars, pytz

---

## File Map

```
trader/
├── config.py                  # All constants: ports, thresholds, paths, asset list
├── data/
│   ├── __init__.py
│   ├── price.py               # OHLCV + technical indicators via OpenBB SDK
│   ├── news.py                # Headlines via Biztoc + Alpha Vantage
│   ├── fundamentals.py        # P/E, revenue, debt via OpenBB SDK (equities)
│   └── macro.py               # BLS, FRED, unemployment via OpenBB SDK
├── brain/
│   ├── __init__.py
│   ├── features.py            # Shared feature engineering: RSI, MACD, BB, ATR, momentum
│   ├── llm.py                 # Claude MCP client: session, prompt, JSON enforcement, cache
│   ├── ml.py                  # XGBoost inference: load model, predict signal
│   ├── train.py               # XGBoost training pipeline, atomic joblib write
│   └── gate.py                # Gate logic: combine LLM stance + ML signal → action
├── execution/
│   ├── __init__.py
│   ├── paper.py               # Paper engine: open/close positions, stop-loss, take-profit
│   └── portfolio.py           # Position sizing, P&L, daily snapshots
├── backtest/
│   ├── __init__.py
│   └── runner.py              # Time-step simulator, metrics (Sharpe, drawdown, win rate)
├── state/
│   ├── __init__.py
│   ├── store.py               # SQLite CRUD for portfolio, thesis, trades, snapshots
│   └── migrations/
│       └── 001_init.sql       # Schema: portfolio, thesis, trades, snapshots tables
├── tests/
│   ├── conftest.py            # Shared fixtures
│   ├── test_features.py       # Feature engineering unit tests
│   ├── test_gate.py           # Gate logic unit tests (all 5 cases)
│   ├── test_paper.py          # Paper engine unit tests
│   ├── test_portfolio.py      # Position sizing + P&L unit tests
│   ├── test_store.py          # SQLite CRUD tests
│   └── test_backtest.py       # Backtest runner integration test (short window)
├── models/                    # joblib files — gitignored
├── logs/                      # Rotating logs — gitignored
├── scheduler.py               # APScheduler: weekly LLM, weekly retrain, daily ML
└── dashboard.py               # Streamlit UI
```

---

## Task 1: Project Scaffold + Config

**Files:**
- Create: `trader/config.py`
- Create: `trader/__init__.py`, `trader/data/__init__.py`, `trader/brain/__init__.py`, `trader/execution/__init__.py`, `trader/backtest/__init__.py`, `trader/state/__init__.py`, `trader/tests/__init__.py`
- Create: `.gitignore`

- [ ] **Step 1: Create directory structure**

```bash
cd /Users/ishaangubbala/OpenBB
mkdir -p trader/{data,brain,execution,backtest,state/migrations,tests,models,logs}
touch trader/__init__.py trader/data/__init__.py trader/brain/__init__.py
touch trader/execution/__init__.py trader/backtest/__init__.py trader/state/__init__.py
touch trader/tests/__init__.py
```

- [ ] **Step 2: Write config.py**

Create `trader/config.py`:

```python
from pathlib import Path

BASE_DIR = Path(__file__).parent

# OpenBB
OPENBB_API_URL = "http://127.0.0.1:6900"
OPENBB_MCP_URL = "http://127.0.0.1:8001/mcp"

# State
DB_PATH = BASE_DIR / "state" / "trader.db"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
BACKTEST_CACHE_DIR = BASE_DIR / "backtest" / "llm_cache"

# Paper trading
STARTING_CAPITAL = 100_000.0
MAX_POSITIONS = 20
MAX_ALLOCATION = 0.10        # 10% of portfolio per position
STOP_LOSS_PCT = 0.05         # -5%
MIN_LLM_CONFIDENCE = 0.60
LABEL_THRESHOLD = 0.015      # 1.5% for buy/sell label
LABEL_HORIZON = 5            # next-5-day forward return

# Model training
TRAINING_YEARS = 2

# Scheduler (cron-style, US/Eastern)
TIMEZONE = "America/New_York"

# Asset watchlist — edit to add/remove assets
ASSETS = [
    # Equities
    {"symbol": "AAPL",  "type": "equity"},
    {"symbol": "MSFT",  "type": "equity"},
    {"symbol": "NVDA",  "type": "equity"},
    {"symbol": "SPY",   "type": "equity"},
    {"symbol": "QQQ",   "type": "equity"},
    # Crypto
    {"symbol": "BTC",   "type": "crypto"},
    {"symbol": "ETH",   "type": "crypto"},
    # Forex
    {"symbol": "EURUSD","type": "forex"},
]
```

- [ ] **Step 3: Write .gitignore**

Create `/Users/ishaangubbala/OpenBB/.gitignore`:

```
trader/models/
trader/logs/
trader/state/trader.db
trader/backtest/llm_cache/
__pycache__/
*.pyc
.DS_Store
```

- [ ] **Step 4: Verify structure**

```bash
find /Users/ishaangubbala/OpenBB/trader -type f | sort
```
Expected: all `__init__.py` files and `config.py` present.

- [ ] **Step 5: Commit**

```bash
cd /Users/ishaangubbala/OpenBB
git add trader/ .gitignore
git commit -m "feat: scaffold trader project structure and config"
```

---

## Task 2: State Layer — SQLite

**Files:**
- Create: `trader/state/migrations/001_init.sql`
- Create: `trader/state/store.py`
- Test: `trader/tests/test_store.py`

- [ ] **Step 1: Write failing test**

Create `trader/tests/test_store.py`:

```python
import pytest
from pathlib import Path
import tempfile
import os

os.environ["TRADER_DB_PATH"] = str(Path(tempfile.mkdtemp()) / "test.db")

from trader.state.store import Store

@pytest.fixture
def store():
    s = Store(os.environ["TRADER_DB_PATH"])
    s.initialize()
    return s

def test_save_and_load_thesis(store):
    store.save_thesis("AAPL", "bullish", 0.8, "Strong earnings", "1-2 weeks")
    thesis = store.get_thesis("AAPL")
    assert thesis["stance"] == "bullish"
    assert thesis["confidence"] == 0.8
    assert thesis["asset"] == "AAPL"

def test_save_and_load_portfolio(store):
    store.save_portfolio(95000.0, [{"symbol": "AAPL", "shares": 10, "entry_price": 250.0}])
    p = store.get_portfolio()
    assert p["cash"] == 95000.0
    assert len(p["positions"]) == 1

def test_log_trade(store):
    store.log_trade("AAPL", "buy", 10, 250.0, "gate:bullish+buy")
    trades = store.get_trades()
    assert len(trades) == 1
    assert trades[0]["symbol"] == "AAPL"
    assert trades[0]["action"] == "buy"

def test_save_snapshot(store):
    store.save_snapshot(100500.0)
    snaps = store.get_snapshots()
    assert len(snaps) == 1
    assert snaps[0]["value"] == 100500.0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/ishaangubbala/OpenBB
/Users/ishaangubbala/OpenBB/conda/envs/openbb/bin/pytest trader/tests/test_store.py -v
```
Expected: ImportError or ModuleNotFoundError for `trader.state.store`.

- [ ] **Step 3: Write migration SQL**

Create `trader/state/migrations/001_init.sql`:

```sql
CREATE TABLE IF NOT EXISTS portfolio (
    id INTEGER PRIMARY KEY,
    cash REAL NOT NULL,
    positions_json TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS thesis (
    asset TEXT PRIMARY KEY,
    stance TEXT NOT NULL,
    confidence REAL NOT NULL,
    reasoning TEXT,
    horizon TEXT,
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    action TEXT NOT NULL,
    shares REAL NOT NULL,
    price REAL NOT NULL,
    reason TEXT,
    executed_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    value REAL NOT NULL,
    recorded_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY
);

INSERT OR IGNORE INTO schema_version VALUES (1);
```

- [ ] **Step 4: Write store.py**

Create `trader/state/store.py`:

```python
import sqlite3
import json
from pathlib import Path
from datetime import datetime, timezone


class Store:
    def __init__(self, db_path: str | Path | None = None):
        if db_path is None:
            from trader.config import DB_PATH
            db_path = DB_PATH
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def initialize(self):
        migration = Path(__file__).parent / "migrations" / "001_init.sql"
        sql = migration.read_text()
        with self._connect() as conn:
            conn.executescript(sql)

    # --- Thesis ---

    def save_thesis(self, asset: str, stance: str, confidence: float, reasoning: str, horizon: str):
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO thesis (asset, stance, confidence, reasoning, horizon, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (asset, stance, confidence, reasoning, horizon, datetime.now(timezone.utc).isoformat())
            )

    def get_thesis(self, asset: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM thesis WHERE asset = ?", (asset,)).fetchone()
        return dict(row) if row else None

    def get_all_thesis(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM thesis").fetchall()
        return [dict(r) for r in rows]

    # --- Portfolio ---

    def save_portfolio(self, cash: float, positions: list[dict]):
        with self._connect() as conn:
            conn.execute("DELETE FROM portfolio")
            conn.execute(
                "INSERT INTO portfolio (cash, positions_json, updated_at) VALUES (?, ?, ?)",
                (cash, json.dumps(positions), datetime.now(timezone.utc).isoformat())
            )

    def get_portfolio(self) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM portfolio ORDER BY id DESC LIMIT 1").fetchone()
        if row is None:
            return None
        return {"cash": row["cash"], "positions": json.loads(row["positions_json"])}

    # --- Trades ---

    def log_trade(self, symbol: str, action: str, shares: float, price: float, reason: str = ""):
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO trades (symbol, action, shares, price, reason, executed_at) VALUES (?, ?, ?, ?, ?, ?)",
                (symbol, action, shares, price, reason, datetime.now(timezone.utc).isoformat())
            )

    def get_trades(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM trades ORDER BY executed_at DESC").fetchall()
        return [dict(r) for r in rows]

    # --- Snapshots ---

    def save_snapshot(self, value: float):
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO snapshots (value, recorded_at) VALUES (?, ?)",
                (value, datetime.now(timezone.utc).isoformat())
            )

    def get_snapshots(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM snapshots ORDER BY recorded_at ASC").fetchall()
        return [dict(r) for r in rows]
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd /Users/ishaangubbala/OpenBB
/Users/ishaangubbala/OpenBB/conda/envs/openbb/bin/pytest trader/tests/test_store.py -v
```
Expected: 4 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add trader/state/ trader/tests/test_store.py
git commit -m "feat: add SQLite state layer with migrations"
```

---

## Task 3: Data Layer

**Files:**
- Create: `trader/data/price.py`
- Create: `trader/data/news.py`
- Create: `trader/data/fundamentals.py`
- Create: `trader/data/macro.py`

No unit tests for data layer (these call external APIs — tested manually). Integration is verified in Task 5 (features) and Task 8 (LLM).

- [ ] **Step 1: Write price.py**

Create `trader/data/price.py`:

```python
"""Fetch OHLCV history and latest quote via OpenBB SDK."""
import pandas as pd
from datetime import date, timedelta


def get_ohlcv(symbol: str, asset_type: str, days: int = 365 * 2) -> pd.DataFrame:
    """Return daily OHLCV DataFrame with columns: date, open, high, low, close, volume."""
    from openbb import obb
    start = (date.today() - timedelta(days=days)).isoformat()

    try:
        if asset_type == "equity":
            result = obb.equity.price.historical(symbol, start=start, provider="yfinance")
        elif asset_type == "crypto":
            result = obb.crypto.price.historical(symbol, start=start, provider="yfinance")
        elif asset_type == "forex":
            result = obb.currency.price.historical(symbol, start=start, provider="yfinance")
        else:
            raise ValueError(f"Unknown asset type: {asset_type}")
    except Exception as e:
        raise RuntimeError(f"Failed to fetch OHLCV for {symbol}: {e}") from e

    df = result.to_df().reset_index()
    df.columns = [c.lower() for c in df.columns]
    df = df[["date", "open", "high", "low", "close", "volume"]].sort_values("date")
    return df.reset_index(drop=True)


def get_one_month_return(symbol: str, asset_type: str) -> float:
    """Return 1-month price return as a percentage."""
    df = get_ohlcv(symbol, asset_type, days=45)
    if len(df) < 20:
        return 0.0
    return float((df["close"].iloc[-1] / df["close"].iloc[-21] - 1) * 100)
```

- [ ] **Step 2: Write news.py**

Create `trader/data/news.py`:

```python
"""Fetch recent news headlines via OpenBB SDK."""
from datetime import date, timedelta


def get_headlines(symbol: str, limit: int = 10) -> list[str]:
    """Return list of recent headline strings for a symbol."""
    from openbb import obb
    try:
        result = obb.news.company(symbols=symbol, limit=limit, provider="biztoc")
        return [item.title for item in result.results if item.title]
    except Exception:
        pass
    # Fallback: world news search
    try:
        result = obb.news.world(limit=limit, provider="biztoc")
        return [item.title for item in result.results if item.title]
    except Exception:
        return []
```

- [ ] **Step 3: Write fundamentals.py**

Create `trader/data/fundamentals.py`:

```python
"""Fetch fundamental data for equities via OpenBB SDK."""


def get_fundamentals(symbol: str) -> dict:
    """Return dict with pe_ratio, revenue_growth, debt_to_equity. Returns empty dict on failure."""
    from openbb import obb
    out = {}
    try:
        r = obb.equity.fundamental.ratios(symbol, provider="yfinance")
        if r.results:
            item = r.results[0]
            out["pe_ratio"] = getattr(item, "pe_ratio", None)
    except Exception:
        pass
    try:
        r = obb.equity.fundamental.overview(symbol, provider="yfinance")
        if r.results:
            item = r.results[0]
            out["debt_to_equity"] = getattr(item, "total_debt_to_equity", None)
    except Exception:
        pass
    return out
```

- [ ] **Step 4: Write macro.py**

Create `trader/data/macro.py`:

```python
"""Fetch macro indicators via OpenBB SDK."""


def get_macro_summary() -> str:
    """Return a short human-readable macro summary string."""
    from openbb import obb
    lines = []
    try:
        r = obb.economy.unemployment(provider="fred", country="united_states")
        if r.results:
            val = r.results[-1]
            lines.append(f"US Unemployment: {getattr(val, 'value', 'N/A')}%")
    except Exception:
        pass
    try:
        r = obb.economy.cpi(country="united_states", provider="fred", frequency="monthly")
        if r.results:
            val = r.results[-1]
            lines.append(f"US CPI (monthly): {getattr(val, 'value', 'N/A')}")
    except Exception:
        pass
    try:
        r = obb.economy.interest_rates(provider="oecd", country="united_states")
        if r.results:
            val = r.results[-1]
            lines.append(f"US Interest Rate: {getattr(val, 'value', 'N/A')}%")
    except Exception:
        pass
    return "; ".join(lines) if lines else "Macro data unavailable"
```

- [ ] **Step 5: Smoke test data layer**

```bash
cd /Users/ishaangubbala/OpenBB
/Users/ishaangubbala/OpenBB/conda/envs/openbb/bin/python -c "
from trader.data.price import get_ohlcv, get_one_month_return
df = get_ohlcv('AAPL', 'equity', days=30)
print(f'OHLCV rows: {len(df)}, cols: {list(df.columns)}')
ret = get_one_month_return('AAPL', 'equity')
print(f'1mo return: {ret:.2f}%')
"
```
Expected: prints row count ~20 and a return percentage.

- [ ] **Step 6: Commit**

```bash
git add trader/data/
git commit -m "feat: add data layer (price, news, fundamentals, macro)"
```

---

## Task 4: Feature Engineering

**Files:**
- Create: `trader/brain/features.py`
- Test: `trader/tests/test_features.py`

- [ ] **Step 1: Write failing tests**

Create `trader/tests/test_features.py`:

```python
import pytest
import pandas as pd
import numpy as np
from trader.brain.features import compute_features, MIN_BARS

def make_ohlcv(n: int) -> pd.DataFrame:
    """Generate synthetic OHLCV with a mild uptrend."""
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    close = np.maximum(close, 1.0)
    return pd.DataFrame({
        "date": dates,
        "open": close * 0.999,
        "high": close * 1.005,
        "low": close * 0.995,
        "close": close,
        "volume": np.random.randint(1_000_000, 10_000_000, n).astype(float),
    })

def test_requires_min_bars():
    df = make_ohlcv(MIN_BARS - 1)
    with pytest.raises(ValueError, match="Insufficient"):
        compute_features(df)

def test_returns_expected_columns():
    df = make_ohlcv(MIN_BARS + 10)
    features = compute_features(df)
    expected = {"rsi", "macd", "macd_signal", "bb_pct", "volume_ratio",
                "ret_5d", "ret_10d", "ret_20d", "atr", "realized_vol"}
    assert expected.issubset(set(features.columns)), f"Missing: {expected - set(features.columns)}"

def test_no_nans_in_last_row():
    df = make_ohlcv(MIN_BARS + 10)
    features = compute_features(df)
    last = features.iloc[-1]
    nans = last[last.isna()].index.tolist()
    assert nans == [], f"NaN in last row: {nans}"

def test_rsi_bounded():
    df = make_ohlcv(MIN_BARS + 10)
    features = compute_features(df)
    assert features["rsi"].between(0, 100).all()
```

- [ ] **Step 2: Run to verify they fail**

```bash
/Users/ishaangubbala/OpenBB/conda/envs/openbb/bin/pytest trader/tests/test_features.py -v
```
Expected: ImportError for `trader.brain.features`.

- [ ] **Step 3: Write features.py**

Create `trader/brain/features.py`:

```python
"""Shared feature engineering used by both training and inference."""
import pandas as pd
import numpy as np

# Minimum bars needed for all features (MACD needs 26, +20 for vol, +safety)
MIN_BARS = 50

FEATURE_COLS = [
    "rsi", "macd", "macd_signal", "bb_pct", "volume_ratio",
    "ret_5d", "ret_10d", "ret_20d", "atr", "realized_vol",
]


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all ML features from OHLCV DataFrame.
    Requires columns: date, open, high, low, close, volume.
    Returns DataFrame with FEATURE_COLS columns (same row index as input).
    Raises ValueError if fewer than MIN_BARS rows.
    """
    if len(df) < MIN_BARS:
        raise ValueError(f"Insufficient history: need {MIN_BARS} bars, got {len(df)}")

    df = df.copy().sort_values("date").reset_index(drop=True)
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    out = pd.DataFrame(index=df.index)

    # RSI(14)
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    out["rsi"] = 100 - (100 / (1 + rs))

    # MACD(12,26,9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    out["macd"] = ema12 - ema26
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()

    # Bollinger Band %B(20)
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    out["bb_pct"] = (close - (bb_mid - 2 * bb_std)) / (4 * bb_std).replace(0, np.nan)

    # Volume ratio (day / 20d avg)
    out["volume_ratio"] = volume / volume.rolling(20).mean().replace(0, np.nan)

    # Momentum returns
    out["ret_5d"] = close.pct_change(5)
    out["ret_10d"] = close.pct_change(10)
    out["ret_20d"] = close.pct_change(20)

    # ATR(14)
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    out["atr"] = tr.rolling(14).mean()

    # 20-day realized volatility (annualized)
    log_ret = np.log(close / close.shift())
    out["realized_vol"] = log_ret.rolling(20).std() * np.sqrt(252)

    return out[FEATURE_COLS]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
/Users/ishaangubbala/OpenBB/conda/envs/openbb/bin/pytest trader/tests/test_features.py -v
```
Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add trader/brain/features.py trader/tests/test_features.py
git commit -m "feat: add shared feature engineering module (RSI, MACD, BB, ATR, momentum)"
```

---

## Task 5: ML Training Pipeline

**Files:**
- Create: `trader/brain/train.py`

No unit test for training (requires 2yr data download — slow). Verified via smoke test.

- [ ] **Step 1: Write train.py**

Create `trader/brain/train.py`:

```python
"""XGBoost training pipeline. Trains one model per asset."""
import logging
import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date, timedelta

import joblib
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from trader.brain.features import compute_features, MIN_BARS
from trader.config import MODELS_DIR, TRAINING_YEARS, LABEL_THRESHOLD, LABEL_HORIZON

logger = logging.getLogger(__name__)

LABEL_MAP = {"buy": 0, "hold": 1, "sell": 2}
LABEL_NAMES = {0: "buy", 1: "hold", 2: "sell"}


def _make_labels(close: pd.Series) -> pd.Series:
    fwd_return = close.shift(-LABEL_HORIZON) / close - 1
    labels = pd.Series("hold", index=close.index)
    labels[fwd_return > LABEL_THRESHOLD] = "buy"
    labels[fwd_return < -LABEL_THRESHOLD] = "sell"
    return labels


def train_model(symbol: str, asset_type: str, end_date: date | None = None) -> bool:
    """
    Train XGBoost model for a symbol. Saves model + scaler to MODELS_DIR.
    Returns True on success, False on failure (insufficient data, etc.).
    end_date: if set, only use data up to this date (for backtest isolation).
    """
    from trader.data.price import get_ohlcv

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    days = TRAINING_YEARS * 365 + LABEL_HORIZON + MIN_BARS + 10

    try:
        df = get_ohlcv(symbol, asset_type, days=days)
    except Exception as e:
        logger.warning(f"[{symbol}] Data fetch failed: {e}")
        return False

    if end_date:
        df = df[pd.to_datetime(df["date"]).dt.date <= end_date]

    if len(df) < MIN_BARS + LABEL_HORIZON + 10:
        logger.warning(f"[{symbol}] Insufficient data: {len(df)} rows")
        return False

    try:
        features = compute_features(df)
    except ValueError as e:
        logger.warning(f"[{symbol}] Feature computation failed: {e}")
        return False

    labels = _make_labels(df["close"].reset_index(drop=True))

    # Align and drop NaN rows and forward-looking rows
    valid = features.notna().all(axis=1) & labels.notna()
    valid.iloc[-LABEL_HORIZON:] = False  # last rows have no forward label
    X = features[valid]
    y = labels[valid].map(LABEL_MAP)

    if len(X) < 50:
        logger.warning(f"[{symbol}] Too few training samples: {len(X)}")
        return False

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42,
        verbosity=0,
    )
    model.fit(X_scaled, y)

    # Atomic write: tmp file → rename
    model_path = MODELS_DIR / f"{symbol}_xgb.joblib"
    scaler_path = MODELS_DIR / f"{symbol}_scaler.joblib"
    tmp_model = model_path.with_suffix(".joblib.tmp")
    tmp_scaler = scaler_path.with_suffix(".joblib.tmp")

    joblib.dump(model, tmp_model)
    joblib.dump(scaler, tmp_scaler)
    os.replace(tmp_model, model_path)
    os.replace(tmp_scaler, scaler_path)

    logger.info(f"[{symbol}] Model trained on {len(X)} samples, saved to {model_path}")
    return True
```

- [ ] **Step 2: Smoke test training on one asset**

```bash
/Users/ishaangubbala/OpenBB/conda/envs/openbb/bin/python -c "
import logging
logging.basicConfig(level=logging.INFO)
from trader.brain.train import train_model
ok = train_model('AAPL', 'equity')
print('Success:', ok)
import os
from trader.config import MODELS_DIR
print('Files:', list(MODELS_DIR.iterdir()))
"
```
Expected: `Success: True` and two `.joblib` files in `trader/models/`.

- [ ] **Step 3: Commit**

```bash
git add trader/brain/train.py
git commit -m "feat: add XGBoost training pipeline with atomic model persistence"
```

---

## Task 6: ML Inference

**Files:**
- Create: `trader/brain/ml.py`
- Test: `trader/tests/test_features.py` (extend with inference test)

- [ ] **Step 1: Add inference test to test_features.py**

Append to `trader/tests/test_features.py`:

```python
def test_ml_predict_returns_valid_signal():
    """End-to-end: compute features on recent data, run inference."""
    import os
    from pathlib import Path
    from trader.config import MODELS_DIR
    model_file = MODELS_DIR / "AAPL_xgb.joblib"
    if not model_file.exists():
        pytest.skip("AAPL model not trained yet — run Task 5 smoke test first")
    from trader.brain.ml import predict_signal
    df = make_ohlcv(MIN_BARS + 5)
    signal, prob = predict_signal("AAPL", df)
    assert signal in ("buy", "sell", "hold")
    assert 0.0 <= prob <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
/Users/ishaangubbala/OpenBB/conda/envs/openbb/bin/pytest trader/tests/test_features.py::test_ml_predict_returns_valid_signal -v
```
Expected: ImportError for `trader.brain.ml`.

- [ ] **Step 3: Write ml.py**

Create `trader/brain/ml.py`:

```python
"""XGBoost inference: load model and predict signal for latest bar."""
import logging
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from trader.brain.features import compute_features, MIN_BARS
from trader.brain.train import LABEL_NAMES
from trader.config import MODELS_DIR

logger = logging.getLogger(__name__)

_model_cache: dict[str, tuple] = {}  # symbol -> (model, scaler)


def _load_model(symbol: str) -> tuple | None:
    model_path = MODELS_DIR / f"{symbol}_xgb.joblib"
    scaler_path = MODELS_DIR / f"{symbol}_scaler.joblib"
    if not model_path.exists() or not scaler_path.exists():
        return None
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    _model_cache[symbol] = (model, scaler)
    return model, scaler


def reload_model(symbol: str):
    """Force reload model from disk (called after retraining)."""
    _model_cache.pop(symbol, None)
    _load_model(symbol)


def predict_signal(symbol: str, ohlcv: pd.DataFrame) -> tuple[str, float]:
    """
    Predict trading signal for the latest bar.
    Returns (signal, probability) where signal is 'buy'|'sell'|'hold'.
    Returns ('hold', 0.0) if model not available or features insufficient.
    """
    pair = _model_cache.get(symbol) or _load_model(symbol)
    if pair is None:
        logger.warning(f"[{symbol}] No model found — returning hold")
        return "hold", 0.0

    model, scaler = pair

    try:
        features = compute_features(ohlcv)
    except ValueError as e:
        logger.warning(f"[{symbol}] Feature error: {e}")
        return "hold", 0.0

    last = features.iloc[[-1]]
    if last.isna().any().any():
        logger.warning(f"[{symbol}] NaN in features — returning hold")
        return "hold", 0.0

    X = scaler.transform(last)
    probs = model.predict_proba(X)[0]
    pred_idx = int(np.argmax(probs))
    signal = LABEL_NAMES[pred_idx]
    prob = float(probs[pred_idx])
    return signal, prob
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
/Users/ishaangubbala/OpenBB/conda/envs/openbb/bin/pytest trader/tests/test_features.py -v
```
Expected: all tests PASS (inference test skips cleanly if model not yet trained, passes if it is).

- [ ] **Step 5: Commit**

```bash
git add trader/brain/ml.py trader/tests/test_features.py
git commit -m "feat: add XGBoost inference module"
```

---

## Task 7: LLM Thesis Client (Claude MCP)

**Files:**
- Create: `trader/brain/llm.py`
- Test: (no unit test — requires live MCP; tested via smoke test)

The OpenBB MCP server uses streamable-http. Steps:
1. POST to `/mcp` with `initialize` to get `mcp-session-id` header
2. POST with same session ID for `tools/call` or use `sampling/createMessage` to send a prompt
3. Parse SSE response (`data: {...}` lines)

- [ ] **Step 1: Probe the MCP sampling endpoint**

```bash
# Get a session ID
SESSION=$(curl -s -D - -X POST "http://127.0.0.1:8001/mcp" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{"jsonrpc":"2.0","id":"1","method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"trader","version":"1.0"}}}' \
  2>&1 | grep -i "mcp-session-id" | awk '{print $2}' | tr -d '\r')
echo "Session: $SESSION"

# Test sampling/createMessage
curl -s -X POST "http://127.0.0.1:8001/mcp" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -H "mcp-session-id: $SESSION" \
  -d "{\"jsonrpc\":\"2.0\",\"id\":\"2\",\"method\":\"sampling/createMessage\",\"params\":{\"messages\":[{\"role\":\"user\",\"content\":{\"type\":\"text\",\"text\":\"Reply with only: {\\\"test\\\": true}\"}}],\"maxTokens\":50}}" \
  2>&1 | head -20
```

- [ ] **Step 2: Write llm.py based on probe results**

Create `trader/brain/llm.py`:

```python
"""Claude MCP client for weekly thesis generation."""
import json
import logging
import re
from datetime import date
from pathlib import Path

import httpx

from trader.config import OPENBB_MCP_URL, BACKTEST_CACHE_DIR

logger = logging.getLogger(__name__)

MCP_TIMEOUT = 30.0
_session_id: str | None = None


def _get_session() -> str:
    """Initialize MCP session and return session ID."""
    global _session_id
    payload = {
        "jsonrpc": "2.0",
        "id": "init",
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "ai-trader", "version": "1.0"},
        },
    }
    resp = httpx.post(
        OPENBB_MCP_URL,
        json=payload,
        headers={"Accept": "application/json, text/event-stream"},
        timeout=MCP_TIMEOUT,
    )
    resp.raise_for_status()
    _session_id = resp.headers.get("mcp-session-id")
    if not _session_id:
        raise RuntimeError("MCP server did not return a session ID")
    return _session_id


def _parse_sse(text: str) -> dict | None:
    """Extract first JSON object from SSE data lines."""
    for line in text.splitlines():
        if line.startswith("data:"):
            try:
                return json.loads(line[5:].strip())
            except json.JSONDecodeError:
                pass
    return None


def _call_mcp(prompt: str, session_id: str) -> str:
    """Send a sampling/createMessage request and return the text response."""
    payload = {
        "jsonrpc": "2.0",
        "id": "sample",
        "method": "sampling/createMessage",
        "params": {
            "messages": [{"role": "user", "content": {"type": "text", "text": prompt}}],
            "maxTokens": 512,
        },
    }
    resp = httpx.post(
        OPENBB_MCP_URL,
        json=payload,
        headers={
            "Accept": "application/json, text/event-stream",
            "mcp-session-id": session_id,
        },
        timeout=MCP_TIMEOUT,
    )
    resp.raise_for_status()
    data = _parse_sse(resp.text)
    if data is None:
        raise ValueError("No SSE data in MCP response")
    # Extract text from result
    result = data.get("result", {})
    content = result.get("content", {})
    if isinstance(content, dict):
        return content.get("text", "")
    return str(content)


def _parse_thesis(raw: str, asset: str) -> dict:
    """Extract JSON thesis from LLM response. Returns neutral fallback on failure."""
    fallback = {
        "asset": asset,
        "stance": "neutral",
        "confidence": 0.0,
        "reasoning": f"Parse failed. Raw: {raw[:200]}",
        "horizon": "unknown",
    }
    # Try direct parse
    try:
        obj = json.loads(raw.strip())
        if isinstance(obj, dict) and "stance" in obj:
            obj["asset"] = asset
            return obj
    except json.JSONDecodeError:
        pass
    # Try extracting JSON block from prose
    match = re.search(r'\{[^{}]+\}', raw, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group())
            if isinstance(obj, dict) and "stance" in obj:
                obj["asset"] = asset
                return obj
        except json.JSONDecodeError:
            pass
    return fallback


THESIS_PROMPT = """You are a financial analyst. Given the following data for {asset}, output a JSON trading thesis.

Price (1mo return): {price_return:.1f}%
News headlines: {headlines}
Macro: {macro_summary}
Fundamentals: {fundamentals}

Respond ONLY with valid JSON matching exactly this schema (no other text):
{{"asset": "{asset}", "stance": "bullish"|"bearish"|"neutral", "confidence": <float 0.0-1.0>, "reasoning": "<string>", "horizon": "<string>"}}"""


def get_thesis(
    asset: str,
    asset_type: str,
    price_return: float,
    headlines: list[str],
    macro_summary: str,
    fundamentals: dict,
    cache_key: str | None = None,
) -> dict:
    """
    Generate a trading thesis for an asset via Claude MCP.
    cache_key: if set, cache response to disk keyed by this string (for backtest reproducibility).
    Returns thesis dict with keys: asset, stance, confidence, reasoning, horizon.
    """
    # Check cache
    if cache_key:
        cache_file = BACKTEST_CACHE_DIR / f"{asset}_{cache_key}.json"
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        if cache_file.exists():
            logger.debug(f"[{asset}] Using cached thesis for key {cache_key}")
            return json.loads(cache_file.read_text())

    # Skip fundamentals for non-equity
    fund_str = str(fundamentals) if asset_type == "equity" and fundamentals else "N/A"
    headlines_str = "; ".join(headlines[:5]) if headlines else "No recent headlines"

    prompt = THESIS_PROMPT.format(
        asset=asset,
        price_return=price_return,
        headlines=headlines_str,
        macro_summary=macro_summary,
        fundamentals=fund_str,
    )

    try:
        session_id = _session_id or _get_session()
    except Exception as e:
        logger.error(f"[{asset}] MCP session failed: {e}")
        return {"asset": asset, "stance": "neutral", "confidence": 0.0, "reasoning": "MCP unavailable", "horizon": "unknown"}

    try:
        raw = _call_mcp(prompt, session_id)
        thesis = _parse_thesis(raw, asset)
    except Exception as e:
        logger.warning(f"[{asset}] MCP call failed: {e}. Retrying...")
        try:
            correction = f"Previous response could not be parsed. Respond ONLY with valid JSON: {prompt}"
            raw = _call_mcp(correction, session_id)
            thesis = _parse_thesis(raw, asset)
        except Exception as e2:
            logger.error(f"[{asset}] Retry failed: {e2}. Defaulting to neutral.")
            thesis = {"asset": asset, "stance": "neutral", "confidence": 0.0, "reasoning": "MCP error", "horizon": "unknown"}

    # Write cache
    if cache_key:
        cache_file.write_text(json.dumps(thesis))

    return thesis
```

- [ ] **Step 3: Smoke test LLM thesis**

```bash
/Users/ishaangubbala/OpenBB/conda/envs/openbb/bin/python -c "
import logging
logging.basicConfig(level=logging.INFO)
from trader.brain.llm import get_thesis
thesis = get_thesis(
    'AAPL', 'equity',
    price_return=3.5,
    headlines=['Apple reports record earnings', 'iPhone sales surge'],
    macro_summary='US Unemployment: 4.1%; US CPI: 3.2%',
    fundamentals={'pe_ratio': 28.5, 'debt_to_equity': 1.8},
)
import json
print(json.dumps(thesis, indent=2))
"
```
Expected: JSON thesis with stance, confidence, reasoning.

- [ ] **Step 4: Commit**

```bash
git add trader/brain/llm.py
git commit -m "feat: add Claude MCP thesis client with SSE parsing and cache"
```

---

## Task 8: Gate Logic

**Files:**
- Create: `trader/brain/gate.py`
- Test: `trader/tests/test_gate.py`

- [ ] **Step 1: Write failing tests**

Create `trader/tests/test_gate.py`:

```python
import pytest
from trader.brain.gate import evaluate_gate, GateResult

def make_thesis(stance, confidence):
    return {"asset": "AAPL", "stance": stance, "confidence": confidence,
            "reasoning": "test", "horizon": "1w"}

def test_bullish_high_conf_buy_enters():
    result = evaluate_gate(make_thesis("bullish", 0.8), "buy", 0.85, has_position=False)
    assert result == GateResult.ENTER

def test_bearish_high_conf_sell_closes():
    result = evaluate_gate(make_thesis("bearish", 0.7), "sell", 0.80, has_position=True)
    assert result == GateResult.CLOSE

def test_bearish_no_position_is_hold():
    result = evaluate_gate(make_thesis("bearish", 0.7), "sell", 0.80, has_position=False)
    assert result == GateResult.HOLD

def test_neutral_stance_is_hold():
    result = evaluate_gate(make_thesis("neutral", 0.9), "buy", 0.90, has_position=False)
    assert result == GateResult.HOLD

def test_low_confidence_is_hold():
    result = evaluate_gate(make_thesis("bullish", 0.5), "buy", 0.90, has_position=False)
    assert result == GateResult.HOLD

def test_bullish_ml_sell_is_hold():
    result = evaluate_gate(make_thesis("bullish", 0.8), "sell", 0.80, has_position=False)
    assert result == GateResult.HOLD

def test_bearish_flip_with_ml_sell_closes_position():
    result = evaluate_gate(make_thesis("bearish", 0.75), "sell", 0.80, has_position=True)
    assert result == GateResult.CLOSE

def test_bearish_flip_with_ml_hold_keeps_position():
    result = evaluate_gate(make_thesis("bearish", 0.75), "hold", 0.50, has_position=True)
    assert result == GateResult.HOLD
```

- [ ] **Step 2: Run to verify they fail**

```bash
/Users/ishaangubbala/OpenBB/conda/envs/openbb/bin/pytest trader/tests/test_gate.py -v
```
Expected: ImportError for `trader.brain.gate`.

- [ ] **Step 3: Write gate.py**

Create `trader/brain/gate.py`:

```python
"""Gate logic: combines LLM thesis and ML signal to produce a trade action."""
from enum import Enum
from trader.config import MIN_LLM_CONFIDENCE


class GateResult(str, Enum):
    ENTER = "enter"   # Open a new long position
    CLOSE = "close"   # Close an existing long position
    HOLD  = "hold"    # Do nothing


def evaluate_gate(
    thesis: dict,
    ml_signal: str,
    ml_prob: float,
    has_position: bool,
) -> GateResult:
    """
    Evaluate the gate and return the action to take.

    thesis: dict with keys stance, confidence
    ml_signal: 'buy' | 'sell' | 'hold'
    ml_prob: ML probability score (0-1)
    has_position: whether we currently hold this asset
    """
    stance = thesis.get("stance", "neutral")
    confidence = float(thesis.get("confidence", 0.0))

    # Treat low-confidence stance as neutral
    effective_stance = stance if confidence >= MIN_LLM_CONFIDENCE else "neutral"

    if effective_stance == "bullish" and ml_signal == "buy":
        return GateResult.ENTER if not has_position else GateResult.HOLD

    if effective_stance == "bearish" and ml_signal == "sell":
        return GateResult.CLOSE if has_position else GateResult.HOLD

    # All other combinations: hold
    return GateResult.HOLD
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
/Users/ishaangubbala/OpenBB/conda/envs/openbb/bin/pytest trader/tests/test_gate.py -v
```
Expected: 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add trader/brain/gate.py trader/tests/test_gate.py
git commit -m "feat: add gate logic combining LLM thesis and ML signal"
```

---

## Task 9: Portfolio Tracker + Position Sizing

**Files:**
- Create: `trader/execution/portfolio.py`
- Test: `trader/tests/test_portfolio.py`

- [ ] **Step 1: Write failing tests**

Create `trader/tests/test_portfolio.py`:

```python
import pytest
from trader.execution.portfolio import Portfolio

def test_initial_state():
    p = Portfolio(starting_capital=100_000.0)
    assert p.cash == 100_000.0
    assert p.positions == {}

def test_position_size_is_10pct():
    p = Portfolio(starting_capital=100_000.0)
    size = p.compute_position_size("AAPL", entry_price=250.0)
    dollar_value = size * 250.0
    assert abs(dollar_value - 10_000.0) < 1.0  # 10% of 100k

def test_open_position_reduces_cash():
    p = Portfolio(starting_capital=100_000.0)
    shares = p.compute_position_size("AAPL", entry_price=250.0)
    p.open_position("AAPL", shares, entry_price=250.0, take_profit_price=262.5, stop_loss_price=237.5)
    assert p.cash < 100_000.0
    assert "AAPL" in p.positions

def test_close_position_restores_cash():
    p = Portfolio(starting_capital=100_000.0)
    shares = p.compute_position_size("AAPL", entry_price=250.0)
    p.open_position("AAPL", shares, entry_price=250.0, take_profit_price=262.5, stop_loss_price=237.5)
    pnl = p.close_position("AAPL", exit_price=260.0, reason="test")
    assert pnl > 0
    assert "AAPL" not in p.positions

def test_max_positions_respected():
    p = Portfolio(starting_capital=1_000_000.0)
    for i in range(20):
        p.open_position(f"SYM{i}", 10, entry_price=100.0, take_profit_price=110.0, stop_loss_price=95.0)
    # 21st should be rejected
    ok = p.open_position("SYM20", 10, entry_price=100.0, take_profit_price=110.0, stop_loss_price=95.0)
    assert ok is False

def test_portfolio_value():
    p = Portfolio(starting_capital=100_000.0)
    shares = p.compute_position_size("AAPL", entry_price=250.0)
    p.open_position("AAPL", shares, entry_price=250.0, take_profit_price=262.5, stop_loss_price=237.5)
    prices = {"AAPL": 260.0}
    value = p.total_value(prices)
    assert value > 100_000.0  # position gained
```

- [ ] **Step 2: Run to verify they fail**

```bash
/Users/ishaangubbala/OpenBB/conda/envs/openbb/bin/pytest trader/tests/test_portfolio.py -v
```
Expected: ImportError for `trader.execution.portfolio`.

- [ ] **Step 3: Write portfolio.py**

Create `trader/execution/portfolio.py`:

```python
"""Portfolio tracker: position sizing, P&L, cash management."""
import logging
from dataclasses import dataclass, field
from trader.config import MAX_POSITIONS, MAX_ALLOCATION

logger = logging.getLogger(__name__)


@dataclass
class Position:
    symbol: str
    shares: float
    entry_price: float
    take_profit_price: float
    stop_loss_price: float
    confidence: float = 0.7
    reason: str = ""


class Portfolio:
    def __init__(self, starting_capital: float):
        self.cash = starting_capital
        self._starting_capital = starting_capital
        self.positions: dict[str, Position] = {}
        self.closed_trades: list[dict] = []

    def compute_position_size(self, symbol: str, entry_price: float) -> float:
        """Return number of shares to buy (10% of portfolio, fractional ok)."""
        dollar_size = self._starting_capital * MAX_ALLOCATION
        # Use current portfolio value for dynamic sizing
        dollar_size = (self.cash + self._position_value_at_entry()) * MAX_ALLOCATION
        return dollar_size / entry_price

    def _position_value_at_entry(self) -> float:
        return sum(p.shares * p.entry_price for p in self.positions.values())

    def open_position(
        self,
        symbol: str,
        shares: float,
        entry_price: float,
        take_profit_price: float,
        stop_loss_price: float,
        confidence: float = 0.7,
        reason: str = "",
    ) -> bool:
        """Open a position. Returns False if rejected (max positions, insufficient cash)."""
        if symbol in self.positions:
            logger.debug(f"[{symbol}] Already have position")
            return False
        if len(self.positions) >= MAX_POSITIONS:
            logger.warning(f"[{symbol}] Max positions ({MAX_POSITIONS}) reached")
            return False
        cost = shares * entry_price
        if cost > self.cash:
            # Adjust shares to available cash
            shares = self.cash / entry_price * 0.99  # 1% buffer
            cost = shares * entry_price
        if shares <= 0:
            return False
        self.cash -= cost
        self.positions[symbol] = Position(
            symbol=symbol,
            shares=shares,
            entry_price=entry_price,
            take_profit_price=take_profit_price,
            stop_loss_price=stop_loss_price,
            confidence=confidence,
            reason=reason,
        )
        logger.info(f"[{symbol}] Opened {shares:.4f} shares @ {entry_price:.2f} (cost: {cost:.2f})")
        return True

    def close_position(self, symbol: str, exit_price: float, reason: str = "") -> float:
        """Close a position. Returns realized P&L."""
        pos = self.positions.pop(symbol, None)
        if pos is None:
            return 0.0
        proceeds = pos.shares * exit_price
        pnl = proceeds - (pos.shares * pos.entry_price)
        self.cash += proceeds
        self.closed_trades.append({
            "symbol": symbol,
            "shares": pos.shares,
            "entry_price": pos.entry_price,
            "exit_price": exit_price,
            "pnl": pnl,
            "reason": reason,
        })
        logger.info(f"[{symbol}] Closed @ {exit_price:.2f}, P&L: {pnl:.2f} ({reason})")
        return pnl

    def total_value(self, current_prices: dict[str, float]) -> float:
        """Total portfolio value given current prices."""
        pos_value = sum(
            p.shares * current_prices.get(p.symbol, p.entry_price)
            for p in self.positions.values()
        )
        return self.cash + pos_value

    def to_dict(self) -> dict:
        return {
            "cash": self.cash,
            "positions": [
                {
                    "symbol": p.symbol,
                    "shares": p.shares,
                    "entry_price": p.entry_price,
                    "take_profit_price": p.take_profit_price,
                    "stop_loss_price": p.stop_loss_price,
                    "confidence": p.confidence,
                }
                for p in self.positions.values()
            ],
        }

    @classmethod
    def from_dict(cls, data: dict, starting_capital: float) -> "Portfolio":
        p = cls(starting_capital)
        p.cash = data["cash"]
        for pos in data.get("positions", []):
            p.positions[pos["symbol"]] = Position(**pos)
        return p
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
/Users/ishaangubbala/OpenBB/conda/envs/openbb/bin/pytest trader/tests/test_portfolio.py -v
```
Expected: 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add trader/execution/portfolio.py trader/tests/test_portfolio.py
git commit -m "feat: add portfolio tracker with position sizing and P&L"
```

---

## Task 10: Paper Trading Engine

**Files:**
- Create: `trader/execution/paper.py`
- Test: `trader/tests/test_paper.py`

- [ ] **Step 1: Write failing tests**

Create `trader/tests/test_paper.py`:

```python
import pytest
from unittest.mock import MagicMock
from trader.execution.paper import PaperEngine
from trader.execution.portfolio import Portfolio
from trader.brain.gate import GateResult

def make_engine():
    portfolio = Portfolio(100_000.0)
    store = MagicMock()
    store.get_portfolio.return_value = None
    return PaperEngine(portfolio=portfolio, store=store)

def test_enter_signal_opens_position():
    engine = make_engine()
    engine.process_signal("AAPL", GateResult.ENTER, entry_price=250.0, confidence=0.8)
    assert "AAPL" in engine.portfolio.positions

def test_close_signal_closes_position():
    engine = make_engine()
    engine.process_signal("AAPL", GateResult.ENTER, entry_price=250.0, confidence=0.8)
    engine.process_signal("AAPL", GateResult.CLOSE, entry_price=260.0, confidence=0.8)
    assert "AAPL" not in engine.portfolio.positions

def test_stop_loss_triggers():
    engine = make_engine()
    engine.process_signal("AAPL", GateResult.ENTER, entry_price=250.0, confidence=0.8)
    closed = engine.check_stops({"AAPL": 236.0})  # below -5%
    assert "AAPL" in closed

def test_take_profit_triggers():
    engine = make_engine()
    engine.process_signal("AAPL", GateResult.ENTER, entry_price=250.0, confidence=0.8)
    pos = engine.portfolio.positions["AAPL"]
    closed = engine.check_stops({"AAPL": pos.take_profit_price + 1.0})
    assert "AAPL" in closed

def test_hold_signal_does_nothing():
    engine = make_engine()
    engine.process_signal("AAPL", GateResult.HOLD, entry_price=250.0, confidence=0.8)
    assert "AAPL" not in engine.portfolio.positions
```

- [ ] **Step 2: Run to verify they fail**

```bash
/Users/ishaangubbala/OpenBB/conda/envs/openbb/bin/pytest trader/tests/test_paper.py -v
```
Expected: ImportError for `trader.execution.paper`.

- [ ] **Step 3: Write paper.py**

Create `trader/execution/paper.py`:

```python
"""Paper trading engine: processes gate signals and manages stop/take-profit."""
import logging
from trader.execution.portfolio import Portfolio
from trader.brain.gate import GateResult
from trader.config import STOP_LOSS_PCT

logger = logging.getLogger(__name__)


class PaperEngine:
    def __init__(self, portfolio: Portfolio, store):
        self.portfolio = portfolio
        self.store = store

    def process_signal(
        self,
        symbol: str,
        action: GateResult,
        entry_price: float,
        confidence: float,
        reason: str = "",
    ):
        """Process a gate decision and execute paper trade if warranted."""
        if action == GateResult.ENTER:
            # Compute take-profit: 5% + (confidence * 10%) above entry
            take_profit = entry_price * (1 + 0.05 + confidence * 0.10)
            stop_loss = entry_price * (1 - STOP_LOSS_PCT)
            shares = self.portfolio.compute_position_size(symbol, entry_price)
            opened = self.portfolio.open_position(
                symbol, shares, entry_price, take_profit, stop_loss,
                confidence=confidence, reason=reason or "gate:bullish+buy",
            )
            if opened:
                self.store.log_trade(symbol, "buy", shares, entry_price, reason or "gate:bullish+buy")

        elif action == GateResult.CLOSE:
            pnl = self.portfolio.close_position(symbol, entry_price, reason=reason or "gate:bearish+sell")
            if pnl != 0.0:
                pos_shares = 0.0  # already closed
                self.store.log_trade(symbol, "sell", 0.0, entry_price, reason or "gate:bearish+sell")

    def check_stops(self, current_prices: dict[str, float]) -> list[str]:
        """
        Check all positions against stop-loss and take-profit levels.
        Returns list of symbols that were closed.
        """
        closed = []
        for symbol, pos in list(self.portfolio.positions.items()):
            price = current_prices.get(symbol)
            if price is None:
                continue
            if price <= pos.stop_loss_price:
                pnl = self.portfolio.close_position(symbol, price, reason="stop_loss")
                self.store.log_trade(symbol, "sell", pos.shares, price, "stop_loss")
                closed.append(symbol)
                logger.info(f"[{symbol}] Stop-loss triggered @ {price:.2f}")
            elif price >= pos.take_profit_price:
                pnl = self.portfolio.close_position(symbol, price, reason="take_profit")
                self.store.log_trade(symbol, "sell", pos.shares, price, "take_profit")
                closed.append(symbol)
                logger.info(f"[{symbol}] Take-profit triggered @ {price:.2f}")
        return closed

    def persist(self):
        """Save current portfolio state to SQLite."""
        d = self.portfolio.to_dict()
        self.store.save_portfolio(d["cash"], d["positions"])
        self.store.save_snapshot(self.portfolio.total_value({}))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
/Users/ishaangubbala/OpenBB/conda/envs/openbb/bin/pytest trader/tests/test_paper.py -v
```
Expected: 5 tests PASS.

- [ ] **Step 5: Run all tests so far**

```bash
/Users/ishaangubbala/OpenBB/conda/envs/openbb/bin/pytest trader/tests/ -v --ignore=trader/tests/test_backtest.py
```
Expected: All pass.

- [ ] **Step 6: Commit**

```bash
git add trader/execution/paper.py trader/tests/test_paper.py
git commit -m "feat: add paper trading engine with stop-loss and take-profit"
```

---

## Task 11: Scheduler

**Files:**
- Create: `trader/scheduler.py`

No unit test (scheduler logic is integration-level). Tested via dry-run.

- [ ] **Step 1: Write scheduler.py**

Create `trader/scheduler.py`:

```python
"""APScheduler-based orchestrator for weekly LLM + daily ML runs."""
import logging
import sys
from datetime import datetime

import pytz
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
import pandas_market_calendars as mcal

from trader.config import ASSETS, STARTING_CAPITAL, TIMEZONE, MODELS_DIR

logger = logging.getLogger(__name__)
nyse = mcal.get_calendar("NYSE")


def _is_equity_trading_day(dt: datetime) -> bool:
    """Return True if dt is a NYSE trading day."""
    schedule = nyse.schedule(start_date=dt.date(), end_date=dt.date())
    return not schedule.empty


def run_weekly_retrain():
    """Sunday 11 PM ET: retrain all models on rolling 2-year window."""
    logger.info("=== Weekly model retraining started ===")
    from trader.brain.train import train_model
    for asset in ASSETS:
        ok = train_model(asset["symbol"], asset["type"])
        logger.info(f"[{asset['symbol']}] Retrain: {'OK' if ok else 'FAILED'}")
    logger.info("=== Weekly model retraining complete ===")


def run_weekly_thesis():
    """Monday 8 AM ET: generate LLM thesis for all assets. Skip if holiday."""
    tz = pytz.timezone(TIMEZONE)
    now = datetime.now(tz)
    if not _is_equity_trading_day(now):
        logger.info("Monday is a market holiday — skipping LLM thesis run")
        return

    logger.info("=== Weekly LLM thesis run started ===")
    from trader.brain.llm import get_thesis
    from trader.brain.llm import _get_session
    from trader.data.price import get_one_month_return
    from trader.data.news import get_headlines
    from trader.data.macro import get_macro_summary
    from trader.data.fundamentals import get_fundamentals
    from trader.state.store import Store

    store = Store()
    macro = get_macro_summary()

    for asset in ASSETS:
        symbol = asset["symbol"]
        try:
            price_return = get_one_month_return(symbol, asset["type"])
            headlines = get_headlines(symbol)
            fundamentals = get_fundamentals(symbol) if asset["type"] == "equity" else {}
            thesis = get_thesis(symbol, asset["type"], price_return, headlines, macro, fundamentals)
            store.save_thesis(symbol, thesis["stance"], thesis["confidence"],
                              thesis.get("reasoning", ""), thesis.get("horizon", ""))
            logger.info(f"[{symbol}] Thesis: {thesis['stance']} ({thesis['confidence']:.2f})")
        except Exception as e:
            logger.error(f"[{symbol}] Thesis failed: {e}")
    logger.info("=== Weekly LLM thesis run complete ===")


def run_daily_signals():
    """Daily 4:30 PM ET: run ML signals and execute paper trades."""
    tz = pytz.timezone(TIMEZONE)
    now = datetime.now(tz)
    is_trading_day = _is_equity_trading_day(now)

    logger.info(f"=== Daily signal run started (trading_day={is_trading_day}) ===")

    from trader.data.price import get_ohlcv
    from trader.brain.ml import predict_signal, reload_model
    from trader.brain.gate import evaluate_gate
    from trader.execution.portfolio import Portfolio
    from trader.execution.paper import PaperEngine
    from trader.state.store import Store

    store = Store()
    store.initialize()

    # Load or init portfolio
    saved = store.get_portfolio()
    if saved:
        portfolio = Portfolio.from_dict(saved, STARTING_CAPITAL)
    else:
        portfolio = Portfolio(STARTING_CAPITAL)

    engine = PaperEngine(portfolio=portfolio, store=store)

    current_prices: dict[str, float] = {}

    for asset in ASSETS:
        symbol = asset["symbol"]
        asset_type = asset["type"]

        # Only process equities on trading days
        if asset_type == "equity" and not is_trading_day:
            continue

        try:
            df = get_ohlcv(symbol, asset_type, days=90)
        except Exception as e:
            logger.warning(f"[{symbol}] OHLCV fetch failed: {e}")
            continue

        if df.empty:
            continue

        current_price = float(df["close"].iloc[-1])
        current_prices[symbol] = current_price

        thesis_row = store.get_thesis(symbol)
        if not thesis_row:
            logger.warning(f"[{symbol}] No thesis — skipping")
            continue
        thesis = {"stance": thesis_row["stance"], "confidence": thesis_row["confidence"]}

        signal, prob = predict_signal(symbol, df)
        has_position = symbol in portfolio.positions
        action = evaluate_gate(thesis, signal, prob, has_position)

        logger.info(f"[{symbol}] price={current_price:.2f} signal={signal} stance={thesis['stance']} action={action}")
        engine.process_signal(symbol, action, entry_price=current_price, confidence=thesis_row["confidence"])

    # Check stops for all positions
    engine.check_stops(current_prices)

    # Persist state + snapshot
    total = portfolio.total_value(current_prices)
    store.save_portfolio(portfolio.to_dict()["cash"], portfolio.to_dict()["positions"])
    store.save_snapshot(total)

    logger.info(f"=== Daily signal run complete. Portfolio value: ${total:,.2f} ===")


def run_missed_jobs_if_needed():
    """On startup, run any jobs that should have run while process was down."""
    # Simple approach: if no thesis in DB, run thesis now
    from trader.state.store import Store
    store = Store()
    store.initialize()
    all_thesis = store.get_all_thesis()
    if not all_thesis:
        logger.info("No thesis found on startup — running initial thesis generation")
        run_weekly_thesis()

    # If no models exist, train now
    if not any(MODELS_DIR.glob("*_xgb.joblib")):
        logger.info("No models found on startup — running initial training")
        run_weekly_retrain()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )

    tz = pytz.timezone(TIMEZONE)
    scheduler = BlockingScheduler(timezone=tz)

    # Sunday 11 PM ET — weekly retraining
    scheduler.add_job(run_weekly_retrain, CronTrigger(day_of_week="sun", hour=23, minute=0, timezone=tz))
    # Monday 8 AM ET — weekly thesis
    scheduler.add_job(run_weekly_thesis, CronTrigger(day_of_week="mon", hour=8, minute=0, timezone=tz))
    # Daily 4:30 PM ET — signals + execution
    scheduler.add_job(run_daily_signals, CronTrigger(hour=16, minute=30, timezone=tz))

    run_missed_jobs_if_needed()

    logger.info("Scheduler started. Press Ctrl+C to stop.")
    try:
        scheduler.start()
    except KeyboardInterrupt:
        logger.info("Scheduler stopped.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Dry-run the daily signal job**

```bash
/Users/ishaangubbala/OpenBB/conda/envs/openbb/bin/python -c "
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
from trader.scheduler import run_daily_signals
run_daily_signals()
"
```
Expected: runs without crashing, logs signal decisions per asset.

- [ ] **Step 3: Commit**

```bash
git add trader/scheduler.py
git commit -m "feat: add APScheduler orchestrator with weekly LLM + daily ML jobs"
```

---

## Task 12: Backtest Runner

**Files:**
- Create: `trader/backtest/runner.py`
- Test: `trader/tests/test_backtest.py`

- [ ] **Step 1: Write failing test**

Create `trader/tests/test_backtest.py`:

```python
import pytest
from datetime import date

def test_backtest_runs_and_returns_metrics():
    """Short 3-month backtest on a single equity."""
    from trader.backtest.runner import run_backtest
    result = run_backtest(
        assets=[{"symbol": "AAPL", "type": "equity"}],
        start_date=date(2024, 1, 1),
        end_date=date(2024, 3, 31),
        starting_capital=100_000.0,
        run_id="test_run",
    )
    assert "total_return_pct" in result
    assert "sharpe_ratio" in result
    assert "max_drawdown_pct" in result
    assert "win_rate" in result
    assert "trades" in result
    assert isinstance(result["total_return_pct"], float)
```

- [ ] **Step 2: Run to verify it fails**

```bash
/Users/ishaangubbala/OpenBB/conda/envs/openbb/bin/pytest trader/tests/test_backtest.py -v
```
Expected: ImportError for `trader.backtest.runner`.

- [ ] **Step 3: Write runner.py**

Create `trader/backtest/runner.py`:

```python
"""Backtest runner: time-step simulator using the same brain + execution pipeline."""
import logging
from datetime import date, timedelta
from typing import Callable

import numpy as np
import pandas as pd

from trader.config import BACKTEST_CACHE_DIR
from trader.brain.features import MIN_BARS
from trader.brain.gate import evaluate_gate
from trader.execution.portfolio import Portfolio

logger = logging.getLogger(__name__)

RISK_FREE_RATE = 0.045  # Fallback if FRED unavailable


def _get_risk_free_rate(start_date: date, end_date: date) -> float:
    """Fetch average T-bill rate from FRED for the period. Falls back to 0.045."""
    try:
        from openbb import obb
        r = obb.economy.fred_series(
            symbol="DTB3",
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
        )
        vals = [v.value for v in r.results if v.value is not None]
        if vals:
            return float(np.mean(vals)) / 100.0
    except Exception as e:
        logger.warning(f"FRED rate fetch failed: {e}. Using {RISK_FREE_RATE:.1%}")
    return RISK_FREE_RATE


def _compute_metrics(equity_curve: list[float], trades: list[dict], rf_rate: float) -> dict:
    arr = np.array(equity_curve)
    if len(arr) < 2:
        return {"total_return_pct": 0.0, "sharpe_ratio": 0.0, "max_drawdown_pct": 0.0, "win_rate": 0.0}

    total_return = (arr[-1] / arr[0] - 1) * 100

    daily_returns = np.diff(arr) / arr[:-1]
    excess = daily_returns - rf_rate / 252
    sharpe = float(np.mean(excess) / np.std(excess) * np.sqrt(252)) if np.std(excess) > 0 else 0.0

    running_max = np.maximum.accumulate(arr)
    drawdowns = (arr - running_max) / running_max
    max_drawdown = float(np.min(drawdowns) * 100)

    profitable = [t for t in trades if t.get("pnl", 0) > 0]
    win_rate = len(profitable) / len(trades) * 100 if trades else 0.0

    return {
        "total_return_pct": float(total_return),
        "sharpe_ratio": sharpe,
        "max_drawdown_pct": max_drawdown,
        "win_rate": float(win_rate),
    }


def run_backtest(
    assets: list[dict],
    start_date: date,
    end_date: date,
    starting_capital: float = 100_000.0,
    run_id: str = "default",
    progress_callback: Callable[[date], None] | None = None,
) -> dict:
    """
    Run a backtest over [start_date, end_date].
    Returns metrics dict + trade log.
    """
    from trader.data.price import get_ohlcv
    from trader.data.news import get_headlines
    from trader.data.macro import get_macro_summary
    from trader.data.fundamentals import get_fundamentals
    from trader.brain.llm import get_thesis
    from trader.brain.ml import predict_signal
    from trader.brain.train import train_model

    # Pre-fetch 2yr history per asset for feature computation
    logger.info(f"[Backtest] Fetching historical data for {len(assets)} assets...")
    asset_data: dict[str, pd.DataFrame] = {}
    for asset in assets:
        sym = asset["symbol"]
        try:
            df = get_ohlcv(sym, asset["type"], days=365 * 3)
            asset_data[sym] = df
        except Exception as e:
            logger.warning(f"[{sym}] Data fetch failed: {e}")

    # Train models on data before start_date
    logger.info("[Backtest] Training models on pre-backtest data...")
    for asset in assets:
        sym = asset["symbol"]
        train_model(sym, asset["type"], end_date=start_date - timedelta(days=1))

    rf_rate = _get_risk_free_rate(start_date, end_date)
    portfolio = Portfolio(starting_capital)
    equity_curve: list[float] = [starting_capital]
    all_trades: list[dict] = []
    current_thesis: dict[str, dict] = {}

    # Step through days
    current = start_date
    macro_summary = get_macro_summary()  # fetch once (macro changes slowly)

    while current <= end_date:
        if progress_callback:
            progress_callback(current)

        # Weekly: Monday — regenerate LLM thesis
        if current.weekday() == 0:
            for asset in assets:
                sym = asset["symbol"]
                df = asset_data.get(sym, pd.DataFrame())
                hist = df[pd.to_datetime(df["date"]).dt.date <= current] if not df.empty else df
                if hist.empty:
                    continue
                price_return = float((hist["close"].iloc[-1] / hist["close"].iloc[-min(21, len(hist)-1)] - 1) * 100)
                week_key = current.isoformat()
                cache_key = f"{run_id}_{week_key}"
                thesis = get_thesis(sym, asset["type"], price_return, [], macro_summary, {}, cache_key=cache_key)
                current_thesis[sym] = thesis

        # Daily: process signals
        current_prices: dict[str, float] = {}
        for asset in assets:
            sym = asset["symbol"]
            df = asset_data.get(sym, pd.DataFrame())
            hist = df[pd.to_datetime(df["date"]).dt.date <= current] if not df.empty else df
            if len(hist) < MIN_BARS:
                continue

            price = float(hist["close"].iloc[-1])
            current_prices[sym] = price

            thesis = current_thesis.get(sym, {"stance": "neutral", "confidence": 0.0})
            signal, prob = predict_signal(sym, hist)
            has_position = sym in portfolio.positions
            action = evaluate_gate(thesis, signal, prob, has_position)

            if action.value == "enter":
                tp = price * (1 + 0.05 + thesis.get("confidence", 0.7) * 0.10)
                sl = price * 0.95
                shares = portfolio.compute_position_size(sym, price)
                portfolio.open_position(sym, shares, price, tp, sl, confidence=thesis.get("confidence", 0.7))

            elif action.value == "close":
                pnl = portfolio.close_position(sym, price, reason="gate:bearish")
                if pnl != 0:
                    all_trades.extend([t for t in portfolio.closed_trades[-1:]])

        # Check stops
        for sym, pos in list(portfolio.positions.items()):
            p = current_prices.get(sym)
            if p is None:
                continue
            if p <= pos.stop_loss_price or p >= pos.take_profit_price:
                reason = "stop_loss" if p <= pos.stop_loss_price else "take_profit"
                pnl = portfolio.close_position(sym, p, reason=reason)
                all_trades.extend([t for t in portfolio.closed_trades[-1:]])

        total = portfolio.total_value(current_prices)
        equity_curve.append(total)
        current += timedelta(days=1)

    metrics = _compute_metrics(equity_curve, all_trades, rf_rate)
    metrics["trades"] = all_trades
    metrics["equity_curve"] = equity_curve
    logger.info(f"[Backtest] Complete. Return: {metrics['total_return_pct']:.1f}%, Sharpe: {metrics['sharpe_ratio']:.2f}")
    return metrics
```

- [ ] **Step 4: Run backtest test (this fetches live data — allow ~2 minutes)**

```bash
/Users/ishaangubbala/OpenBB/conda/envs/openbb/bin/pytest trader/tests/test_backtest.py -v -s
```
Expected: PASS with metrics printed. May take 1-2 minutes to fetch data and call LLM.

- [ ] **Step 5: Commit**

```bash
git add trader/backtest/ trader/tests/test_backtest.py
git commit -m "feat: add backtest runner with time-step simulation and performance metrics"
```

---

## Task 13: Streamlit Dashboard

**Files:**
- Create: `trader/dashboard.py`

No unit test (UI). Tested by running locally.

- [ ] **Step 1: Write dashboard.py**

Create `trader/dashboard.py`:

```python
"""Streamlit dashboard for monitoring the AI paper trader."""
import threading
from datetime import date

import streamlit as st
import pandas as pd

from trader.state.store import Store
from trader.config import STARTING_CAPITAL, ASSETS

st.set_page_config(page_title="AI Paper Trader", layout="wide")

store = Store()
store.initialize()


def get_portfolio_summary():
    p = store.get_portfolio()
    if not p:
        return STARTING_CAPITAL, []
    return p["cash"], p["positions"]


def get_total_value(cash, positions):
    pos_value = sum(p["shares"] * p["entry_price"] for p in positions)
    return cash + pos_value


# --- Header ---
st.title("AI Paper Trader")

cash, positions = get_portfolio_summary()
total = get_total_value(cash, positions)
pnl = total - STARTING_CAPITAL
pnl_pct = (pnl / STARTING_CAPITAL) * 100

col1, col2, col3, col4 = st.columns(4)
col1.metric("Portfolio Value", f"${total:,.2f}")
col2.metric("Cash", f"${cash:,.2f}")
col3.metric("Open Positions", len(positions))
col4.metric("Total P&L", f"${pnl:,.2f}", f"{pnl_pct:+.2f}%")

st.divider()

# --- Positions ---
st.subheader("Open Positions")
if positions:
    pos_df = pd.DataFrame(positions)
    pos_df["unrealized_pnl"] = (pos_df.get("current_price", pos_df["entry_price"]) - pos_df["entry_price"]) * pos_df["shares"]
    st.dataframe(pos_df, use_container_width=True)
else:
    st.info("No open positions.")

# --- Thesis ---
st.subheader("Active Thesis (Claude)")
all_thesis = store.get_all_thesis()
if all_thesis:
    for t in all_thesis:
        color = {"bullish": "🟢", "bearish": "🔴", "neutral": "⚪"}.get(t["stance"], "⚪")
        conf_pct = int(t["confidence"] * 100)
        with st.expander(f"{color} {t['asset']} — {t['stance'].upper()} ({conf_pct}% confidence)"):
            st.write(f"**Reasoning:** {t.get('reasoning', 'N/A')}")
            st.write(f"**Horizon:** {t.get('horizon', 'N/A')}")
            st.write(f"**Updated:** {t.get('updated_at', 'N/A')}")
else:
    st.info("No thesis yet. Run the weekly LLM job first.")

st.divider()

# --- Trade History ---
st.subheader("Trade History")
trades = store.get_trades()
if trades:
    st.dataframe(pd.DataFrame(trades), use_container_width=True)
else:
    st.info("No trades yet.")

# --- Equity Curve ---
st.subheader("Portfolio Equity Curve")
snaps = store.get_snapshots()
if len(snaps) > 1:
    snap_df = pd.DataFrame(snaps)
    snap_df["recorded_at"] = pd.to_datetime(snap_df["recorded_at"])
    st.line_chart(snap_df.set_index("recorded_at")["value"])
else:
    st.info("Not enough snapshots for chart yet.")

st.divider()

# --- Backtest ---
st.subheader("Backtest")

col_start, col_end, col_cap, col_run = st.columns(4)
bt_start = col_start.date_input("Start date", value=date(2024, 1, 1))
bt_end = col_end.date_input("End date", value=date(2024, 6, 30))
bt_cap = col_cap.number_input("Starting capital ($)", value=100_000, step=10_000)

if "bt_result" not in st.session_state:
    st.session_state["bt_result"] = None
if "bt_running" not in st.session_state:
    st.session_state["bt_running"] = False
if "bt_progress" not in st.session_state:
    st.session_state["bt_progress"] = ""


def _run_backtest_thread(assets, start, end, capital):
    from trader.backtest.runner import run_backtest
    result = run_backtest(assets=assets, start_date=start, end_date=end,
                          starting_capital=capital, run_id=f"dash_{start}_{end}")
    st.session_state["bt_result"] = result
    st.session_state["bt_running"] = False


if col_run.button("Run Backtest", disabled=st.session_state["bt_running"]):
    st.session_state["bt_running"] = True
    st.session_state["bt_result"] = None
    thread = threading.Thread(
        target=_run_backtest_thread,
        args=(ASSETS, bt_start, bt_end, float(bt_cap)),
        daemon=True,
    )
    thread.start()

if st.session_state["bt_running"]:
    st.info("Backtest running in background… (refresh page to check results)")

result = st.session_state.get("bt_result")
if result:
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric("Total Return", f"{result['total_return_pct']:.2f}%")
    m_col2.metric("Sharpe Ratio", f"{result['sharpe_ratio']:.2f}")
    m_col3.metric("Max Drawdown", f"{result['max_drawdown_pct']:.2f}%")
    m_col4.metric("Win Rate", f"{result['win_rate']:.1f}%")

    if result.get("equity_curve"):
        st.line_chart(result["equity_curve"])

    if result.get("trades"):
        st.dataframe(pd.DataFrame(result["trades"]), use_container_width=True)
```

- [ ] **Step 2: Run dashboard**

```bash
/Users/ishaangubbala/OpenBB/conda/envs/openbb/bin/streamlit run /Users/ishaangubbala/OpenBB/trader/dashboard.py --server.port 8502
```
Expected: Browser opens at `http://localhost:8502` showing the trader dashboard.

- [ ] **Step 3: Run full test suite**

```bash
/Users/ishaangubbala/OpenBB/conda/envs/openbb/bin/pytest trader/tests/ -v --ignore=trader/tests/test_backtest.py
```
Expected: All tests PASS.

- [ ] **Step 4: Commit**

```bash
git add trader/dashboard.py
git commit -m "feat: add Streamlit dashboard with portfolio, thesis, backtest UI"
```

---

## Task 14: End-to-End Smoke Test

- [ ] **Step 1: Initialize DB and run first thesis + model training**

```bash
/Users/ishaangubbala/OpenBB/conda/envs/openbb/bin/python -c "
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
from trader.state.store import Store
store = Store()
store.initialize()
print('DB initialized')

from trader.scheduler import run_weekly_retrain, run_weekly_thesis
run_weekly_retrain()
run_weekly_thesis()
print('Done')
"
```
Expected: models trained for all assets, thesis saved for all assets.

- [ ] **Step 2: Run one daily signal cycle**

```bash
/Users/ishaangubbala/OpenBB/conda/envs/openbb/bin/python -c "
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
from trader.scheduler import run_daily_signals
run_daily_signals()
"
```
Expected: signals logged per asset, portfolio value printed.

- [ ] **Step 3: Launch dashboard and verify**

```bash
/Users/ishaangubbala/OpenBB/conda/envs/openbb/bin/streamlit run /Users/ishaangubbala/OpenBB/trader/dashboard.py --server.port 8502
```
Open `http://localhost:8502` — verify portfolio, thesis, and trade history are visible.

- [ ] **Step 4: Final commit**

```bash
cd /Users/ishaangubbala/OpenBB
git add -A
git commit -m "feat: complete AI paper trader — all components integrated"
```

---

## Running the System

**Start live paper trading (runs on schedule):**
```bash
source /Users/ishaangubbala/OpenBB/conda/bin/activate openbb
python /Users/ishaangubbala/OpenBB/trader/scheduler.py
```

**Start dashboard (separate terminal):**
```bash
source /Users/ishaangubbala/OpenBB/conda/bin/activate openbb
streamlit run /Users/ishaangubbala/OpenBB/trader/dashboard.py --server.port 8502
```
