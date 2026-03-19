"""Shared pytest fixtures for the trader test suite."""
import os
import tempfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trader.execution.portfolio import Portfolio


# ── OHLCV helpers ──────────────────────────────────────────────────────────

def _make_ohlcv(n: int = 100, start: str = "2023-01-01") -> pd.DataFrame:
    """Synthetic OHLCV DataFrame with realistic random-walk prices."""
    rng = np.random.default_rng(42)
    closes = 100.0 * np.cumprod(1 + rng.normal(0.0005, 0.015, n))
    dates = pd.date_range(start, periods=n, freq="B")
    return pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "open": closes * rng.uniform(0.99, 1.01, n),
            "high": closes * rng.uniform(1.00, 1.02, n),
            "low": closes * rng.uniform(0.98, 1.00, n),
            "close": closes,
            "volume": rng.integers(1_000_000, 10_000_000, n).astype(float),
        }
    )


@pytest.fixture
def ohlcv_df() -> pd.DataFrame:
    return _make_ohlcv(100)


@pytest.fixture
def ohlcv_df_small() -> pd.DataFrame:
    """Fewer than MIN_BARS — triggers ValueError in compute_features."""
    return _make_ohlcv(30)


# ── Portfolio ──────────────────────────────────────────────────────────────

@pytest.fixture
def portfolio() -> Portfolio:
    return Portfolio(100_000.0)


# ── SQLite store (temp dir) ────────────────────────────────────────────────

@pytest.fixture
def tmp_db(tmp_path) -> Path:
    """Return path to a fresh temporary SQLite DB with migrations applied."""
    import trader.config as cfg
    db = tmp_path / "test_trader.db"
    # Patch DB_PATH so Store uses the temp DB
    original = cfg.DB_PATH
    cfg.DB_PATH = db
    yield db
    cfg.DB_PATH = original


@pytest.fixture
def store(tmp_db):
    """Return an initialised Store instance backed by the temp DB."""
    from trader.state.store import Store
    s = Store(db_path=tmp_db)
    s.initialize()
    return s


# ── Date helpers ───────────────────────────────────────────────────────────

@pytest.fixture
def today() -> date:
    return date(2024, 1, 15)
