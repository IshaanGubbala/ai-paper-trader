"""Tests for SQLite store — no external dependencies."""
import pytest


def test_save_and_get_thesis(store):
    store.save_thesis("AAPL", "bullish", 0.82, "Strong earnings", "1-2 weeks")
    row = store.get_thesis("AAPL")
    assert row["stance"] == "bullish"
    assert abs(row["confidence"] - 0.82) < 1e-6
    assert row["asset"] == "AAPL"


def test_get_thesis_missing(store):
    assert store.get_thesis("NONEXISTENT") is None


def test_save_thesis_upsert(store):
    store.save_thesis("AAPL", "bullish", 0.8, "r1", "1w")
    store.save_thesis("AAPL", "bearish", 0.9, "r2", "2w")
    rows = store.get_all_thesis()
    assert len(rows) == 1
    assert rows[0]["stance"] == "bearish"


def test_save_and_get_portfolio(store):
    positions = [{"symbol": "AAPL", "shares": 10, "entry_price": 150.0}]
    store.save_portfolio(85_000.0, positions)
    p = store.get_portfolio()
    assert abs(p["cash"] - 85_000.0) < 1e-6
    assert p["positions"][0]["symbol"] == "AAPL"


def test_get_portfolio_empty(store):
    assert store.get_portfolio() is None


def test_log_trade_and_get(store):
    store.log_trade("NVDA", "buy", 5.0, 420.0, "gate:bullish")
    trades = store.get_trades()
    assert len(trades) == 1
    t = trades[0]
    assert t["symbol"] == "NVDA"
    assert t["action"] == "buy"
    assert abs(t["shares"] - 5.0) < 1e-6


def test_save_and_get_snapshots(store):
    store.save_snapshot(100_000.0)
    store.save_snapshot(102_500.0)
    snaps = store.get_snapshots()
    assert len(snaps) == 2
    assert snaps[0]["value"] < snaps[1]["value"]
