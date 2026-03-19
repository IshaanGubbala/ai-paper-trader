"""Tests for PaperEngine — uses a mock Store so no DB is required."""
from unittest.mock import MagicMock, call

import pytest

from trader.brain.gate import GateResult
from trader.execution.paper import PaperEngine
from trader.execution.portfolio import Portfolio


@pytest.fixture
def mock_store():
    return MagicMock()


@pytest.fixture
def engine(mock_store):
    port = Portfolio(100_000.0)
    return PaperEngine(portfolio=port, store=mock_store)


# ── process_signal ─────────────────────────────────────────────────────────

def test_enter_opens_position(engine, mock_store):
    engine.process_signal("AAPL", GateResult.ENTER, 150.0, confidence=0.8)
    assert "AAPL" in engine.portfolio.positions
    mock_store.log_trade.assert_called_once()
    args = mock_store.log_trade.call_args[0]
    assert args[0] == "AAPL"
    assert args[1] == "buy"


def test_enter_does_not_duplicate(engine, mock_store):
    engine.process_signal("AAPL", GateResult.ENTER, 150.0, confidence=0.8)
    engine.process_signal("AAPL", GateResult.ENTER, 155.0, confidence=0.8)
    assert len(engine.portfolio.positions) == 1
    assert mock_store.log_trade.call_count == 1  # only first buy logged


def test_close_removes_position(engine, mock_store):
    engine.process_signal("AAPL", GateResult.ENTER, 150.0, confidence=0.8)
    engine.process_signal("AAPL", GateResult.CLOSE, 160.0, confidence=0.8)
    assert "AAPL" not in engine.portfolio.positions
    assert mock_store.log_trade.call_count == 2  # buy + sell


def test_hold_does_nothing(engine, mock_store):
    engine.process_signal("AAPL", GateResult.HOLD, 150.0, confidence=0.8)
    assert len(engine.portfolio.positions) == 0
    mock_store.log_trade.assert_not_called()


def test_close_without_position_is_safe(engine, mock_store):
    """Closing a position we don't hold should not raise or call log_trade."""
    engine.process_signal("AAPL", GateResult.CLOSE, 150.0, confidence=0.8)
    mock_store.log_trade.assert_not_called()


# ── check_stops ────────────────────────────────────────────────────────────

def test_stop_loss_triggers(engine, mock_store):
    engine.process_signal("AAPL", GateResult.ENTER, 100.0, confidence=0.8)
    # Stop loss is at 100 * (1 - 0.05) = 95.0
    closed = engine.check_stops({"AAPL": 94.0})
    assert "AAPL" in closed
    assert "AAPL" not in engine.portfolio.positions


def test_take_profit_triggers(engine, mock_store):
    engine.process_signal("AAPL", GateResult.ENTER, 100.0, confidence=0.8)
    # Take-profit at 100 * (1 + 0.05 + 0.8 * 0.10) = 100 * 1.13 = 113.0
    closed = engine.check_stops({"AAPL": 115.0})
    assert "AAPL" in closed


def test_no_stop_in_normal_range(engine, mock_store):
    engine.process_signal("AAPL", GateResult.ENTER, 100.0, confidence=0.8)
    closed = engine.check_stops({"AAPL": 102.0})
    assert closed == []
    assert "AAPL" in engine.portfolio.positions


# ── persist ────────────────────────────────────────────────────────────────

def test_persist_passes_prices_to_snapshot(engine, mock_store):
    engine.process_signal("AAPL", GateResult.ENTER, 100.0, confidence=0.8)
    engine.persist(current_prices={"AAPL": 110.0})
    # save_snapshot should be called with a value > starting cash for this test
    mock_store.save_snapshot.assert_called_once()
    snapshot_value = mock_store.save_snapshot.call_args[0][0]
    # Portfolio: cash reduced by ~10k, position worth 110 * (n shares)
    # Total should be close to 100k (no big PnL yet at entry price)
    assert snapshot_value > 0
