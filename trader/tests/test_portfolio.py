"""Tests for Portfolio — no external dependencies."""
import pytest

from trader.execution.portfolio import Portfolio
from trader.config import MAX_POSITIONS, MAX_ALLOCATION, STOP_LOSS_PCT


def test_initial_state(portfolio):
    assert portfolio.cash == 100_000.0
    assert len(portfolio.positions) == 0


def test_open_position_deducts_cash(portfolio):
    price = 100.0
    shares = portfolio.compute_position_size("AAPL", price)
    portfolio.open_position("AAPL", shares, price, price * 1.10, price * 0.95)
    cost = shares * price
    assert abs(portfolio.cash - (100_000.0 - cost)) < 1e-6


def test_open_position_returns_false_for_duplicate(portfolio):
    shares = portfolio.compute_position_size("AAPL", 100.0)
    portfolio.open_position("AAPL", shares, 100.0, 110.0, 95.0)
    result = portfolio.open_position("AAPL", shares, 100.0, 110.0, 95.0)
    assert result is False
    assert len(portfolio.positions) == 1


def test_open_position_respects_max_positions(portfolio):
    for i in range(MAX_POSITIONS):
        sym = f"SYM{i}"
        price = 10.0
        # Compute shares based on current portfolio value each time
        shares = portfolio.compute_position_size(sym, price)
        portfolio.open_position(sym, shares, price, price * 1.10, price * 0.95)

    # One more should be rejected
    result = portfolio.open_position("EXTRA", 1.0, 10.0, 11.0, 9.5)
    assert result is False


def test_close_position_returns_pnl(portfolio):
    portfolio.open_position("AAPL", 10.0, 100.0, 110.0, 95.0)
    pnl = portfolio.close_position("AAPL", 110.0, reason="take_profit")
    assert abs(pnl - 100.0) < 1e-6  # 10 shares × $10 gain


def test_close_position_adds_to_closed_trades(portfolio):
    portfolio.open_position("AAPL", 10.0, 100.0, 110.0, 95.0)
    portfolio.close_position("AAPL", 95.0, reason="stop_loss")
    assert len(portfolio.closed_trades) == 1
    assert portfolio.closed_trades[0]["reason"] == "stop_loss"
    assert portfolio.closed_trades[0]["pnl"] < 0


def test_close_nonexistent_position_returns_zero(portfolio):
    pnl = portfolio.close_position("GHOST", 100.0, "test")
    assert pnl == 0.0


def test_total_value_with_current_prices(portfolio):
    portfolio.open_position("AAPL", 10.0, 100.0, 110.0, 95.0)
    tv = portfolio.total_value({"AAPL": 120.0})
    # cash = 100000 - 1000 = 99000; positions = 10 * 120 = 1200
    assert abs(tv - (99_000.0 + 1_200.0)) < 1e-6


def test_total_value_uses_entry_price_when_no_current_price(portfolio):
    portfolio.open_position("AAPL", 10.0, 100.0, 110.0, 95.0)
    tv = portfolio.total_value({})  # no current price supplied
    # Should fall back to entry price
    assert abs(tv - 100_000.0) < 1e-6


def test_position_size_is_ten_percent(portfolio):
    # With a clean $100k portfolio and no positions
    shares = portfolio.compute_position_size("AAPL", 100.0)
    cost = shares * 100.0
    assert abs(cost / 100_000.0 - MAX_ALLOCATION) < 1e-6


def test_serialization_round_trip(portfolio):
    portfolio.open_position("AAPL", 10.0, 150.0, 165.0, 142.5, confidence=0.8)
    d = portfolio.to_dict()
    p2 = Portfolio.from_dict(d, starting_capital=100_000.0)
    assert abs(p2.cash - portfolio.cash) < 1e-6
    assert "AAPL" in p2.positions
    assert abs(p2.positions["AAPL"].confidence - 0.8) < 1e-6
