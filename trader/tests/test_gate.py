"""Tests for gate logic — pure Python, no IO."""
import pytest

from trader.brain.gate import GateResult, evaluate_gate


def _thesis(stance, confidence):
    return {"stance": stance, "confidence": confidence}


# ── ENTER conditions ───────────────────────────────────────────────────────

def test_enter_on_bullish_buy_no_position():
    result = evaluate_gate(_thesis("bullish", 0.8), "buy", 0.75, has_position=False)
    assert result == GateResult.ENTER


def test_no_enter_when_already_holding():
    result = evaluate_gate(_thesis("bullish", 0.8), "buy", 0.75, has_position=True)
    assert result == GateResult.HOLD


def test_no_enter_on_low_confidence():
    result = evaluate_gate(_thesis("bullish", 0.3), "buy", 0.75, has_position=False)
    assert result == GateResult.HOLD


def test_no_enter_when_ml_says_hold():
    result = evaluate_gate(_thesis("bullish", 0.8), "hold", 0.60, has_position=False)
    assert result == GateResult.HOLD


def test_no_enter_on_neutral_stance():
    result = evaluate_gate(_thesis("neutral", 0.9), "buy", 0.90, has_position=False)
    assert result == GateResult.HOLD


# ── CLOSE conditions ───────────────────────────────────────────────────────

def test_close_on_bearish_sell_with_position():
    result = evaluate_gate(_thesis("bearish", 0.75), "sell", 0.80, has_position=True)
    assert result == GateResult.CLOSE


def test_no_close_without_position():
    result = evaluate_gate(_thesis("bearish", 0.75), "sell", 0.80, has_position=False)
    assert result == GateResult.HOLD


def test_no_close_on_low_confidence():
    result = evaluate_gate(_thesis("bearish", 0.4), "sell", 0.80, has_position=True)
    assert result == GateResult.HOLD


# ── Edge cases ─────────────────────────────────────────────────────────────

def test_exactly_min_confidence_threshold():
    """Confidence exactly at MIN_LLM_CONFIDENCE (0.60) should be accepted."""
    result = evaluate_gate(_thesis("bullish", 0.60), "buy", 0.75, has_position=False)
    assert result == GateResult.ENTER


def test_just_below_min_confidence():
    result = evaluate_gate(_thesis("bullish", 0.59), "buy", 0.75, has_position=False)
    assert result == GateResult.HOLD


def test_missing_fields_default_neutral():
    """Empty thesis dict should default to neutral/0.0 confidence → HOLD."""
    result = evaluate_gate({}, "buy", 0.90, has_position=False)
    assert result == GateResult.HOLD
