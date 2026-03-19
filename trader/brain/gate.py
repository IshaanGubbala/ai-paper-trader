"""Gate logic — combines LLM thesis and ML signal into a trade action."""
from enum import Enum
from trader.config import MIN_LLM_CONFIDENCE


class GateResult(str, Enum):
    ENTER = "enter"   # Open new long position
    CLOSE = "close"   # Close existing long position
    HOLD  = "hold"    # Do nothing


def evaluate_gate(
    thesis: dict,
    ml_signal: str,
    ml_prob: float,
    has_position: bool,
) -> GateResult:
    """
    Return the action to take based on LLM thesis + ML signal.

    Gate truth table (all other combinations → HOLD):
    - bullish (conf≥0.6) + ML buy  + no position → ENTER
    - bearish (conf≥0.6) + ML sell + has position → CLOSE
    - neutral OR conf<0.6              → HOLD always
    """
    stance = thesis.get("stance", "neutral")
    confidence = float(thesis.get("confidence", 0.0))

    # Low confidence treated as neutral regardless of stated stance
    if confidence < MIN_LLM_CONFIDENCE:
        return GateResult.HOLD

    if stance == "bullish" and ml_signal == "buy" and not has_position:
        return GateResult.ENTER

    if stance == "bearish" and ml_signal == "sell" and has_position:
        return GateResult.CLOSE

    return GateResult.HOLD
