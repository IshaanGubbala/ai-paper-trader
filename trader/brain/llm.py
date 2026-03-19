"""Claude thesis client — uses Anthropic SDK directly."""
import json
import logging
import re
from pathlib import Path

import anthropic

from trader.config import ANTHROPIC_MODEL, BACKTEST_CACHE_DIR

logger = logging.getLogger(__name__)

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    return _client


THESIS_PROMPT = """You are a financial analyst. Given the following data for {asset}, output a JSON trading thesis.

Price (1mo return): {price_return:.1f}%
News headlines: {headlines}
Macro: {macro_summary}
Fundamentals: {fundamentals}

Respond ONLY with valid JSON matching exactly this schema (no other text, no markdown):
{{"asset": "{asset}", "stance": "bullish" or "bearish" or "neutral", "confidence": <float 0.0-1.0>, "reasoning": "<string>", "horizon": "<string>"}}"""


def _parse_thesis(raw: str, asset: str) -> dict:
    """Extract JSON thesis from LLM response. Returns neutral fallback on failure."""
    fallback = {
        "asset": asset,
        "stance": "neutral",
        "confidence": 0.0,
        "reasoning": f"Parse failed. Raw: {raw[:200]}",
        "horizon": "unknown",
    }
    # Direct parse
    try:
        obj = json.loads(raw.strip())
        if isinstance(obj, dict) and "stance" in obj:
            obj["asset"] = asset
            return obj
    except json.JSONDecodeError:
        pass
    # Extract from prose
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
    Generate a trading thesis for an asset via Claude.
    cache_key: if set, cache response to disk (for backtest reproducibility).
    Returns thesis dict: asset, stance, confidence, reasoning, horizon.
    """
    # Check cache
    if cache_key:
        cache_file = BACKTEST_CACHE_DIR / f"{asset}_{cache_key}.json"
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        if cache_file.exists():
            logger.debug(f"[{asset}] Cache hit for key {cache_key}")
            return json.loads(cache_file.read_text())

    fund_str = str(fundamentals) if asset_type == "equity" and fundamentals else "N/A"
    headlines_str = "; ".join(headlines[:5]) if headlines else "No recent headlines"

    prompt = THESIS_PROMPT.format(
        asset=asset,
        price_return=price_return,
        headlines=headlines_str,
        macro_summary=macro_summary,
        fundamentals=fund_str,
    )

    thesis = None
    try:
        client = _get_client()
        response = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text
        thesis = _parse_thesis(raw, asset)

        # Retry once if parse failed
        if thesis["confidence"] == 0.0 and thesis["stance"] == "neutral" and "Parse failed" in thesis["reasoning"]:
            correction = f"Your previous response could not be parsed as JSON. {prompt}"
            response2 = client.messages.create(
                model=ANTHROPIC_MODEL,
                max_tokens=512,
                messages=[{"role": "user", "content": correction}],
            )
            thesis = _parse_thesis(response2.content[0].text, asset)

    except anthropic.APIError as e:
        logger.error(f"[{asset}] Anthropic API error: {e}")
        thesis = {"asset": asset, "stance": "neutral", "confidence": 0.0,
                  "reasoning": f"API error: {e}", "horizon": "unknown"}
    except Exception as e:
        logger.error(f"[{asset}] Unexpected error in get_thesis: {e}")
        thesis = {"asset": asset, "stance": "neutral", "confidence": 0.0,
                  "reasoning": f"Error: {e}", "horizon": "unknown"}

    # Write cache
    if cache_key and thesis:
        cache_file.write_text(json.dumps(thesis))

    return thesis
