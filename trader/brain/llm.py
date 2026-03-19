"""OpenAI thesis client — uses openai SDK directly."""
import json
import logging
import re
from pathlib import Path

from openai import OpenAI

from trader.config import OPENAI_MODEL, BACKTEST_CACHE_DIR

logger = logging.getLogger(__name__)

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()  # reads OPENAI_API_KEY from env
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
    Generate a trading thesis for an asset via OpenAI.
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
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.choices[0].message.content or ""
        thesis = _parse_thesis(raw, asset)

        # Retry once if parse failed
        if thesis["confidence"] == 0.0 and thesis["stance"] == "neutral" and "Parse failed" in thesis["reasoning"]:
            correction = f"Your previous response could not be parsed as JSON. {prompt}"
            response2 = client.chat.completions.create(
                model=OPENAI_MODEL,
                max_tokens=512,
                messages=[{"role": "user", "content": correction}],
            )
            thesis = _parse_thesis(response2.choices[0].message.content or "", asset)

    except Exception as e:
        logger.error(f"[{asset}] OpenAI API error: {e}")
        thesis = {"asset": asset, "stance": "neutral", "confidence": 0.0,
                  "reasoning": f"API error: {e}", "horizon": "unknown"}

    # Write cache
    if cache_key and thesis:
        cache_file.write_text(json.dumps(thesis))

    return thesis
