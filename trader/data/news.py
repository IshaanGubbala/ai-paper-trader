"""Fetch recent news headlines via OpenBB SDK."""


def get_headlines(symbol: str, limit: int = 10) -> list[str]:
    """Return list of recent headline strings for a symbol."""
    from openbb import obb
    try:
        result = obb.news.company(symbols=symbol, limit=limit, provider="biztoc")
        return [item.title for item in result.results if item.title]
    except Exception:
        pass
    try:
        result = obb.news.world(limit=limit, provider="biztoc")
        return [item.title for item in result.results if item.title]
    except Exception:
        return []
