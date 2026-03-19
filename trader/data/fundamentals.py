"""Fetch fundamental data for equities via OpenBB SDK."""


def get_fundamentals(symbol: str) -> dict:
    """Return dict with pe_ratio, revenue_growth, debt_to_equity. Empty dict on failure."""
    from openbb import obb
    out = {}
    try:
        r = obb.equity.fundamental.ratios(symbol, provider="yfinance")
        if r.results:
            item = r.results[0]
            out["pe_ratio"] = getattr(item, "pe_ratio", None)
            out["revenue_growth"] = getattr(item, "revenue_growth", None)
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
