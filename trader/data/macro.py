"""Fetch macro indicators via OpenBB SDK."""


def get_macro_summary() -> str:
    """Return a short human-readable macro summary string."""
    from openbb import obb
    lines = []
    try:
        r = obb.economy.unemployment(provider="fred", country="united_states")
        if r.results:
            lines.append(f"US Unemployment: {getattr(r.results[-1], 'value', 'N/A')}%")
    except Exception:
        pass
    try:
        r = obb.economy.cpi(country="united_states", provider="fred", frequency="monthly")
        if r.results:
            lines.append(f"US CPI (monthly): {getattr(r.results[-1], 'value', 'N/A')}")
    except Exception:
        pass
    try:
        r = obb.economy.interest_rates(provider="oecd", country="united_states")
        if r.results:
            lines.append(f"US Interest Rate: {getattr(r.results[-1], 'value', 'N/A')}%")
    except Exception:
        pass
    return "; ".join(lines) if lines else "Macro data unavailable"
