"""Fetch macro indicators via OpenBB SDK."""


def get_macro_summary() -> str:
    """Return a short human-readable macro summary string."""
    from openbb import obb
    lines = []

    try:
        r = obb.economy.unemployment(provider="fred", country="united_states")
        if r.results:
            val = getattr(r.results[-1], 'value', None)
            if val is not None:
                lines.append(f"US Unemployment: {val}%")
    except Exception:
        pass

    try:
        r = obb.economy.cpi(country="united_states", provider="fred", frequency="monthly")
        if r.results:
            val = getattr(r.results[-1], 'value', None)
            if val is not None:
                lines.append(f"US CPI YoY: {val}")
    except Exception:
        pass

    # Fed Funds effective rate via FRED (DFF = daily fed funds rate, annualized)
    try:
        from datetime import date, timedelta
        r = obb.economy.fred_series(
            symbol="DFF",
            start_date=(date.today() - timedelta(days=30)).isoformat(),
        )
        vals = [v.value for v in r.results if v.value is not None]
        if vals:
            rate = round(vals[-1], 2)
            lines.append(f"Fed Funds Rate: {rate}%")
    except Exception:
        pass

    return "; ".join(lines) if lines else "Macro data unavailable"
