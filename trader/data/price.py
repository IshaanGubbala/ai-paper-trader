"""Fetch OHLCV history via OpenBB SDK."""
import pandas as pd
from datetime import date, timedelta


def get_ohlcv(symbol: str, asset_type: str, days: int = 365 * 2) -> pd.DataFrame:
    """Return daily OHLCV DataFrame sorted by date."""
    from openbb import obb
    start = (date.today() - timedelta(days=days)).isoformat()
    try:
        if asset_type == "equity":
            result = obb.equity.price.historical(symbol, start=start, provider="yfinance")
        elif asset_type == "crypto":
            result = obb.crypto.price.historical(symbol, start=start, provider="yfinance")
        elif asset_type == "forex":
            result = obb.currency.price.historical(symbol, start=start, provider="yfinance")
        else:
            raise ValueError(f"Unknown asset type: {asset_type}")
    except Exception as e:
        raise RuntimeError(f"Failed to fetch OHLCV for {symbol}: {e}") from e

    df = result.to_df().reset_index()
    df.columns = [c.lower() for c in df.columns]
    needed = [c for c in ["date", "open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[needed].sort_values("date").reset_index(drop=True)
    return df


def get_one_month_return(symbol: str, asset_type: str) -> float:
    """Return 1-month price return as a percentage."""
    df = get_ohlcv(symbol, asset_type, days=45)
    if len(df) < 20:
        return 0.0
    return float((df["close"].iloc[-1] / df["close"].iloc[-21] - 1) * 100)
