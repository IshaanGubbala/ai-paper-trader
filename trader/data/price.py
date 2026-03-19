"""Fetch OHLCV history via yfinance (direct) for reliability."""
import pandas as pd
from datetime import date, timedelta


_CRYPTO_SUFFIX = "-USD"   # yfinance: BTC → BTC-USD
_FOREX_SUFFIX  = "=X"    # yfinance: EURUSD → EURUSD=X


def _yf_symbol(symbol: str, asset_type: str) -> str:
    """Map internal symbol to the format yfinance expects."""
    if asset_type == "crypto" and not symbol.endswith(_CRYPTO_SUFFIX):
        return symbol + _CRYPTO_SUFFIX
    if asset_type == "forex" and not symbol.endswith("=X"):
        return symbol + _FOREX_SUFFIX
    return symbol


def get_ohlcv(symbol: str, asset_type: str, days: int = 365 * 2) -> pd.DataFrame:
    """Return daily OHLCV DataFrame sorted by date."""
    import yfinance as yf
    start = (date.today() - timedelta(days=days)).isoformat()
    yf_sym = _yf_symbol(symbol, asset_type)
    try:
        df = yf.download(yf_sym, start=start, progress=False, auto_adjust=True)
    except Exception as e:
        raise RuntimeError(f"Failed to fetch OHLCV for {symbol}: {e}") from e

    if df.empty:
        raise RuntimeError(f"Failed to fetch OHLCV for {symbol}: empty result")

    df = df.reset_index()
    # yfinance returns MultiIndex columns when downloading single ticker — flatten
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() if c[1] == "" or c[1] == yf_sym else c[0].lower()
                      for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]

    df = df.rename(columns={"datetime": "date"})
    needed = [c for c in ["date", "open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[needed].sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return df


def get_one_month_return(symbol: str, asset_type: str) -> float:
    """Return 1-month price return as a percentage."""
    df = get_ohlcv(symbol, asset_type, days=45)
    if len(df) < 20:
        return 0.0
    return float((df["close"].iloc[-1] / df["close"].iloc[-21] - 1) * 100)
