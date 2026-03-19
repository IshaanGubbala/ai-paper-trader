"""Shared feature engineering — used identically by training and inference."""
import pandas as pd
import numpy as np

MIN_BARS = 50  # MACD(26) + 20d vol + safety buffer

FEATURE_COLS = [
    "rsi", "macd", "macd_signal", "bb_pct", "volume_ratio",
    "ret_5d", "ret_10d", "ret_20d", "atr", "realized_vol",
]


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all ML features from OHLCV DataFrame.
    Requires columns: date, open, high, low, close, volume.
    Returns DataFrame with FEATURE_COLS columns (same index as input).
    Raises ValueError if fewer than MIN_BARS rows.
    """
    if len(df) < MIN_BARS:
        raise ValueError(f"Insufficient history: need {MIN_BARS} bars, got {len(df)}")

    df = df.copy().sort_values("date").reset_index(drop=True)
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)

    out = pd.DataFrame(index=df.index)

    # RSI(14)
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    out["rsi"] = 100 - (100 / (1 + rs))

    # MACD(12,26,9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    out["macd"] = ema12 - ema26
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()

    # Bollinger Band %B(20): (price - lower) / (upper - lower)
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    band_width = (4 * bb_std).replace(0, np.nan)
    out["bb_pct"] = (close - (bb_mid - 2 * bb_std)) / band_width

    # Volume ratio (day / 20d avg)
    out["volume_ratio"] = volume / volume.rolling(20).mean().replace(0, np.nan)

    # Momentum returns
    out["ret_5d"] = close.pct_change(5)
    out["ret_10d"] = close.pct_change(10)
    out["ret_20d"] = close.pct_change(20)

    # ATR(14)
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    out["atr"] = tr.rolling(14).mean()

    # 20-day realized volatility (annualized)
    log_ret = np.log(close / close.shift())
    out["realized_vol"] = log_ret.rolling(20).std() * np.sqrt(252)

    return out[FEATURE_COLS]
