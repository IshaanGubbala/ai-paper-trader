"""Tests for feature computation — no external API calls."""
import numpy as np
import pandas as pd
import pytest

from trader.brain.features import compute_features, FEATURE_COLS, MIN_BARS


def test_compute_features_returns_all_columns(ohlcv_df):
    feat = compute_features(ohlcv_df)
    assert set(FEATURE_COLS).issubset(feat.columns)


def test_compute_features_no_nan_at_tail(ohlcv_df):
    feat = compute_features(ohlcv_df)
    # The last row (used for inference) must have no NaN in feature columns
    last = feat[FEATURE_COLS].iloc[-1]
    assert last.notna().all(), f"NaN found in last row: {last[last.isna()].index.tolist()}"


def test_compute_features_raises_on_short_df(ohlcv_df_small):
    with pytest.raises(ValueError, match="Insufficient"):
        compute_features(ohlcv_df_small)


def test_rsi_bounded(ohlcv_df):
    feat = compute_features(ohlcv_df)
    rsi = feat["rsi"].dropna()
    assert (rsi >= 0).all() and (rsi <= 100).all()


def test_bb_pct_finite(ohlcv_df):
    feat = compute_features(ohlcv_df)
    bb = feat["bb_pct"].dropna()
    assert np.isfinite(bb).all()


def test_volume_ratio_positive(ohlcv_df):
    feat = compute_features(ohlcv_df)
    vr = feat["volume_ratio"].dropna()
    assert (vr > 0).all()
