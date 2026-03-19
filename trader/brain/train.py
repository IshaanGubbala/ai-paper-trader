"""XGBoost training pipeline — one model per asset, atomic joblib write."""
import logging
import os
from datetime import date, timedelta

import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from trader.brain.features import compute_features, MIN_BARS
from trader.config import MODELS_DIR, TRAINING_YEARS, LABEL_THRESHOLD, LABEL_HORIZON

logger = logging.getLogger(__name__)

LABEL_MAP = {"buy": 0, "hold": 1, "sell": 2}
LABEL_NAMES = {0: "buy", 1: "hold", 2: "sell"}


def _make_labels(close: pd.Series) -> pd.Series:
    fwd_return = close.shift(-LABEL_HORIZON) / close - 1
    labels = pd.Series("hold", index=close.index)
    labels[fwd_return > LABEL_THRESHOLD] = "buy"
    labels[fwd_return < -LABEL_THRESHOLD] = "sell"
    return labels


def train_model(symbol: str, asset_type: str, end_date: date | None = None) -> bool:
    """
    Train XGBoost for a symbol. Saves model + scaler atomically to MODELS_DIR.
    end_date: only use data up to this date (backtest isolation).
    Returns True on success.
    """
    from trader.data.price import get_ohlcv

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    days = TRAINING_YEARS * 365 + LABEL_HORIZON + MIN_BARS + 10

    try:
        df = get_ohlcv(symbol, asset_type, days=days)
    except Exception as e:
        logger.warning(f"[{symbol}] Data fetch failed: {e}")
        return False

    if end_date:
        df = df[pd.to_datetime(df["date"]).dt.date <= end_date]

    if len(df) < MIN_BARS + LABEL_HORIZON + 10:
        logger.warning(f"[{symbol}] Insufficient data: {len(df)} rows")
        return False

    try:
        features = compute_features(df)
    except ValueError as e:
        logger.warning(f"[{symbol}] Feature error: {e}")
        return False

    labels = _make_labels(df["close"].reset_index(drop=True))

    valid = features.notna().all(axis=1) & labels.notna()
    valid.iloc[-LABEL_HORIZON:] = False  # no forward label for last rows
    X = features[valid]
    y = labels[valid].map(LABEL_MAP)

    if len(X) < 50:
        logger.warning(f"[{symbol}] Too few samples: {len(X)}")
        return False

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Note: use_label_encoder removed in XGBoost 2.x+
    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=42,
        verbosity=0,
    )
    model.fit(X_scaled, y)

    # Atomic write: tmp → rename (prevents partial-read by inference job)
    model_path = MODELS_DIR / f"{symbol}_xgb.joblib"
    scaler_path = MODELS_DIR / f"{symbol}_scaler.joblib"
    tmp_model = model_path.with_suffix(".joblib.tmp")
    tmp_scaler = scaler_path.with_suffix(".joblib.tmp")

    joblib.dump(model, tmp_model)
    joblib.dump(scaler, tmp_scaler)
    os.replace(tmp_model, model_path)
    os.replace(tmp_scaler, scaler_path)

    logger.info(f"[{symbol}] Model trained on {len(X)} samples → {model_path.name}")
    return True
