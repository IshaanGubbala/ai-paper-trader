"""XGBoost inference — load model, predict signal for latest bar."""
import logging

import joblib
import numpy as np
import pandas as pd

from trader.brain.features import compute_features, MIN_BARS
from trader.brain.train import LABEL_NAMES
from trader.config import MODELS_DIR

logger = logging.getLogger(__name__)

_model_cache: dict[str, tuple] = {}  # symbol -> (model, scaler)


def _load_model(symbol: str):
    model_path = MODELS_DIR / f"{symbol}_xgb.joblib"
    scaler_path = MODELS_DIR / f"{symbol}_scaler.joblib"
    if not model_path.exists() or not scaler_path.exists():
        return None
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    _model_cache[symbol] = (model, scaler)
    return model, scaler


def reload_model(symbol: str):
    """Force reload from disk after retraining."""
    _model_cache.pop(symbol, None)
    _load_model(symbol)


def predict_signal(symbol: str, ohlcv: pd.DataFrame) -> tuple[str, float]:
    """
    Predict trading signal for the latest bar.
    Returns (signal, probability). Falls back to ('hold', 0.0) on any failure.
    """
    pair = _model_cache.get(symbol) or _load_model(symbol)
    if pair is None:
        logger.warning(f"[{symbol}] No model found — returning hold")
        return "hold", 0.0

    model, scaler = pair

    try:
        features = compute_features(ohlcv)
    except ValueError as e:
        logger.warning(f"[{symbol}] Feature error: {e}")
        return "hold", 0.0

    last = features.iloc[[-1]]
    if last.isna().any().any():
        logger.warning(f"[{symbol}] NaN in features — returning hold")
        return "hold", 0.0

    X = scaler.transform(last)
    probs = model.predict_proba(X)[0]
    pred_idx = int(np.argmax(probs))
    return LABEL_NAMES[pred_idx], float(probs[pred_idx])
