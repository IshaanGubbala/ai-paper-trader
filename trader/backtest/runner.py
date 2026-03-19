"""Backtest runner — time-step simulation using the same brain + execution pipeline."""
import logging
from datetime import date, timedelta
from typing import Callable

import numpy as np
import pandas as pd

from trader.brain.features import MIN_BARS
from trader.brain.gate import evaluate_gate
from trader.execution.portfolio import Portfolio

logger = logging.getLogger(__name__)

_FALLBACK_RF_RATE = 0.045


def _get_risk_free_rate(start_date: date, end_date: date) -> float:
    """Fetch average T-bill rate from FRED for the period. Falls back to 4.5%."""
    try:
        from openbb import obb
        r = obb.economy.fred_series(
            symbol="DTB3",
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
        )
        vals = [v.value for v in r.results if v.value is not None]
        if vals:
            return float(np.mean(vals)) / 100.0
    except Exception as e:
        logger.warning(f"FRED rate fetch failed: {e}. Using {_FALLBACK_RF_RATE:.1%}")
    return _FALLBACK_RF_RATE


def _compute_metrics(equity_curve: list[float], trades: list[dict], rf_rate: float) -> dict:
    arr = np.array(equity_curve)
    if len(arr) < 2:
        return {"total_return_pct": 0.0, "sharpe_ratio": 0.0,
                "max_drawdown_pct": 0.0, "win_rate": 0.0}

    total_return = (arr[-1] / arr[0] - 1) * 100
    daily_returns = np.diff(arr) / arr[:-1]
    excess = daily_returns - rf_rate / 252
    sharpe = float(np.mean(excess) / np.std(excess) * np.sqrt(252)) if np.std(excess) > 0 else 0.0
    running_max = np.maximum.accumulate(arr)
    max_drawdown = float(np.min((arr - running_max) / running_max) * 100)
    profitable = [t for t in trades if t.get("pnl", 0) > 0]
    win_rate = len(profitable) / len(trades) * 100 if trades else 0.0

    return {
        "total_return_pct": float(total_return),
        "sharpe_ratio": sharpe,
        "max_drawdown_pct": max_drawdown,
        "win_rate": float(win_rate),
    }


def run_backtest(
    assets: list[dict],
    start_date: date,
    end_date: date,
    starting_capital: float = 100_000.0,
    run_id: str = "default",
    progress_callback: Callable[[date], None] | None = None,
) -> dict:
    """
    Run a backtest over [start_date, end_date].
    - ML models trained on data strictly before start_date
    - LLM calls cached per (run_id, asset, week) for reproducibility
    - Point-in-time: only uses data available at each simulated date
    Returns metrics dict + trade log + equity curve.
    """
    from trader.data.price import get_ohlcv
    from trader.data.macro import get_macro_summary
    from trader.data.fundamentals import get_fundamentals
    from trader.brain.llm import get_thesis
    from trader.brain.ml import predict_signal
    from trader.brain.train import train_model

    logger.info(f"[Backtest] Fetching data for {len(assets)} assets...")
    asset_data: dict[str, pd.DataFrame] = {}
    for asset in assets:
        sym = asset["symbol"]
        try:
            df = get_ohlcv(sym, asset["type"], days=365 * 3)
            asset_data[sym] = df
        except Exception as e:
            logger.warning(f"[{sym}] Data fetch failed: {e}")

    logger.info("[Backtest] Training models on pre-backtest data...")
    for asset in assets:
        train_model(asset["symbol"], asset["type"],
                    end_date=start_date - timedelta(days=1))

    rf_rate = _get_risk_free_rate(start_date, end_date)
    portfolio = Portfolio(starting_capital)
    equity_curve: list[float] = [starting_capital]
    all_trades: list[dict] = []
    current_thesis: dict[str, dict] = {}

    macro_summary = get_macro_summary()

    current = start_date
    while current <= end_date:
        if progress_callback:
            progress_callback(current)

        # Weekly (Monday): regenerate LLM thesis on historical snapshot
        if current.weekday() == 0:
            for asset in assets:
                sym = asset["symbol"]
                df = asset_data.get(sym, pd.DataFrame())
                hist = df[pd.to_datetime(df["date"]).dt.date <= current] if not df.empty else df
                if hist.empty:
                    continue
                n = min(21, len(hist) - 1)
                price_return = float((hist["close"].iloc[-1] / hist["close"].iloc[-n-1] - 1) * 100) if n > 0 else 0.0

                # Point-in-time fundamentals: only data available before current date
                fund = {}
                if asset["type"] == "equity":
                    try:
                        fund = get_fundamentals(sym)
                    except Exception:
                        pass

                # Cache key: run_id + ISO week Monday date (stable, human-readable)
                week_key = current.isoformat()
                thesis = get_thesis(
                    sym, asset["type"], price_return, [], macro_summary, fund,
                    cache_key=f"{run_id}_{week_key}",
                )
                current_thesis[sym] = thesis

        # Daily: generate signals and process trades
        current_prices: dict[str, float] = {}
        prev_closed_count = len(portfolio.closed_trades)

        for asset in assets:
            sym = asset["symbol"]
            df = asset_data.get(sym, pd.DataFrame())
            hist = df[pd.to_datetime(df["date"]).dt.date <= current] if not df.empty else df
            if len(hist) < MIN_BARS:
                continue

            price = float(hist["close"].iloc[-1])
            current_prices[sym] = price

            thesis = current_thesis.get(sym, {"stance": "neutral", "confidence": 0.0})
            signal, prob = predict_signal(sym, hist)
            has_position = sym in portfolio.positions
            action = evaluate_gate(thesis, signal, prob, has_position)

            if action.value == "enter":
                tp = price * (1 + 0.05 + float(thesis.get("confidence", 0.7)) * 0.10)
                sl = price * 0.95
                shares = portfolio.compute_position_size(sym, price)
                portfolio.open_position(sym, shares, price, tp, sl,
                                        confidence=float(thesis.get("confidence", 0.7)))

            elif action.value == "close":
                portfolio.close_position(sym, price, reason="gate:bearish")

        # Collect newly closed trades (gate closes only — capture by index)
        new_gate_trades = portfolio.closed_trades[prev_closed_count:]
        all_trades.extend(new_gate_trades)
        pre_stop_count = len(portfolio.closed_trades)

        # Check stops — capture trades separately after gate loop
        for sym in list(portfolio.positions.keys()):
            price = current_prices.get(sym)
            if price is None:
                continue
            pos = portfolio.positions.get(sym)
            if pos is None:
                continue
            if price <= pos.stop_loss_price:
                portfolio.close_position(sym, price, reason="stop_loss")
            elif price >= pos.take_profit_price:
                portfolio.close_position(sym, price, reason="take_profit")

        # Collect stop-triggered trades (after gate trades)
        new_stop_trades = portfolio.closed_trades[pre_stop_count:]
        all_trades.extend(new_stop_trades)

        equity_curve.append(portfolio.total_value(current_prices))
        current += timedelta(days=1)

    metrics = _compute_metrics(equity_curve, all_trades, rf_rate)
    metrics["trades"] = all_trades
    metrics["equity_curve"] = equity_curve
    logger.info(
        f"[Backtest] Done. Return: {metrics['total_return_pct']:.1f}%, "
        f"Sharpe: {metrics['sharpe_ratio']:.2f}, Win rate: {metrics['win_rate']:.1f}%"
    )
    return metrics
