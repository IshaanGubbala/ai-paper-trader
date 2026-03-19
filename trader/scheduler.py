"""APScheduler jobs — weekly LLM thesis + daily ML signals, paper trades."""
import logging
from datetime import datetime, timezone

import pytz
from apscheduler.schedulers.blocking import BlockingScheduler

from trader.config import ASSETS
from trader.state.store import Store
from trader.execution.portfolio import Portfolio
from trader.execution.paper import PaperEngine
from trader.brain.gate import evaluate_gate

logger = logging.getLogger(__name__)
ET = pytz.timezone("US/Eastern")


def _is_market_open(asset_type: str, dt: datetime) -> bool:
    """Return False for forex on weekends, True otherwise (holidays ignored)."""
    if asset_type == "forex":
        return dt.weekday() < 5  # Mon–Fri only
    return True  # equities/crypto — APScheduler cron handles market hours


def _run_weekly_thesis(store: Store) -> None:
    """Monday 6 AM ET — regenerate LLM thesis for all assets."""
    from trader.data.price import get_ohlcv, get_one_month_return
    from trader.data.macro import get_macro_summary
    from trader.data.fundamentals import get_fundamentals
    from trader.data.news import get_headlines
    from trader.brain.llm import get_thesis

    logger.info("[Scheduler] Weekly thesis job starting")
    macro = get_macro_summary()

    for asset in ASSETS:
        sym = asset["symbol"]
        try:
            price_return = get_one_month_return(sym, asset["type"])
            headlines = get_headlines(sym)
            fund = get_fundamentals(sym) if asset["type"] == "equity" else {}
            thesis = get_thesis(sym, asset["type"], price_return, headlines, macro, fund)
            store.save_thesis(
                asset=sym,
                stance=thesis["stance"],
                confidence=float(thesis.get("confidence", 0.0)),
                reasoning=thesis.get("reasoning", ""),
                horizon=thesis.get("horizon", ""),
            )
            logger.info(f"[{sym}] Thesis saved: {thesis['stance']} ({thesis['confidence']:.2f})")
        except Exception as e:
            logger.error(f"[{sym}] Weekly thesis failed: {e}")


def _run_weekly_retrain(store: Store) -> None:
    """Monday 7 AM ET — retrain ML models on fresh data."""
    from trader.brain.train import train_model

    logger.info("[Scheduler] Weekly ML retrain job starting")
    for asset in ASSETS:
        sym = asset["symbol"]
        try:
            ok = train_model(sym, asset["type"])
            logger.info(f"[{sym}] Retrain {'succeeded' if ok else 'skipped (no data)'}")
        except Exception as e:
            logger.error(f"[{sym}] Retrain failed: {e}")


def _run_daily_signals(store: Store, engine: PaperEngine) -> None:
    """Daily 4:30 PM ET — generate signals, paper-trade, persist state."""
    from trader.data.price import get_ohlcv
    from trader.brain.ml import predict_signal

    now_et = datetime.now(ET)
    logger.info(f"[Scheduler] Daily signal job: {now_et.strftime('%Y-%m-%d %H:%M ET')}")

    current_prices: dict[str, float] = {}

    for asset in ASSETS:
        sym = asset["symbol"]

        if not _is_market_open(asset["type"], now_et):
            logger.debug(f"[{sym}] Skipping — market closed for {asset['type']}")
            continue

        try:
            df = get_ohlcv(sym, asset["type"], days=90)
            if df.empty:
                continue

            price = float(df["close"].iloc[-1])
            current_prices[sym] = price

            thesis_row = store.get_thesis(sym)
            thesis = (
                {"stance": thesis_row["stance"], "confidence": thesis_row["confidence"]}
                if thesis_row else {"stance": "neutral", "confidence": 0.0}
            )

            signal, prob = predict_signal(sym, df)
            has_position = sym in engine.portfolio.positions
            action = evaluate_gate(thesis, signal, prob, has_position)

            engine.process_signal(sym, action, price,
                                  confidence=float(thesis.get("confidence", 0.0)),
                                  reason=f"gate:{thesis['stance']}")

        except Exception as e:
            logger.error(f"[{sym}] Daily signal failed: {e}")

    # Stop-loss / take-profit sweep
    engine.check_stops(current_prices)

    # Persist
    engine.persist(current_prices)
    logger.info("[Scheduler] Daily signal job complete")


def build_portfolio_and_engine(store: Store) -> PaperEngine:
    """Restore portfolio from DB (or start fresh) and return a PaperEngine."""
    from trader.config import STARTING_CAPITAL
    saved = store.get_portfolio()
    if saved:
        portfolio = Portfolio.from_dict(saved, STARTING_CAPITAL)
        logger.info(f"[Scheduler] Restored portfolio — cash: ${portfolio.cash:,.2f}")
    else:
        portfolio = Portfolio(STARTING_CAPITAL)
        logger.info(f"[Scheduler] Starting fresh portfolio — ${STARTING_CAPITAL:,.0f}")
    return PaperEngine(portfolio=portfolio, store=store)


def run() -> None:
    """Entry-point: start the scheduler (blocking)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    store = Store()
    store.initialize()
    engine = build_portfolio_and_engine(store)

    scheduler = BlockingScheduler(timezone=ET)

    # Monday 6 AM ET — LLM thesis
    scheduler.add_job(
        _run_weekly_thesis,
        trigger="cron",
        day_of_week="mon",
        hour=6,
        minute=0,
        kwargs={"store": store},
        id="weekly_thesis",
        replace_existing=True,
    )

    # Monday 7 AM ET — ML retrain
    scheduler.add_job(
        _run_weekly_retrain,
        trigger="cron",
        day_of_week="mon",
        hour=7,
        minute=0,
        kwargs={"store": store},
        id="weekly_retrain",
        replace_existing=True,
    )

    # Mon–Fri 4:30 PM ET — daily signals (equities close)
    # Crypto is 24/7 but we check once daily for simplicity
    scheduler.add_job(
        _run_daily_signals,
        trigger="cron",
        day_of_week="mon-fri",
        hour=16,
        minute=30,
        kwargs={"store": store, "engine": engine},
        id="daily_signals",
        replace_existing=True,
    )

    logger.info("[Scheduler] Starting — jobs registered. Press Ctrl+C to stop.")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("[Scheduler] Stopped.")


if __name__ == "__main__":
    run()
