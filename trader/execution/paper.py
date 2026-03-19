"""Paper trading engine — processes gate signals, enforces stop/take-profit."""
import logging

from trader.brain.gate import GateResult
from trader.config import STOP_LOSS_PCT

logger = logging.getLogger(__name__)


class PaperEngine:
    def __init__(self, portfolio, store):
        self.portfolio = portfolio
        self.store = store

    def process_signal(
        self,
        symbol: str,
        action: GateResult,
        entry_price: float,
        confidence: float,
        reason: str = "",
    ):
        """Execute paper trade based on gate decision."""
        if action == GateResult.ENTER:
            # Take-profit: 5% + (confidence × 10%) above entry
            take_profit = entry_price * (1 + 0.05 + confidence * 0.10)
            stop_loss = entry_price * (1 - STOP_LOSS_PCT)
            shares = self.portfolio.compute_position_size(symbol, entry_price)
            opened = self.portfolio.open_position(
                symbol, shares, entry_price, take_profit, stop_loss,
                confidence=confidence, reason=reason or "gate:bullish+buy",
            )
            if opened:
                self.store.log_trade(
                    symbol, "buy", shares, entry_price,
                    reason or "gate:bullish+buy",
                )

        elif action == GateResult.CLOSE:
            pos = self.portfolio.positions.get(symbol)
            if pos:
                shares = pos.shares
                pnl = self.portfolio.close_position(
                    symbol, entry_price, reason=reason or "gate:bearish+sell"
                )
                self.store.log_trade(
                    symbol, "sell", shares, entry_price,
                    reason or "gate:bearish+sell",
                )

    def check_stops(self, current_prices: dict[str, float]) -> list[str]:
        """
        Check all positions against stop-loss and take-profit at current prices.
        Returns list of closed symbols.
        Stop-loss enforcement is daily-only (checked once at 4:30 PM ET).
        Intraday gaps (including overnight crypto moves) are a known limitation.
        """
        closed = []
        for symbol in list(self.portfolio.positions.keys()):
            price = current_prices.get(symbol)
            if price is None:
                continue
            pos = self.portfolio.positions.get(symbol)
            if pos is None:
                continue

            if price <= pos.stop_loss_price:
                shares = pos.shares
                self.portfolio.close_position(symbol, price, reason="stop_loss")
                self.store.log_trade(symbol, "sell", shares, price, "stop_loss")
                closed.append(symbol)
                logger.info(f"[{symbol}] Stop-loss @ {price:.2f}")
            elif price >= pos.take_profit_price:
                shares = pos.shares
                self.portfolio.close_position(symbol, price, reason="take_profit")
                self.store.log_trade(symbol, "sell", shares, price, "take_profit")
                closed.append(symbol)
                logger.info(f"[{symbol}] Take-profit @ {price:.2f}")

        return closed

    def persist(self, current_prices: dict[str, float] | None = None):
        """Save portfolio state and snapshot to SQLite."""
        prices = current_prices or {}
        d = self.portfolio.to_dict()
        self.store.save_portfolio(d["cash"], d["positions"])
        # Snapshot uses current prices so equity curve is accurate
        self.store.save_snapshot(self.portfolio.total_value(prices))
