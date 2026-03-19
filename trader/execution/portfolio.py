"""Portfolio tracker — position sizing, P&L, cash management."""
import logging
from dataclasses import dataclass, field

from trader.config import MAX_POSITIONS, MAX_ALLOCATION, STOP_LOSS_PCT

logger = logging.getLogger(__name__)


@dataclass
class Position:
    symbol: str
    shares: float
    entry_price: float
    take_profit_price: float
    stop_loss_price: float
    confidence: float = 0.7
    reason: str = ""


class Portfolio:
    def __init__(self, starting_capital: float):
        self.cash = starting_capital
        self._starting_capital = starting_capital
        self.positions: dict[str, Position] = {}
        self.closed_trades: list[dict] = []

    def compute_position_size(self, symbol: str, entry_price: float) -> float:
        """
        Returns shares to buy.
        Effective rule: position = 10% of current portfolio value (cap always binds).
        """
        portfolio_value = self.cash + sum(
            p.shares * p.entry_price for p in self.positions.values()
        )
        dollar_size = portfolio_value * MAX_ALLOCATION
        return dollar_size / entry_price if entry_price > 0 else 0.0

    def open_position(
        self,
        symbol: str,
        shares: float,
        entry_price: float,
        take_profit_price: float,
        stop_loss_price: float,
        confidence: float = 0.7,
        reason: str = "",
    ) -> bool:
        """Open a position. Returns False if rejected."""
        if symbol in self.positions:
            return False
        if len(self.positions) >= MAX_POSITIONS:
            logger.warning(f"[{symbol}] Max positions ({MAX_POSITIONS}) reached")
            return False
        cost = shares * entry_price
        if cost > self.cash:
            shares = (self.cash * 0.99) / entry_price  # use 99% of available cash
            cost = shares * entry_price
        if shares <= 0:
            return False
        self.cash -= cost
        self.positions[symbol] = Position(
            symbol=symbol,
            shares=shares,
            entry_price=entry_price,
            take_profit_price=take_profit_price,
            stop_loss_price=stop_loss_price,
            confidence=confidence,
            reason=reason,
        )
        logger.info(f"[{symbol}] Opened {shares:.4f} shares @ {entry_price:.2f}")
        return True

    def close_position(self, symbol: str, exit_price: float, reason: str = "") -> float:
        """Close a position. Returns realized P&L (0 if not held)."""
        pos = self.positions.pop(symbol, None)
        if pos is None:
            return 0.0
        proceeds = pos.shares * exit_price
        pnl = proceeds - (pos.shares * pos.entry_price)
        self.cash += proceeds
        trade = {
            "symbol": symbol,
            "shares": pos.shares,
            "entry_price": pos.entry_price,
            "exit_price": exit_price,
            "pnl": pnl,
            "reason": reason,
        }
        self.closed_trades.append(trade)
        logger.info(f"[{symbol}] Closed @ {exit_price:.2f}, P&L: {pnl:+.2f} ({reason})")
        return pnl

    def total_value(self, current_prices: dict[str, float]) -> float:
        """Total portfolio value given current prices."""
        pos_value = sum(
            p.shares * current_prices.get(p.symbol, p.entry_price)
            for p in self.positions.values()
        )
        return self.cash + pos_value

    def to_dict(self) -> dict:
        return {
            "cash": self.cash,
            "positions": [
                {
                    "symbol": p.symbol,
                    "shares": p.shares,
                    "entry_price": p.entry_price,
                    "take_profit_price": p.take_profit_price,
                    "stop_loss_price": p.stop_loss_price,
                    "confidence": p.confidence,
                    "reason": p.reason,
                }
                for p in self.positions.values()
            ],
        }

    @classmethod
    def from_dict(cls, data: dict, starting_capital: float) -> "Portfolio":
        p = cls(starting_capital)
        p.cash = data["cash"]
        for pos in data.get("positions", []):
            p.positions[pos["symbol"]] = Position(**pos)
        return p
