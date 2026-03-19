from pathlib import Path

BASE_DIR = Path(__file__).parent

# Anthropic
ANTHROPIC_MODEL = "claude-haiku-4-5"  # cheap (~$0.002/week for 8 assets)

# OpenBB MCP (data tools only — NOT used for LLM inference)
OPENBB_MCP_URL = "http://127.0.0.1:8001/mcp"

# State
DB_PATH = BASE_DIR / "state" / "trader.db"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
BACKTEST_CACHE_DIR = BASE_DIR / "backtest" / "llm_cache"

# Paper trading
STARTING_CAPITAL = 100_000.0
MAX_POSITIONS = 20
MAX_ALLOCATION = 0.10        # 10% of portfolio per position (always binding)
STOP_LOSS_PCT = 0.05         # -5% stop-loss; effective risk = 0.5% of portfolio
MIN_LLM_CONFIDENCE = 0.60
LABEL_THRESHOLD = 0.015      # 1.5% threshold for buy/sell label
LABEL_HORIZON = 5            # next-5-day forward return

# Model training
TRAINING_YEARS = 2

# Scheduler timezone
TIMEZONE = "America/New_York"

# Asset watchlist
ASSETS = [
    {"symbol": "AAPL",   "type": "equity"},
    {"symbol": "MSFT",   "type": "equity"},
    {"symbol": "NVDA",   "type": "equity"},
    {"symbol": "SPY",    "type": "equity"},
    {"symbol": "QQQ",    "type": "equity"},
    {"symbol": "BTC",    "type": "crypto"},
    {"symbol": "ETH",    "type": "crypto"},
    {"symbol": "EURUSD", "type": "forex"},
]
