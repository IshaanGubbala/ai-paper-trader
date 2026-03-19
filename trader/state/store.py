import sqlite3
import json
from pathlib import Path
from datetime import datetime, timezone


class Store:
    def __init__(self, db_path=None):
        if db_path is None:
            from trader.config import DB_PATH
            db_path = DB_PATH
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def initialize(self):
        migration = Path(__file__).parent / "migrations" / "001_init.sql"
        sql = migration.read_text()
        with self._connect() as conn:
            conn.executescript(sql)

    # --- Thesis ---

    def save_thesis(self, asset, stance, confidence, reasoning, horizon):
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO thesis (asset, stance, confidence, reasoning, horizon, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (asset, stance, confidence, reasoning, horizon,
                 datetime.now(timezone.utc).isoformat())
            )

    def get_thesis(self, asset):
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM thesis WHERE asset = ?", (asset,)).fetchone()
        return dict(row) if row else None

    def get_all_thesis(self):
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM thesis").fetchall()
        return [dict(r) for r in rows]

    # --- Portfolio ---

    def save_portfolio(self, cash, positions):
        with self._connect() as conn:
            conn.execute("DELETE FROM portfolio")
            conn.execute(
                "INSERT INTO portfolio (cash, positions_json, updated_at) VALUES (?, ?, ?)",
                (cash, json.dumps(positions), datetime.now(timezone.utc).isoformat())
            )

    def get_portfolio(self):
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM portfolio ORDER BY id DESC LIMIT 1"
            ).fetchone()
        if row is None:
            return None
        return {"cash": row["cash"], "positions": json.loads(row["positions_json"])}

    # --- Trades ---

    def log_trade(self, symbol, action, shares, price, reason=""):
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO trades (symbol, action, shares, price, reason, executed_at) VALUES (?, ?, ?, ?, ?, ?)",
                (symbol, action, shares, price, reason,
                 datetime.now(timezone.utc).isoformat())
            )

    def get_trades(self):
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM trades ORDER BY executed_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    # --- Snapshots ---

    def save_snapshot(self, value):
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO snapshots (value, recorded_at) VALUES (?, ?)",
                (value, datetime.now(timezone.utc).isoformat())
            )

    def get_snapshots(self):
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM snapshots ORDER BY recorded_at ASC"
            ).fetchall()
        return [dict(r) for r in rows]
