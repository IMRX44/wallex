"""SQLite persistence: quotes (for training), signals, orders, trades, equity, AI decisions, positions."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Iterable

from .types import Order, Position, TradeResult

SCHEMA = """
CREATE TABLE IF NOT EXISTS quotes (ts REAL, symbol TEXT, bid REAL, ask REAL, bid_vol REAL, ask_vol REAL,
    fair REAL, n_refs INTEGER, vol24 REAL);
CREATE INDEX IF NOT EXISTS ix_quotes_sym_ts ON quotes(symbol, ts);
CREATE TABLE IF NOT EXISTS signals (id INTEGER PRIMARY KEY, ts REAL, engine TEXT, symbol TEXT, features TEXT,
    prob REAL, decision TEXT, reason TEXT);
CREATE TABLE IF NOT EXISTS orders (client_id TEXT PRIMARY KEY, ts REAL, engine TEXT, symbol TEXT, side TEXT,
    type TEXT, price REAL, qty REAL, status TEXT, filled_qty REAL, avg_price REAL, fee REAL, tag TEXT, updated REAL);
CREATE TABLE IF NOT EXISTS trades (id INTEGER PRIMARY KEY, engine TEXT, symbol TEXT, entry_ts REAL, exit_ts REAL,
    entry_price REAL, exit_price REAL, qty REAL, pnl_tmn REAL, ret REAL, reason_in TEXT, reason_out TEXT, meta TEXT);
CREATE TABLE IF NOT EXISTS equity (ts REAL, equity REAL, cash_tmn REAL, usdt REAL, exposure REAL, drawdown REAL);
CREATE TABLE IF NOT EXISTS decisions (id INTEGER PRIMARY KEY, ts REAL, kind TEXT, model TEXT, input TEXT,
    output TEXT, rationale TEXT, ok INTEGER);
CREATE TABLE IF NOT EXISTS events (ts REAL, level TEXT, msg TEXT);
CREATE TABLE IF NOT EXISTS positions (id TEXT PRIMARY KEY, data TEXT);
CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT);
"""



def _json_default(v):
    """numpy scalars -> float; anything else (datetime, Path, ...) -> str."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return str(v)

class Store:
    def __init__(self, path: str | Path):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.executescript(SCHEMA)

    def _exec(self, sql: str, args: Iterable[Any] = ()) -> sqlite3.Cursor:
        with self._lock:
            return self.conn.execute(sql, tuple(args))

    def query(self, sql: str, args: Iterable[Any] = ()) -> list[dict]:
        with self._lock:
            cur = self.conn.execute(sql, tuple(args))
            cols = [c[0] for c in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]

    # ----- writes -----
    def add_quotes(self, rows: list[tuple]) -> None:
        if rows:
            with self._lock:
                self.conn.executemany("INSERT INTO quotes VALUES (?,?,?,?,?,?,?,?,?)", rows)

    def add_signal(self, engine: str, symbol: str, features: dict, prob: float | None, decision: str, reason: str) -> None:
        self._exec("INSERT INTO signals (ts,engine,symbol,features,prob,decision,reason) VALUES (?,?,?,?,?,?,?)",
                   (time.time(), engine, symbol, json.dumps(features, ensure_ascii=False, default=_json_default),
                    prob, decision, reason))

    def upsert_order(self, o: Order) -> None:
        self._exec("INSERT OR REPLACE INTO orders VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                   (o.client_id, o.created_ts, o.engine, o.symbol, o.side, o.type, o.price, o.qty, o.status,
                    o.filled_qty, o.avg_price, o.fee_quote, o.tag, o.updated_ts))

    def add_trade(self, t: TradeResult) -> None:
        self._exec("INSERT INTO trades (engine,symbol,entry_ts,exit_ts,entry_price,exit_price,qty,pnl_tmn,ret,"
                   "reason_in,reason_out,meta) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                   (t.engine, t.symbol, t.entry_ts, t.exit_ts, t.entry_price, t.exit_price, t.qty, t.pnl_tmn,
                    t.ret, t.reason_in, t.reason_out, json.dumps(t.meta, ensure_ascii=False, default=_json_default)))

    def add_equity(self, equity: float, cash_tmn: float, usdt: float, exposure: float, drawdown: float) -> None:
        self._exec("INSERT INTO equity VALUES (?,?,?,?,?,?)", (time.time(), equity, cash_tmn, usdt, exposure, drawdown))

    def add_decision(self, kind: str, model: str, inp: Any, out: Any, rationale: str, ok: bool) -> None:
        self._exec("INSERT INTO decisions (ts,kind,model,input,output,rationale,ok) VALUES (?,?,?,?,?,?,?)",
                   (time.time(), kind, model, json.dumps(inp, ensure_ascii=False, default=str)[:20000],
                    json.dumps(out, ensure_ascii=False, default=str), rationale, int(ok)))

    def event(self, level: str, msg: str) -> None:
        self._exec("INSERT INTO events VALUES (?,?,?)", (time.time(), level, msg))

    def save_position(self, p: Position) -> None:
        self._exec("INSERT OR REPLACE INTO positions VALUES (?,?)", (p.id, json.dumps(p.to_dict(), default=_json_default)))

    def delete_position(self, pid: str) -> None:
        self._exec("DELETE FROM positions WHERE id=?", (pid,))

    def load_positions(self) -> list[Position]:
        return [Position(**json.loads(r["data"])) for r in self.query("SELECT data FROM positions")]

    def set_kv(self, k: str, v: Any) -> None:
        self._exec("INSERT OR REPLACE INTO kv VALUES (?,?)", (k, json.dumps(v, default=str)))

    def get_kv(self, k: str, default: Any = None) -> Any:
        rows = self.query("SELECT v FROM kv WHERE k=?", (k,))
        return json.loads(rows[0]["v"]) if rows else default

    def prune_quotes(self, keep_days: float = 10) -> None:
        self._exec("DELETE FROM quotes WHERE ts < ?", (time.time() - keep_days * 86400,))
