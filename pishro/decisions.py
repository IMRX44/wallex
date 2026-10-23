"""Decision log: one JSON line per decision, with the signal that triggered it and why.

Required by the challenge ("for every trade, store the signal and why it bought/sold").
Written to ``data/decisions.jsonl`` (append-only, easy to hand in) and mirrored to SQLite
for the dashboard. Every order the bot sends has a matching entry here, which also
proves the trade was algorithmic.
"""
from __future__ import annotations

import datetime as dt
import json
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

TEHRAN = dt.timezone(dt.timedelta(hours=3, minutes=30))


def _clean(x: Any) -> Any:
    if isinstance(x, float):
        return round(x, 8) if x == x and abs(x) != float("inf") else None
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    return x


class DecisionLog:
    def __init__(self, path: str | Path, store=None, mode: str = "paper"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.store = store
        self.mode = mode
        self.recent: deque = deque(maxlen=400)
        self._lock = threading.Lock()
        self.seq = 0

    def log(self, kind: str, symbol: str = "", reason_fa: str = "", /, **data: Any) -> dict:
        now = time.time()
        self.seq += 1
        core = {"ts": round(now, 3), "time_tehran": dt.datetime.fromtimestamp(now, TEHRAN).strftime("%Y-%m-%d %H:%M:%S"),
                "seq": self.seq, "mode": self.mode, "kind": kind, "symbol": symbol, "reason": reason_fa}
        rec = {**core, **{k: v for k, v in _clean(data).items() if k not in core}}
        line = json.dumps(rec, ensure_ascii=False, default=str)
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        self.recent.append(rec)
        if self.store is not None:
            self.store.add_signal(kind, symbol, rec, data.get("prob"), kind, reason_fa)
        return rec

    def tail(self, n: int = 100, kinds: set[str] | None = None) -> list[dict]:
        items = [r for r in self.recent if not kinds or r["kind"] in kinds]
        return items[-n:]
