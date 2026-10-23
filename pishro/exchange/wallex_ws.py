"""Wallex v1 WebSocket (``wss://api.wallex.ir/ws``): order-book depth and public trades.

Subscribe:   ["subscribe", {"channel": "BTCTMN@buyDepth"}]
Messages:    ["BTCTMN@buyDepth", [{"price": .., "quantity": ..}, ...]]
             ["BTCTMN@trade", {...}] or a list of trades
The server drops connections after ~30 minutes; we reconnect and resubscribe.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Callable

log = logging.getLogger(__name__)


def _f(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def parse_levels(data) -> list[tuple[float, float]]:
    out = []
    for x in data or []:
        if isinstance(x, dict):
            p, q = _f(x.get("price")), _f(x.get("quantity"))
        elif isinstance(x, (list, tuple)) and len(x) >= 2:
            p, q = _f(x[0]), _f(x[1])
        else:
            continue
        if p > 0 and q > 0:
            out.append((p, q))
    return out


def parse_trades(data) -> list[float]:
    rows = data if isinstance(data, list) else [data]
    return [_f(r.get("price")) for r in rows if isinstance(r, dict) and _f(r.get("price")) > 0]


class WallexStream:
    def __init__(self, url: str = "wss://api.wallex.ir/ws"):
        self.url = url
        self.symbols: set[str] = set()
        self._resubscribe = asyncio.Event()

    def set_symbols(self, symbols: set[str]) -> None:
        if symbols != self.symbols:
            self.symbols = set(symbols)
            self._resubscribe.set()

    async def run(self, stop: asyncio.Event,
                  on_depth: Callable[[str, str, list[tuple[float, float]]], None],
                  on_trade: Callable[[str, float], None] | None = None) -> None:
        import websockets
        while not stop.is_set():
            self._resubscribe.clear()
            syms = sorted(self.symbols)
            if not syms:
                await asyncio.sleep(2)
                continue
            try:
                async with websockets.connect(self.url, ping_interval=20, close_timeout=3) as ws:
                    for s in syms:
                        for ch in ("buyDepth", "sellDepth", "trade"):
                            await ws.send(json.dumps(["subscribe", {"channel": f"{s}@{ch}"}]))
                    while not stop.is_set() and not self._resubscribe.is_set():
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=5)
                        except asyncio.TimeoutError:
                            continue
                        msg = json.loads(raw)
                        if not (isinstance(msg, list) and len(msg) == 2 and isinstance(msg[0], str) and "@" in msg[0]):
                            continue
                        sym, kind = msg[0].split("@", 1)
                        if kind == "buyDepth":
                            on_depth(sym, "bid", parse_levels(msg[1]))
                        elif kind == "sellDepth":
                            on_depth(sym, "ask", parse_levels(msg[1]))
                        elif kind == "trade" and on_trade:
                            for p in parse_trades(msg[1]):
                                on_trade(sym, p)
            except Exception as e:  # noqa: BLE001 - REST polling covers gaps; staleness gate protects
                log.info("wallex ws reconnect: %r", e)
                await asyncio.sleep(2)
