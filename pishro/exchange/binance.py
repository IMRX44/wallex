"""Binance spot prices as the *global leader* reference.

``api.binance.com`` is geo-blocked from Iran, but the public market-data mirrors
``data-api.binance.vision`` (REST) and ``data-stream.binance.vision`` (WebSocket) answer.
One REST call returns best bid/ask for every symbol, so the whole reference set is
refreshed in a single request; the optional WebSocket pushes sub-second updates.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Callable

import httpx

log = logging.getLogger(__name__)


def _f(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


class BinanceRef:
    def __init__(self, rest: str = "https://data-api.binance.vision", ws: str = "wss://data-stream.binance.vision",
                 timeout: float = 8.0):
        self.rest = rest.rstrip("/")
        self.ws = ws.rstrip("/")
        self._client = httpx.AsyncClient(timeout=timeout, headers={"User-Agent": "pishro-bot/1.0"})
        self.mids: dict[str, float] = {}      # BASE -> USDT mid
        self.ts: dict[str, float] = {}
        self.symbols: set[str] = set()        # BASE assets with an active <BASE>USDT market
        self.last_ok = 0.0

    async def close(self) -> None:
        await self._client.aclose()

    async def load_symbols(self) -> set[str]:
        r = await self._client.get(f"{self.rest}/api/v3/exchangeInfo", params={"permissions": "SPOT"})
        r.raise_for_status()
        out = set()
        for s in r.json().get("symbols", []):
            if s.get("quoteAsset") == "USDT" and s.get("status") == "TRADING":
                out.add(s["baseAsset"].upper())
        self.symbols = out
        return out

    async def poll(self) -> dict[str, float]:
        r = await self._client.get(f"{self.rest}/api/v3/ticker/bookTicker")
        r.raise_for_status()
        now = time.time()
        for row in r.json():
            sym = row.get("symbol", "")
            if not sym.endswith("USDT"):
                continue
            b, a = _f(row.get("bidPrice")), _f(row.get("askPrice"))
            if b > 0 and a >= b:
                base = sym[:-4]
                self.mids[base] = (a + b) / 2
                self.ts[base] = now
        self.last_ok = now
        return self.mids

    async def klines(self, base: str, interval: str, start_ms: int, end_ms: int) -> list[list]:
        out, s = [], start_ms
        while s < end_ms:
            r = await self._client.get(f"{self.rest}/api/v3/klines", params={
                "symbol": f"{base}USDT", "interval": interval, "startTime": s, "endTime": end_ms, "limit": 1000})
            if r.status_code != 200:
                break
            k = r.json()
            if not k:
                break
            out.extend(k)
            s = k[-1][0] + 60_000
            if len(k) < 1000:
                break
        return out

    async def stream(self, bases: list[str], stop: asyncio.Event, on_update: Callable[[str], None] | None = None) -> None:
        """bookTicker WebSocket for ``bases``; reconnects forever, REST polling covers gaps."""
        import websockets
        while not stop.is_set():
            streams = "/".join(f"{b.lower()}usdt@bookTicker" for b in bases[:200])
            if not streams:
                await asyncio.sleep(5)
                continue
            try:
                async with websockets.connect(f"{self.ws}/stream?streams={streams}", ping_interval=20) as ws:
                    async for raw in ws:
                        if stop.is_set():
                            break
                        d = json.loads(raw).get("data", {})
                        sym = d.get("s", "")
                        b, a = _f(d.get("b")), _f(d.get("a"))
                        if sym.endswith("USDT") and b > 0 and a >= b:
                            base = sym[:-4]
                            self.mids[base] = (a + b) / 2
                            self.ts[base] = time.time()
                            if on_update:
                                on_update(base)
            except Exception as e:  # noqa: BLE001 - network drops are expected
                log.info("binance ws reconnect: %r", e)
                await asyncio.sleep(3)
