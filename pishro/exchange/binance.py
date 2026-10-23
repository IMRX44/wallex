"""Binance spot prices as the *global leader* reference.

``api.binance.com`` is geo-blocked from Iran, and in practice the public mirror
``data-api.binance.vision`` answers 451 from Iranian IPs too. ``fallback_rest`` lists
Binance-compatible REST APIs (MEXC: same /api/v3 paths and payloads) tried in order when
the current one fails; the first that answers is kept.
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
                 timeout: float = 8.0, fallback_rest: list[str] | None = None):
        self.rests = [u.rstrip("/") for u in [rest, *(fallback_rest or [])] if u]
        self.rest = self.rests[0]
        self._dead: dict[str, float] = {}     # base url -> until (legal block 451/403: do not retry for a while)
        self.ws = ws.rstrip("/")
        self._client = httpx.AsyncClient(timeout=timeout, headers={"User-Agent": "pishro-bot/1.0"})
        self.mids: dict[str, float] = {}      # BASE -> USDT mid
        self.ts: dict[str, float] = {}
        self.symbols: set[str] = set()        # BASE assets with an active <BASE>USDT market
        self.last_ok = 0.0

    async def close(self) -> None:
        await self._client.aclose()

    async def _get(self, path: str, params: dict | None = None, timeout: float | None = None) -> httpx.Response:
        """GET on the current REST base; on a network error or non-200 move to the next base."""
        err: Exception | None = None
        for _ in range(len(self.rests)):
            try:
                r = await self._client.get(f"{self.rest}{path}", params=params,
                                           **({"timeout": timeout} if timeout else {}))
                if r.status_code == 200:
                    return r
                err = httpx.HTTPStatusError(f"{r.status_code} from {self.rest}{path}", request=r.request, response=r)
            except httpx.HTTPError as e:
                err = e
            if isinstance(err, httpx.HTTPStatusError) and err.response.status_code in (451, 403)                     and self.rest == self.rests[0] and len(self.rests) > 1:
                self._dead[self.rest] = time.time() + 1800
            if len(self.rests) == 1:
                break
            alive = [u for u in self.rests if self._dead.get(u, 0) < time.time()]
            if alive == [self.rest]:      # the only live venue failed transiently: no point switching
                break
            i = (self.rests.index(self.rest) + 1) % len(self.rests)
            log.warning("reference %s failed (%r); switching to %s", self.rest, err, self.rests[i])
            self.rest = self.rests[i]
        raise err  # type: ignore[misc]

    async def load_symbols(self) -> set[str]:
        # MEXC's full exchangeInfo is ~2 MB: allow time and retry instead of failing the warm-up
        r = None
        for attempt in range(3):
            try:
                r = await self._get("/api/v3/exchangeInfo", timeout=40)
                break
            except httpx.HTTPError as e:
                log.warning("exchangeInfo attempt %d failed: %r", attempt + 1, e)
                await asyncio.sleep(2)
        if r is None:
            raise httpx.HTTPError("exchangeInfo unavailable after 3 attempts")
        out = set()
        for s in r.json().get("symbols", []):
            # Binance: status TRADING; MEXC: status "1"
            if s.get("quoteAsset") == "USDT" and s.get("status") in ("TRADING", "1", 1, "ENABLED"):
                out.add(s["baseAsset"].upper())
        self.symbols = out
        return out

    async def poll(self) -> dict[str, float]:
        r = await self._get("/api/v3/ticker/bookTicker")
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
        page_ms = 500 * 60_000            # smallest page among the venues (MEXC)
        while s < end_ms:
            r = None
            for attempt in range(3):      # MEXC answers 451 intermittently to datacenter IPs (Liara)
                try:
                    r = await self._get("/api/v3/klines", params={
                        "symbol": f"{base}USDT", "interval": interval, "startTime": s, "endTime": end_ms, "limit": 1000})
                    break
                except httpx.HTTPError:
                    await asyncio.sleep(1.0 + 2 * attempt)
            if r is None:                 # give up on this page only; the gap stays a gap (never forward-filled)
                s += page_ms
                continue
            k = r.json()
            if not isinstance(k, list) or not k:
                break
            out.extend(k)
            nxt = int(k[-1][0]) + 60_000
            if nxt <= s:   # page sizes differ (Binance 1000, MEXC 500): stop only when no progress
                break
            s = nxt
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
