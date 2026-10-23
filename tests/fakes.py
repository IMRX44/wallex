"""In-memory fake Wallex + Binance driven by synthetic minute data and a fake clock.

Used by the end-to-end test to run the real Bot (paper mode) through hours of market
time in seconds, exercising warm-up, orders, fills, exits, risk and the decision log.
"""
from __future__ import annotations

import numpy as np

from pishro.types import MarketInfo, Quote


class Clock:
    def __init__(self, t: float):
        self.t = float(t)

    def __call__(self) -> float:
        return self.t


class FakeWallex:
    def __init__(self, hist: dict, clock: Clock, spread: float = 0.0015):
        self.clock = clock
        self.spread = spread
        self.frames = {"USDTTMN": hist["usdt"]}
        for sym, m in hist["markets"].items():
            self.frames[sym] = m["wallex"]
        self.base_url = "fake"
        self.error_times: list[float] = []
        self.phase = 0  # 0=open 1=low 2=high 3=close within the minute

    def errors_last_minute(self) -> int:
        return 0

    async def close(self) -> None:
        pass

    def _row(self, sym: str):
        df = self.frames[sym]
        minute = int(self.clock() // 60 * 60)
        idx = df.index.searchsorted(minute, side="right") - 1
        return df.iloc[max(idx, 0)]

    async def markets(self):
        infos, quotes = {}, {}
        for sym, df in self.frames.items():
            base, quote = sym[:-3], "TMN"
            infos[sym] = MarketInfo(sym, base, quote, tick=0.0, step=0.0, min_qty=0.0, min_notional=100_000)
            r = self._row(sym)
            px = [r["o"], r["l"], r["h"], r["c"]][self.phase]
            hs = self.spread / 2
            quotes[sym] = Quote(sym, bid=px * (1 - hs), ask=px * (1 + hs), bid_vol=1e9 / px, ask_vol=1e9 / px,
                                last=px, quote_volume_24h=5e10, ts=self.clock())
        return infos, quotes

    async def candles(self, symbol: str, resolution: str, frm: int, to: int):
        df = self.frames[symbol]
        d = df[(df.index >= frm) & (df.index < to)]
        return {"t": list(d.index), "o": list(d["o"]), "h": list(d["h"]), "l": list(d["l"]), "c": list(d["c"]),
                "v": list(d["v"])}

    async def global_stats(self):
        return {}


class FakeBinance:
    def __init__(self, hist: dict, clock: Clock):
        self.clock = clock
        self.refs = {m["base"]: m["ref"] for m in hist["markets"].values()}
        self.mids: dict[str, float] = {}
        self.ts: dict[str, float] = {}
        self.symbols = set(self.refs)

    async def load_symbols(self):
        return self.symbols

    async def poll(self):
        minute = int(self.clock() // 60 * 60)
        for b, s in self.refs.items():
            i = max(s.index.searchsorted(minute, side="right") - 1, 0)
            self.mids[b] = float(s.iloc[i])
            self.ts[b] = self.clock()
        return self.mids

    async def klines(self, base: str, interval: str, start_ms: int, end_ms: int):
        s = self.refs[base]
        d = s[(s.index >= start_ms // 1000) & (s.index < end_ms // 1000)]
        return [[int(t) * 1000, 0, 0, 0, float(v)] for t, v in d.items()]

    async def stream(self, *a, **k):
        return None

    async def close(self):
        pass


def install(bot, wallex: FakeWallex, binance: FakeBinance) -> None:
    bot.client = wallex
    bot.market.client = wallex
    bot.binance = binance
    bot.market.binance = binance
    bot.nobitex = None
    bot.market.nobitex = None
    bot.stream = None
    bot.market.stream = None


async def drive(bot, wallex: FakeWallex, clock: Clock, minutes: int, ticks_per_minute: int = 4) -> None:
    for _ in range(minutes):
        for ph in range(ticks_per_minute):
            wallex.phase = ph % 4
            clock.t += 60 / ticks_per_minute
            await bot.market.refresh_markets()
            await bot.market.refresh_refs()
            bot.market.update()
            await bot._on_tick()
            for asset, (total, locked) in (await bot.broker.balances()).items():
                assert total >= -1e-6, f"negative balance {asset}={total}"
                assert locked >= -1e-6, f"negative lock {asset}={locked}"
    _ = np  # keep numpy import used for type hints in some environments
