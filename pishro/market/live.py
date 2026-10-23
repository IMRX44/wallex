"""Live market state for the bot.

* ``/v1/markets`` (one request, every ~2 s) gives best bid/ask of *every* Wallex market.
* Binance best bid/ask for all coins (one REST request, plus an optional WebSocket).
* Nobitex order books as an independent sanity check.
* Wallex WebSocket depth/trades for the markets we have orders in (faster fills/updates).

For each tradable toman market the feed keeps a ``PremiumState`` that is updated once per
closed minute, exactly like the backtest. At start-up it replays the last days of 1-minute
history (Wallex UDF + Binance klines) so statistics are warm from the first minute.
"""
from __future__ import annotations

import asyncio
import logging
import math
import time
from collections import deque

import pandas as pd

from ..exchange.binance import BinanceRef
from ..exchange.nobitex import NobitexClient
from ..exchange.wallex import WallexClient
from ..exchange.wallex_ws import WallexStream
from ..store import Store
from ..strategy.reversion import Params, PremiumState
from ..types import MarketInfo, Quote
from .fairvalue import USDT_TMN, FairValue, compute_fair
from .universe import Universe, load_excluded

log = logging.getLogger(__name__)
HIST_KEEP = 7 * 1440


class _MinuteBar:
    __slots__ = ("minute", "o", "h", "l", "c", "fair", "usdt")

    def __init__(self, minute: int, px: float, fair: float, usdt: float):
        self.minute, self.o, self.h, self.l, self.c, self.fair, self.usdt = minute, px, px, px, px, fair, usdt

    def add(self, px: float, fair: float, usdt: float) -> None:
        self.h, self.l, self.c, self.fair, self.usdt = max(self.h, px), min(self.l, px), px, fair, usdt


class LiveMarket:
    def __init__(self, cfg, client: WallexClient, binance: BinanceRef, store: Store | None = None,
                 nobitex: NobitexClient | None = None, stream: WallexStream | None = None):
        self.cfg = cfg
        self.client = client
        self.binance = binance
        self.nobitex = nobitex
        self.stream = stream
        self.store = store
        self.params = Params.from_cfg(cfg)
        self.universe = Universe(load_excluded(cfg.get("universe.excluded_file")), ["TMN"],
                                 float(cfg.get("universe.min_volume_24h_tmn", 0)),
                                 float(cfg.get("universe.max_spread_pct", 0.012)))
        self.max_dis = float(cfg.get("references.max_ref_disagreement", 0.012))
        self.infos: dict[str, MarketInfo] = {}
        self.quotes: dict[str, Quote] = {}
        self.fairs: dict[str, FairValue] = {}
        self.states: dict[str, PremiumState] = {}
        self.bars: dict[str, _MinuteBar] = {}
        self.history: dict[str, deque] = {}       # sym -> (ts, o, h, l, c, v_tmn, ref_usdt)
        self.usdt_history: deque = deque(maxlen=HIST_KEEP)
        self.vol_scale: dict[str, float] = {}
        self.nobitex_mids: dict[str, dict[str, float]] = {}
        self.depth: dict[str, dict[str, list]] = {}
        self.last_ok = 0.0
        self._nb_ts = 0.0
        self._universe_ts = 0.0
        self._binance_ok = False
        self.ticks = 0
        self.on_trade_cb = None
        self.watch: set[str] = set()      # markets with our orders: subscribed on the WebSocket

    # ------------------------------------------------------------------ helpers
    @property
    def usdt_tmn(self) -> float:
        q = self.quotes.get(USDT_TMN)
        return q.mid if q and q.valid else 0.0

    def data_age(self) -> float:
        return time.time() - self.last_ok if self.last_ok else 1e9

    def ref_age(self, base: str) -> float:
        ts = self.binance.ts.get(base)
        return time.time() - ts if ts else 1e9

    def to_tmn(self, asset: str, qty: float) -> float:
        if asset == "TMN":
            return qty
        q = self.quotes.get(f"{asset}TMN")
        if q and q.valid:
            return qty * q.bid
        q = self.quotes.get(f"{asset}USDT")
        if q and q.valid and self.usdt_tmn:
            return qty * q.bid * self.usdt_tmn
        return 0.0

    def book_value(self, symbol: str, side: str, levels: int = 3) -> float:
        """Toman value of the top ``levels`` of the book on ``side`` ('ask' for buying)."""
        d = self.depth.get(symbol, {}).get(side)
        if d:
            return sum(p * q for p, q in d[:levels])
        q = self.quotes.get(symbol)
        if not q:
            return 0.0
        return (q.ask * q.ask_vol) if side == "ask" else (q.bid * q.bid_vol)

    def vol24(self, symbol: str) -> float:
        q = self.quotes.get(symbol)
        return (q.quote_volume_24h if q else 0.0) * self.vol_scale.get(symbol, 1.0)

    # ------------------------------------------------------------------ warm-up
    async def warmup(self, days: float = 3.0) -> None:
        """Replay recent 1-minute history so statistics are ready immediately."""
        await self.refresh_markets(force_universe=True)
        try:
            await self.binance.load_symbols()
        except Exception as e:  # noqa: BLE001
            log.warning("binance symbols: %r", e)
        end = int(time.time()) // 60 * 60 - 60
        start = end - int(days * 86400)
        sem = asyncio.Semaphore(4)

        async def udf(sym):
            rows = []
            for f in range(start, end, 43200):
                async with sem:
                    try:
                        d = await self.client.candles(sym, "1", f, min(f + 43200, end))
                    except Exception as e:  # noqa: BLE001
                        log.info("warmup udf %s: %s", sym, e)
                        continue
                rows += list(zip(d["t"], d["o"], d["h"], d["l"], d["c"], d["v"]))
            return sym, pd.DataFrame(rows, columns=["ts", "o", "h", "l", "c", "v"]).drop_duplicates("ts").set_index("ts").sort_index()

        async def kl(base):
            async with sem:
                try:
                    k = await self.binance.klines(base, "1m", start * 1000, end * 1000)
                except Exception as e:  # noqa: BLE001
                    log.info("warmup klines %s: %s", base, e)
                    k = []
            return base, pd.Series({x[0] // 1000: float(x[4]) for x in k}, dtype=float)

        syms = sorted(self.universe.tradable)
        bases = sorted({self.infos[s].base for s in syms if self.infos[s].base in self.binance.symbols})
        # coins without a global market (e.g. USOON, XAUT): Wallex <COIN>USDT is the reference, as in compute_fair
        cross = sorted({f"{self.infos[s].base}USDT" for s in syms if self.infos[s].base not in self.binance.symbols
                        and f"{self.infos[s].base}USDT" in self.infos})
        res = await asyncio.gather(udf(USDT_TMN), *[udf(s) for s in syms], *[udf(s) for s in cross],
                                   *[kl(b) for b in bases])
        wx = {r[0]: r[1] for r in res if r[0] in self.infos or r[0] == USDT_TMN}
        refs = {r[0]: r[1] for r in res if r[0] in bases}
        grid_n = max(1, (end - start) // 60)
        for b, ser in list(refs.items()):
            # a global history with holes (blocked pages) would bias the baseline: prefer Wallex <COIN>USDT
            cov = ser.dropna().shape[0] / grid_n
            if cov < 0.8:
                log.warning("warm-up: %s reference history covers only %.0f%%", b, cov * 100)
                refs.pop(b)
                cs = f"{b}USDT"
                if cs in self.infos and cs not in cross:
                    cross.append(cs)
                    _, df = await udf(cs)
                    wx[cs] = df
        for cs in cross:
            df = wx.get(cs)
            if df is not None and not df.empty:
                refs[self.infos[cs].base] = df["c"].astype(float)
        grid = pd.Index(range(start - start % 60, end + 1, 60))
        usdt = wx.get(USDT_TMN)
        if usdt is None or usdt.empty:
            log.warning("warm-up: no USDTTMN history; statistics will warm up live")
            return
        usdt_c = usdt["c"].reindex(grid).ffill()
        for ts, c in usdt_c.dropna().items():
            self.usdt_history.append((int(ts), float(c)))
        n_ok = 0
        uc = usdt_c.to_numpy(float)
        for sym in syms:
            base = self.infos[sym].base
            df, ref = wx.get(sym), refs.get(base)
            if df is None or df.empty or ref is None or ref.empty:
                continue
            w = df.reindex(grid)
            c = w["c"].ffill().to_numpy(float)
            v = w["v"].fillna(0.0).to_numpy(float)
            o = w["o"].to_numpy(float)
            h = w["h"].to_numpy(float)
            lo = w["l"].to_numpy(float)
            r = ref.reindex(grid).ffill(limit=3).to_numpy(float)   # never carry a stale reference across a gap
            fair = r * uc
            st = self.states.setdefault(sym, PremiumState(sym, self.params))
            hist = self.history.setdefault(sym, deque(maxlen=HIST_KEEP))
            vol_sum = 0.0
            for k, ts in enumerate(grid):
                ci, fi = c[k], fair[k]
                if math.isnan(ci) or math.isnan(fi):
                    continue
                traded = v[k] > 0
                ok, hk, lk = (o[k], h[k], lo[k]) if traded else (ci, ci, ci)
                vt = v[k] * ci
                st.on_bar(int(ts), ci, lk, hk, fi, vt, uc[k], traded=traded)
                hist.append((int(ts), ok, hk, lk, ci, v[k], r[k]))
                if ts > end - 86400:
                    vol_sum += vt
            q = self.quotes.get(sym)
            if q and q.quote_volume_24h > 0 and vol_sum > 0:
                self.vol_scale[sym] = vol_sum / q.quote_volume_24h
            n_ok += 1
        log.info("warm-up done: %d/%d markets with %d days of history", n_ok, len(syms), days)

    # ------------------------------------------------------------------ polling
    async def refresh_markets(self, force_universe: bool = False) -> None:
        infos, quotes = await self.client.markets()
        now = time.time()
        # keep fresher WebSocket depth for subscribed markets
        for sym, d in self.depth.items():
            q = quotes.get(sym)
            if q and d.get("ts", 0) > now - 5 and d.get("bid") and d.get("ask"):
                q.bid, q.bid_vol = d["bid"][0]
                q.ask, q.ask_vol = d["ask"][0]
        self.infos, self.quotes = infos, quotes
        self.last_ok = now
        if force_universe or now - self._universe_ts > 60 * float(self.cfg.get("universe.refresh_minutes", 15)):
            self._universe_ts = now
            tr = self.universe.refresh(infos, quotes, self.usdt_tmn)
            log.info("universe: %d tradable toman markets (of %d listed)", len(tr), len(infos))
        else:
            self.universe.infos = infos

    async def refresh_refs(self) -> None:
        try:
            await self.binance.poll()
        except Exception as e:  # noqa: BLE001
            log.info("binance poll: %s", e)
        now = time.time()
        if self.nobitex and self.cfg.get("references.nobitex", True) and \
                now - self._nb_ts > float(self.cfg.get("references.nobitex_interval_sec", 10)):
            self._nb_ts = now
            try:
                self.nobitex_mids = await self.nobitex.mids()
            except Exception as e:  # noqa: BLE001
                log.debug("nobitex: %s", e)

    def update(self) -> None:
        """Recompute fair values and roll minute bars (call after every poll)."""
        now = time.time()
        minute = int(now // 60 * 60)
        usdt = self.usdt_tmn
        fairs = {}
        for sym in self.universe.tradable:
            info, q = self.infos.get(sym), self.quotes.get(sym)
            if not info or not q or not q.valid:
                continue
            fv = compute_fair(info.base, self.quotes, self.binance.mids, self.nobitex_mids)
            if fv is None:
                continue
            if fv.source == "binance" and self.ref_age(info.base) > 30:
                continue
            fairs[sym] = fv
            st = self.states.setdefault(sym, PremiumState(sym, self.params))
            bar = self.bars.get(sym)
            if bar is None or bar.minute != minute:
                if bar is not None:
                    self._close_bar(sym, st, bar)
                self.bars[sym] = _MinuteBar(minute, q.mid, fv.fair, usdt)
            else:
                bar.add(q.mid, fv.fair, usdt)
        if usdt and (not self.usdt_history or self.usdt_history[-1][0] < minute - 60):
            self.usdt_history.append((minute - 60, usdt))
        self.fairs = fairs
        self.ticks += 1

    def _close_bar(self, sym: str, st: PremiumState, bar: _MinuteBar) -> None:
        if st.last_ts >= bar.minute:
            return
        st.on_bar(bar.minute, bar.c, bar.l, bar.h, bar.fair, 0.0, bar.usdt, traded=True)
        ref = bar.fair / bar.usdt if bar.usdt else 0.0
        q = self.quotes.get(sym)
        # live minute volume is not observed; use the 24h average in the backtest's unit
        v = (self.vol24(sym) / 1440 / bar.c) if (q and bar.c > 0) else 0.0
        self.history.setdefault(sym, deque(maxlen=HIST_KEEP)).append(
            (bar.minute, bar.o, bar.h, bar.l, bar.c, v, ref))
        if self.store:
            fv = self.fairs.get(sym)
            q = self.quotes.get(sym)
            if q:
                self.store.add_quotes([(bar.minute, sym, q.bid, q.ask, q.bid_vol, q.ask_vol,
                                        fv.fair if fv else None, 1, q.quote_volume_24h)])

    # ------------------------------------------------------------------ WebSocket callbacks
    def on_depth(self, sym: str, side: str, levels: list) -> None:
        d = self.depth.setdefault(sym, {})
        levels = sorted(levels, key=lambda x: -x[0]) if side == "bid" else sorted(levels, key=lambda x: x[0])
        d[side] = levels
        d["ts"] = time.time()
        q = self.quotes.get(sym)
        if q and levels:
            if side == "bid":
                q.bid, q.bid_vol = levels[0]
            else:
                q.ask, q.ask_vol = levels[0]
            q.ts = d["ts"]

    def on_trade(self, sym: str, price: float) -> None:
        if self.on_trade_cb:
            self.on_trade_cb(sym, price)

    def history_frame(self) -> dict:
        """Recent history in the backtest's format (for on-line re-training)."""
        usdt = pd.DataFrame(list(self.usdt_history), columns=["ts", "c"]).drop_duplicates("ts").set_index("ts")
        markets = {}
        for sym, rows in self.history.items():
            if len(rows) < 600:
                continue
            df = pd.DataFrame(list(rows), columns=["ts", "o", "h", "l", "c", "v", "ref"]).drop_duplicates("ts").set_index("ts")
            markets[sym] = {"wallex": df[["o", "h", "l", "c", "v"]], "ref": df["ref"], "source": "live",
                            "base": sym[:-3]}
        return {"usdt": usdt, "markets": markets, "meta": {}}

    async def run(self, stop: asyncio.Event, on_tick=None) -> None:
        interval = float(self.cfg.get("exchange.poll_interval_sec", 2.0))
        while not stop.is_set():
            t0 = time.time()
            try:
                await asyncio.gather(self.refresh_markets(), self.refresh_refs())
                self.update()
                if self.stream:
                    self.stream.set_symbols(set(self.watch))
                if on_tick:
                    await on_tick()
            except Exception as e:  # noqa: BLE001 - keep looping; the risk layer watches staleness
                log.warning("market tick failed: %r", e)
            await asyncio.sleep(max(0.05, interval - (time.time() - t0)))
