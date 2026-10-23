"""Premium Reversion: the trading logic, shared verbatim by the live engine and the backtest.

Idea
----
Every Wallex toman market X/TMN has a *fair* price implied by the global market:

    fair = Binance X/USDT mid × Wallex USDT/TMN mid

A local market never trades exactly at fair. It carries a *premium* (local demand, rial
frictions) that drifts slowly. We learn each market's normal premium with an EWMA
(``baseline``) and its typical noise (``std``). The tradable event is a *temporary*
deviation below that normal level: a local seller dumps into the thin Wallex book, or
Wallex has not yet followed a move that already happened on Binance (lead-lag). Both
revert, because arbitrageurs and the next local buyers pull the price back to
``fair × (1 + baseline)``.

    buy  level  = fair × (1 + baseline) × (1 − edge)
    sell target = max(fair × (1 + baseline) × (1 + exit_edge), break-even × (1 + min_profit))

``edge`` adapts to fees and to the market's own premium noise. Gates block entries when
the global market is crashing (falling knife), USDT/TMN is jumping, or the market's
premium behaviour changed regime. Every number here is computed from 1-minute bars in
both live trading and the backtest, so backtest and live decisions use the same inputs.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

FEATURES = [
    "edge", "std", "std_ratio", "baseline", "dev_now", "ref_ret_5", "ref_ret_15", "ref_ret_60",
    "mkt_ret_15", "mkt_ret_60", "usdt_ret_15", "usdt_ret_60", "log_vol_24h", "revert_rate", "n_dips",
    "hour_sin", "hour_cos",
]

DIP_DEPTH = 0.006      # a "dip" for revert statistics: low at least 0.6% under normal premium
DIP_WINDOW = 60        # minutes allowed for a dip to recover
STATS_WINDOW = 3 * 1440


@dataclass
class Params:
    baseline_halflife_min: float = 30
    long_halflife_min: float = 2880
    mid_halflife_min: float = 360
    warmup_minutes: int = 90
    edge_floor_zero_fee: float = 0.004
    edge_floor_fee_mult: float = 1.6
    edge_vol_mult: float = 2.5
    edge_max: float = 0.03
    min_profit: float = 0.001
    exit_edge: float = 0.0
    stop_pct: float = 0.025
    max_hold_minutes: float = 240
    shock_ref_drop: float = 0.005
    shock_usdt_jump: float = 0.003
    regime_std_ratio_max: float = 2.2

    @classmethod
    def from_cfg(cls, cfg) -> "Params":
        s = cfg.get("strategy", {}) or {}
        kw = {k: type(getattr(cls, k))(s[k]) for k in cls.__dataclass_fields__ if k in s}
        return cls(**kw)


def _alpha(halflife_bars: float) -> float:
    return 1 - 0.5 ** (1 / max(halflife_bars, 1e-9))


@dataclass
class PremiumState:
    """Premium statistics of one market, updated once per closed 1-minute bar."""
    symbol: str
    p: Params
    mean: float = 0.0
    var: float = 0.0
    var_mid: float = 0.0
    var_long: float = 0.0
    n: int = 0
    last_ts: float = 0.0
    bars: deque = field(default_factory=lambda: deque(maxlen=STATS_WINDOW + 10))
    vol_window: deque = field(default_factory=lambda: deque(maxlen=1440))
    vol_sum: float = 0.0
    dip_outcomes: deque = field(default_factory=deque)
    active_dip: tuple | None = None   # (start_ts, target, bars_elapsed)

    # ------------------------------------------------------------------ update
    def on_bar(self, ts: float, close: float, low: float, high: float, fair: float,
               vol_tmn: float = 0.0, usdt: float = 0.0, traded: bool = True) -> None:
        if fair <= 0 or close <= 0:
            return
        prev_adj = self.fair_adj(fair) if self.n else 0.0
        self._track_dip(ts, low, high, prev_adj)
        if traded:
            dev = close / fair - 1
            if self.n == 0:
                self.mean, self.var, self.var_mid, self.var_long = dev, 1e-6, 1e-6, 1e-6
            else:
                for attr_m, hl in (("var", self.p.baseline_halflife_min),
                                   ("var_mid", self.p.mid_halflife_min),
                                   ("var_long", self.p.long_halflife_min)):
                    a = _alpha(hl)
                    d = dev - self.mean
                    setattr(self, attr_m, (1 - a) * (getattr(self, attr_m) + a * d * d))
                self.mean += _alpha(self.p.baseline_halflife_min) * (dev - self.mean)
            self.n += 1
        self.bars.append((ts, close, fair, usdt))
        if len(self.vol_window) == self.vol_window.maxlen:
            self.vol_sum -= self.vol_window[0]
        self.vol_window.append(vol_tmn)
        self.vol_sum += vol_tmn
        self.last_ts = ts

    def _track_dip(self, ts: float, low: float, high: float, adj: float) -> None:
        if adj <= 0:
            return
        if self.active_dip is not None:
            start, target, k = self.active_dip
            if high >= target:
                self.dip_outcomes.append((start, 1))
                self.active_dip = None
            elif k + 1 >= DIP_WINDOW:
                self.dip_outcomes.append((start, 0))
                self.active_dip = None
            else:
                self.active_dip = (start, target, k + 1)
        elif low <= adj * (1 - DIP_DEPTH):
            self.active_dip = (ts, adj, 0)
        while self.dip_outcomes and ts - self.dip_outcomes[0][0] > STATS_WINDOW * 60:
            self.dip_outcomes.popleft()

    # ------------------------------------------------------------------ views
    @property
    def ready(self) -> bool:
        return self.n >= self.p.warmup_minutes

    @property
    def baseline(self) -> float:
        return self.mean

    @property
    def std(self) -> float:
        return max(math.sqrt(max(self.var, 0.0)), 0.0005)

    @property
    def std_ratio(self) -> float:
        lo = max(math.sqrt(max(self.var_long, 0.0)), 0.0005)
        return max(math.sqrt(max(self.var_mid, 0.0)), 0.0005) / lo

    @property
    def revert_rate(self) -> float:
        n = len(self.dip_outcomes)
        ok = sum(x for _, x in self.dip_outcomes)
        return (ok + 1) / (n + 2)

    def fair_adj(self, fair: float) -> float:
        return fair * (1 + self.mean)

    def _ret(self, idx: int, k: int) -> float:
        if len(self.bars) <= k:
            return 0.0
        a, b = self.bars[-1][idx], self.bars[-1 - k][idx]
        return a / b - 1 if b > 0 else 0.0

    def ref_ret(self, k: int) -> float:
        return self._ret(2, k)

    def mkt_ret(self, k: int) -> float:
        return self._ret(1, k)

    def usdt_ret(self, k: int) -> float:
        return self._ret(3, k)

    def features(self, ts: float, edge: float, dev_now: float, vol24: float | None = None) -> dict:
        hour = ((ts / 3600.0) + 3.5) % 24
        vol = self.vol_sum if vol24 is None else vol24
        return {
            "edge": edge, "std": self.std, "std_ratio": self.std_ratio, "baseline": self.mean,
            "dev_now": dev_now, "ref_ret_5": self.ref_ret(5), "ref_ret_15": self.ref_ret(15),
            "ref_ret_60": self.ref_ret(60), "mkt_ret_15": self.mkt_ret(15), "mkt_ret_60": self.mkt_ret(60),
            "usdt_ret_15": self.usdt_ret(15), "usdt_ret_60": self.usdt_ret(60),
            "log_vol_24h": math.log1p(max(vol, 0.0)), "revert_rate": self.revert_rate,
            "n_dips": float(len(self.dip_outcomes)), "hour_sin": math.sin(2 * math.pi * hour / 24),
            "hour_cos": math.cos(2 * math.pi * hour / 24),
        }


# ---------------------------------------------------------------------- rules
def entry_edge(st: PremiumState, maker_fee: float, taker_fee: float, p: Params,
               agent_mult: float = 1.0, taker: bool = False) -> float:
    """Required discount under the normal-premium price before buying."""
    round_trip = (taker_fee if taker else maker_fee) + maker_fee
    floor = p.edge_floor_zero_fee
    if round_trip > 0:
        floor = max(floor, p.edge_floor_fee_mult * round_trip + p.min_profit)
    e = max(floor, p.edge_vol_mult * st.std)
    return min(e * max(agent_mult, 0.8), p.edge_max)


def entry_level(st: PremiumState, fair: float, edge: float) -> float:
    return st.fair_adj(fair) * (1 - edge)


def breakeven(entry_price: float, buy_fee: float, sell_fee: float) -> float:
    """Sell price that returns exactly the toman spent (buy fee is taken in the coin)."""
    return entry_price / ((1 - buy_fee) * (1 - sell_fee))


def exit_target(st: PremiumState, fair: float, entry_price: float, buy_fee: float, sell_fee: float,
                p: Params, exit_edge: float | None = None) -> float:
    ee = p.exit_edge if exit_edge is None else exit_edge
    return max(st.fair_adj(fair) * (1 + ee), breakeven(entry_price, buy_fee, sell_fee) * (1 + p.min_profit))


def stop_price(entry_price: float, p: Params) -> float:
    return entry_price * (1 - p.stop_pct)


def size_from_prob(prob: float | None, min_prob: float, full_size_prob: float) -> float:
    """Position-size multiplier from the model's calibrated probability.

    p < min_prob -> 0 (skip); p = min_prob -> 0.5; p >= full_size_prob -> 1.0.
    Without a model (prob None) the rule-based size is used as is.
    """
    if prob is None:
        return 1.0
    if prob < min_prob:
        return 0.0
    return 0.5 + 0.5 * min(1.0, (prob - min_prob) / max(full_size_prob - min_prob, 1e-6))


def gates(st: PremiumState, p: Params) -> str:
    """Return a reason string if entries must be blocked for this market, else ''."""
    if not st.ready:
        return "warmup"
    if st.ref_ret(5) <= -p.shock_ref_drop:
        return "shock_ref"          # global market falling: Wallex will follow, don't catch the knife
    if st.usdt_ret(5) >= p.shock_usdt_jump:
        return "shock_usdt"         # tether jump moves every fair price; wait for it to settle
    if st.std_ratio > p.regime_std_ratio_max:
        return "regime"             # premium noise far above its history: the market changed behaviour
    return ""
