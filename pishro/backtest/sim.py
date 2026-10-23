"""Minute-by-minute portfolio backtest of the Premium Reversion strategy.

It uses the *same* ``PremiumState``, entry/exit rules, ML model and ``RiskManager`` as
the live bot. Fill assumptions are conservative:

* Orders for minute t are computed from data up to the close of minute t-1 (no look-ahead;
  the live bot re-quotes every few seconds, so this is the pessimistic case).
* A resting buy at L fills only if minute t traded at least ``fill_margin`` below L with
  enough volume; it fills at L as maker.
* The sniper (taker) buys only if minute t *closed* below its level, at close + slippage.
* The exit target fills only if the high trades ``fill_margin`` through it.
* Stops and time-stops sell at the worse of stop/open (gaps) or close, minus slippage.
* If stop and target touch in the same minute, the stop is assumed first.
* Unfilled orders reserve cash exactly as on the exchange.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..risk import RiskManager
from ..strategy.reversion import (FEATURES, Params, PremiumState, entry_edge, entry_level, exit_target,
                                  gates, size_from_prob, stop_price)

log = logging.getLogger(__name__)


@dataclass
class SimSettings:
    maker_fee: float = 0.0025
    taker_fee: float = 0.003
    capital: float = 5_000_000
    fill_margin: float = 0.0005
    min_minute_vol_tmn: float = 1_000_000
    taker_slippage: float = 0.0008
    resting_markets: int = 4
    sniper: bool = True
    position_frac: float = 0.20
    min_prob: float = 0.55
    full_size_prob: float = 0.72
    collect: bool = False
    collect_notional: float = 1_000_000
    entry_cutoff_min: float = 120
    flatten_min: float = 45
    min_order_tmn: float = 200_000

    @classmethod
    def from_cfg(cls, cfg, maker: float, taker: float, **kw) -> "SimSettings":
        b, s, ml, r, ch = (cfg.get(k, {}) or {} for k in ("backtest", "strategy", "ml", "risk", "challenge"))
        return cls(maker_fee=maker, taker_fee=taker, capital=float(cfg.get("initial_capital_tmn", 5e6)),
                   fill_margin=float(b.get("fill_margin", 0.0005)),
                   min_minute_vol_tmn=float(b.get("min_minute_volume_tmn", 1e6)),
                   taker_slippage=float(b.get("taker_slippage", 0.0008)),
                   resting_markets=int(s.get("resting_markets", 4)), sniper=bool(s.get("sniper_enabled", True)),
                   position_frac=float(s.get("position_frac", 0.2)), min_prob=float(ml.get("min_prob", 0.55)),
                   full_size_prob=float(ml.get("full_size_prob", 0.72)),
                   min_order_tmn=float(r.get("min_order_tmn", 2e5)), **kw)


@dataclass
class SimResult:
    trades: pd.DataFrame
    equity: pd.Series
    events: pd.DataFrame
    settings: SimSettings
    blocked: dict = field(default_factory=dict)


def prepare(hist: dict, start: int | None = None, end: int | None = None) -> dict:
    """Align every market, its reference and USDT/TMN on one 1-minute grid."""
    usdt = hist["usdt"]
    t0 = int(start if start is not None else usdt.index.min())
    t1 = int(end if end is not None else usdt.index.max())
    grid = np.arange(t0 - t0 % 60, t1 + 1, 60)
    usdt_c = usdt["c"].reindex(grid).ffill().to_numpy(float)
    out = {"grid": grid, "usdt": usdt_c, "markets": {}}
    for sym, m in hist["markets"].items():
        wx = m["wallex"].reindex(grid)
        v = wx["v"].fillna(0.0).to_numpy(float)
        c = wx["c"].ffill().to_numpy(float)
        o = np.where(np.isnan(wx["o"].to_numpy(float)), c, wx["o"].to_numpy(float))
        h = np.where(np.isnan(wx["h"].to_numpy(float)), c, wx["h"].to_numpy(float))
        lo = np.where(np.isnan(wx["l"].to_numpy(float)), c, wx["l"].to_numpy(float))
        ref = m["ref"].reindex(grid).ffill().to_numpy(float)
        fair = ref * usdt_c
        if np.nanmedian(np.abs(c / fair - 1)) > 0.2:   # wrong unit/mapping (e.g. 1000x tokens)
            log.warning("%s: Wallex price far from reference, skipped", sym)
            continue
        out["markets"][sym] = {"o": o, "h": h, "l": lo, "c": c, "v": v, "traded": v > 0,
                               "vol_tmn": v * np.nan_to_num(c), "fair": fair}
    return out


def run(prep: dict, params: Params, s: SimSettings, model=None, trade_start: int | None = None,
        trade_end: int | None = None, risk_cfg=None) -> SimResult:
    grid, usdt = prep["grid"], prep["usdt"]
    mk = prep["markets"]
    syms = sorted(mk)
    trade_start = trade_start if trade_start is not None else int(grid[0])
    trade_end = trade_end if trade_end is not None else int(grid[-1])
    states = {m: PremiumState(m, params) for m in syms}
    risk = RiskManager(risk_cfg) if risk_cfg is not None else RiskManager(_DefaultCfg())
    cash = s.capital
    positions: dict[str, dict] = {}
    resting: dict[str, dict] = {}
    snipers: dict[str, dict] = {}
    trades, events = [], []
    eq_ts, eq_val = [], []
    blocked: dict[str, int] = {}
    flatten_all = False

    def close_pos(m: str, px: float, fee: float, reason: str, ts: float) -> None:
        nonlocal cash
        pos = positions.pop(m)
        proceeds = pos["qty"] * px * (1 - fee)
        cash += proceeds
        pnl = proceeds - pos["cost"]
        rec = {"symbol": m, "entry_ts": pos["ts"], "exit_ts": ts, "entry": pos["entry"], "exit": px,
               "qty": pos["qty"], "cost": pos["cost"], "proceeds": proceeds, "pnl": pnl, "ret": pnl / pos["cost"],
               "reason": reason, "mode": pos["mode"], "prob": pos["prob"], "edge": pos["edge"],
               "hold_min": (ts - pos["ts"]) / 60}
        if s.collect:
            events.append({**pos["feats"], **rec, "y": int(pnl > 0), "ts": pos["ts"]})
        else:
            trades.append(rec)
            risk.on_trade_closed(rec["ret"], ts)

    for i, ts in enumerate(grid):
        ts = int(ts)
        in_window = trade_start <= ts <= trade_end
        end_flatten = in_window and (trade_end - ts) <= s.flatten_min * 60
        # ---------------- fills and exits on bar i (decided at end of i-1)
        if in_window:
            for m in list(positions):
                a, pos = mk[m], positions[m]
                o, h, lo, c, traded = a["o"][i], a["h"][i], a["l"][i], a["c"][i], a["traded"][i]
                if flatten_all or end_flatten or ts >= trade_end:
                    close_pos(m, c * (1 - s.taker_slippage), s.taker_fee, "flatten", ts)
                elif traded and lo <= pos["stop"]:
                    close_pos(m, min(pos["stop"], o) * (1 - s.taker_slippage), s.taker_fee, "stop", ts)
                elif traded and h >= pos["target"] * (1 + s.fill_margin) and a["vol_tmn"][i] >= s.min_minute_vol_tmn:
                    close_pos(m, pos["target"], s.maker_fee, "target", ts)
                elif ts - pos["ts"] >= params.max_hold_minutes * 60:
                    close_pos(m, c * (1 - s.taker_slippage), s.taker_fee, "time", ts)
            for m, r in list(resting.items()):
                a = mk[m]
                if m in positions or not a["traded"][i] or a["vol_tmn"][i] < s.min_minute_vol_tmn:
                    continue
                if a["l"][i] <= r["level"] * (1 - s.fill_margin):
                    positions[m] = _open(r, r["level"], s.maker_fee, ts, "maker", params)
                    r["filled"] = True
            for m, r in list(snipers.items()):
                a = mk[m]
                if m in positions or not a["traded"][i] or a["vol_tmn"][i] < s.min_minute_vol_tmn:
                    continue
                if a["c"][i] <= r["level"]:
                    px = a["c"][i] * (1 + s.taker_slippage)
                    if len(positions) >= risk.max_open or cash - r["notional"] < r["cash_floor"]:
                        continue
                    cash -= r["notional"]
                    positions[m] = _open(r, px, s.taker_fee, ts, "taker", params)
        # release cash reserved by unfilled resting orders (they are re-placed below)
        for r in resting.values():
            if not r.get("filled") and not s.collect:
                cash += r["notional"]
        resting, snipers = {}, {}

        # ---------------- update statistics with the closed bar i
        for m in syms:
            a = mk[m]
            if math.isnan(a["c"][i]) or math.isnan(a["fair"][i]):
                continue
            states[m].on_bar(ts, a["c"][i], a["l"][i], a["h"][i], a["fair"][i], a["vol_tmn"][i], usdt[i],
                             traded=bool(a["traded"][i]))
        # exit targets follow the live normal-premium price (re-quoted like the live bot)
        for m, pos in positions.items():
            a, st = mk[m], states[m]
            pos["target"] = exit_target(st, a["fair"][i], pos["entry"], pos["buy_fee"], s.maker_fee, params)

        if not in_window:
            continue
        equity = cash + sum(p["qty"] * mk[m]["c"][i] for m, p in positions.items())
        eq_ts.append(ts)
        eq_val.append(equity)
        if not s.collect:
            risk.update_equity(equity, ts)
            if risk.flatten_requested:
                flatten_all = True
                risk.flatten_requested = False
            elif not positions:
                flatten_all = False
        if ts >= trade_end - s.entry_cutoff_min * 60 or i + 1 >= len(grid):
            continue
        ok, why = risk.trading_allowed(ts) if not s.collect else (True, "")
        if not ok:
            blocked[why[:20]] = blocked.get(why[:20], 0) + 1
            continue

        # ---------------- candidates for minute i+1
        cands = []
        for m in syms:
            if m in positions:
                continue
            st, a = states[m], mk[m]
            g = gates(st, params)
            if g:
                blocked[g] = blocked.get(g, 0) + 1
                continue
            fair = a["fair"][i]
            if not fair > 0:
                continue
            e = entry_edge(st, s.maker_fee, s.taker_fee, params)
            level = entry_level(st, fair, e)
            dev_now = a["c"][i] / st.fair_adj(fair) - 1
            cands.append((m, e, level, st.features(ts, e, dev_now)))
        if not cands:
            continue
        probs = [None] * len(cands)
        if model is not None and model.ready:
            probs = list(model.predict(pd.DataFrame([c[3] for c in cands])[FEATURES]))
        scored = []
        for (m, e, level, f), p in zip(cands, probs):
            if p is not None and p < s.min_prob and not s.collect:
                blocked["ml_skip"] = blocked.get("ml_skip", 0) + 1
                continue
            score = (p if p is not None else f["revert_rate"]) * e
            scored.append((score, m, e, level, f, p))
        scored.sort(key=lambda x: -x[0])

        if s.collect:
            for _, m, e, level, f, p in scored:
                resting[m] = {"sym": m, "level": level, "notional": s.collect_notional, "feats": f, "prob": p, "edge": e}
            continue
        slots = max(0, min(s.resting_markets, risk.max_open - len(positions)))
        for k, (_, m, e, level, f, p) in enumerate(scored):
            notional = _size(risk, equity, level, params, s, p)
            if k < slots:
                free = cash - equity * risk.min_cash
                notional = min(notional, free)
                if notional < s.min_order_tmn:
                    continue
                cash -= notional
                resting[m] = {"sym": m, "level": level, "notional": notional, "feats": f, "prob": p, "edge": e}
            elif s.sniper:
                st = states[m]
                et = entry_edge(st, s.maker_fee, s.taker_fee, params, taker=True)
                lvl = entry_level(st, mk[m]["fair"][i], et)
                notional = min(notional, cash - equity * risk.min_cash)
                if notional >= s.min_order_tmn:
                    snipers[m] = {"sym": m, "level": lvl, "notional": notional, "feats": f, "prob": p, "edge": et,
                                  "cash_floor": equity * risk.min_cash}

    # close anything left at the very end (the challenge is scored in toman)
    if positions:
        i = len(grid) - 1
        for m in list(positions):
            close_pos(m, mk[m]["c"][i] * (1 - s.taker_slippage), s.taker_fee, "flatten", int(grid[i]))
        if eq_ts:
            eq_val[-1] = cash
    return SimResult(pd.DataFrame(trades), pd.Series(eq_val, index=pd.to_datetime(eq_ts, unit="s")),
                     pd.DataFrame(events), s, blocked)


def _open(r: dict, px: float, fee: float, ts: float, mode: str, params: Params) -> dict:
    qty = r["notional"] / px * (1 - fee)
    return {"qty": qty, "entry": px, "cost": r["notional"], "ts": ts, "buy_fee": fee, "mode": mode,
            "stop": stop_price(px, params), "target": px * 1.01, "feats": r["feats"], "prob": r["prob"],
            "edge": r["edge"]}


def _size(risk: RiskManager, equity: float, level: float, params: Params, s: SimSettings, p) -> float:
    base = equity * s.position_frac * size_from_prob(p, s.min_prob, s.full_size_prob)
    return min(base, risk.size(equity, level, stop_price(level, params)))


class _DefaultCfg:
    def get(self, key, default=None):
        return default


def run_backtest(prep: dict, params: Params, s: SimSettings, model=None, trade_start=None, trade_end=None,
                 risk_cfg=None) -> SimResult:
    return run(prep, params, s, model, trade_start, trade_end, risk_cfg)
