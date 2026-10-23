"""`python -m pishro tune`: walk-forward parameter search that is hard to overfit.

* Only markets the live bot can trade (24h volume and spread filters of the live universe).
* The history is split in time: parameters are *chosen* on the first ``select_frac`` and
  *judged* on the rest (the holdout is never seen during selection).
* Objective: return% / (1 + max drawdown%), with drawdown <= ``max_dd_pct`` and at least
  ``min_trades`` trades; anything else scores -inf.
* Robustness: a configuration is ranked by the mean score of itself and its nearest
  neighbours in normalised parameter space, so a lonely spike loses to a stable plateau.
* The defaults change only if the chosen set beats them on the holdout by ``min_gain``
  (relative) without a worse drawdown; rolling 5-day windows show whether it is luck.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import math
import os
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from ..market.universe import load_excluded
from ..strategy.reversion import Params
from . import data, report, sim

log = logging.getLogger(__name__)

# (low, high, kind) — stop_pct and position_frac stay inside the hard risk limits (<= 2.5%, <= 25%)
SPACE: dict[str, tuple[float, float, str]] = {
    "edge_vol_mult": (1.8, 3.2, "p"),
    "edge_floor_zero_fee": (0.003, 0.006, "p"),
    "edge_floor_fee_mult": (1.2, 2.0, "p"),
    "exit_edge": (0.0, 0.003, "p"),
    "stop_pct": (0.015, 0.025, "p"),
    "max_hold_minutes": (120, 360, "p"),
    "resting_markets": (3, 6, "s"),
    "position_frac": (0.15, 0.25, "s"),
}


def _round(k: str, v: float) -> float:
    if k in ("resting_markets", "max_hold_minutes"):
        return int(round(v))
    return round(float(v), 5)


def defaults(cfg) -> dict:
    p = Params.from_cfg(cfg)
    s = cfg.get("strategy", {}) or {}
    out = {k: getattr(p, k) for k, (_, _, kind) in SPACE.items() if kind == "p"}
    out["resting_markets"] = int(s.get("resting_markets", 4))
    out["position_frac"] = float(s.get("position_frac", 0.2))
    return out


def sample(n: int, seed: int = 7) -> list[dict]:
    """Latin-hypercube style sample: every parameter range is covered evenly."""
    rng = random.Random(seed)
    cols = {}
    for k, (lo, hi, _) in SPACE.items():
        cuts = [(i + rng.random()) / n for i in range(n)]
        rng.shuffle(cuts)
        cols[k] = [_round(k, lo + c * (hi - lo)) for c in cuts]
    return [{k: cols[k][i] for k in SPACE} for i in range(n)]


def score(m: dict, max_dd_pct: float, min_trades: int) -> float:
    if m["trades"] < min_trades or m["max_drawdown_pct"] > max_dd_pct:
        return float("-inf")
    return m["return_pct"] / (1 + m["max_drawdown_pct"])


def robust_scores(configs: list[dict], scores: list[float], k: int = 3) -> list[float]:
    """Mean score of each config and its k nearest neighbours (parameters scaled to [0, 1])."""
    X = np.array([[(c[n] - SPACE[n][0]) / (SPACE[n][1] - SPACE[n][0]) for n in SPACE] for c in configs])
    s = np.array(scores, dtype=float)
    out = []
    for i in range(len(configs)):
        d = np.sqrt(((X - X[i]) ** 2).sum(axis=1))
        idx = np.argsort(d)[: k + 1]
        vals = s[idx]
        out.append(float("-inf") if np.isneginf(vals).any() else float(vals.mean()))
    return out


def live_markets(cfg, hist: dict) -> list[str]:
    """Markets the live universe would trade (same volume / spread / exclusion rules)."""
    meta = hist.get("meta", {})
    if not meta:                       # e.g. synthetic data: no volume information
        return sorted(hist["markets"])
    usdt = float(hist["usdt"]["c"].iloc[-1])
    min_vol = float(cfg.get("universe.min_volume_24h_tmn", 3e9))
    excluded = load_excluded(cfg.get("universe.excluded_file"))
    keep = []
    for sym in hist["markets"]:
        m = meta.get(sym, {})
        vol = float(m.get("vol", 0)) * (usdt if m.get("quote") == "USDT" else 1.0)
        if sym not in excluded and vol >= min_vol:
            keep.append(sym)
    return sorted(keep)


# ------------------------------------------------------------------ worker
_PREP: dict | None = None
_CFG = None


def _init(prep: dict, cfg) -> None:
    global _PREP, _CFG
    _PREP, _CFG = prep, cfg


def _settings(cfg, conf: dict, maker: float, taker: float, capital: float) -> tuple[Params, sim.SimSettings]:
    base = Params.from_cfg(cfg)
    params = dataclasses.replace(base, **{k: v for k, v in conf.items() if SPACE[k][2] == "p"})
    s = sim.SimSettings.from_cfg(cfg, maker, taker)
    s.resting_markets, s.position_frac, s.capital = int(conf["resting_markets"]), float(conf["position_frac"]), capital
    return params, s


def evaluate(job: tuple) -> dict:
    conf, maker, taker, start, end, capital, exclude = job
    prep = _PREP
    if exclude:
        prep = dict(prep, markets={k: v for k, v in prep["markets"].items() if k not in exclude})
    params, s = _settings(_CFG, conf, maker, taker, capital)
    res = sim.run(prep, params, s, model=None, trade_start=start, trade_end=end, risk_cfg=_CFG)
    m = report.metrics(res.trades, res.equity, capital)
    keys = ("return_pct", "max_drawdown_pct", "trades", "win_rate_pct", "profit_factor", "days")
    t = res.trades
    return {"conf": conf, "metrics": {k: (m.get(k) if m.get(k) is not None else 0.0) for k in keys},
            "trades": {"exit_ts": [float(x) for x in t["exit_ts"]], "pnl": [float(x) for x in t["pnl"]],
                       "symbol": list(t["symbol"])} if len(t) else {}}


def rolling(trades: dict, start: int, end: int, capital: float, days: float = 5) -> list[dict]:
    """Realised P&L per ``days``-day window (by exit time) — is the result spread out or one lucky week?"""
    if not trades:
        return []
    df = pd.DataFrame(trades)
    out, t = [], start
    step = int(days * 86400)
    while t < end:
        w = df[(df.exit_ts >= t) & (df.exit_ts < t + step)]
        out.append({"from": int(t), "days": round(min(step, end - t) / 86400, 1),
                    "ret_pct": round(float(w.pnl.sum()) / capital * 100, 2), "trades": int(len(w))})
        t += step
    return out


# ------------------------------------------------------------------ driver
def run_tune(cfg, samples: int = 36, workers: int | None = None, select_frac: float = 0.7,
             max_dd_pct: float = 3.0, min_trades: int = 20, min_gain: float = 0.15,
             capital: float | None = None, exclude: tuple[str, ...] = (), seed: int = 7,
             fee_scenarios: list | None = None) -> dict:
    hist = data.load_history(cfg)
    markets = live_markets(cfg, hist)
    hist = dict(hist, markets={k: v for k, v in hist["markets"].items() if k in markets})
    prep = sim.prepare(hist)
    grid = prep["grid"]
    t0, t1 = int(grid[0]), int(grid[-1])
    split = int(t0 + (t1 - t0) * select_frac)
    capital = float(capital or cfg.get("initial_capital_tmn", 5e6))
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    base = defaults(cfg)
    configs = [base] + sample(samples, seed)
    fee_scenarios = fee_scenarios or cfg.get("backtest.fee_scenarios", [[0.0025, 0.003], [0.0, 0.0]])
    log.info("tune: %d live-eligible markets, %.1f days (select %.1f / holdout %.1f), %d configs x %d fee scenarios",
             len(prep["markets"]), (t1 - t0) / 86400, (split - t0) / 86400, (t1 - split) / 86400,
             len(configs), len(fee_scenarios))
    out = {"markets": sorted(prep["markets"]), "period": [t0, t1], "split": split, "capital": capital,
           "space": {k: v[:2] for k, v in SPACE.items()}, "excluded": list(exclude), "scenarios": []}
    with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(prep, cfg)) as ex:
        for maker, taker in fee_scenarios:
            maker, taker = float(maker), float(taker)
            sel = list(ex.map(evaluate, [(c, maker, taker, t0, split, capital, exclude) for c in configs]))
            sc = [score(r["metrics"], max_dd_pct, min_trades) for r in sel]
            rob = robust_scores(configs, sc)
            best_i = int(np.argmax(rob)) if any(math.isfinite(x) for x in rob) else 0
            chosen = configs[best_i]
            hold = list(ex.map(evaluate, [(c, maker, taker, split, t1, capital, exclude) for c in (base, chosen)]))
            full = list(ex.map(evaluate, [(c, maker, taker, t0, t1, capital, exclude) for c in (base, chosen)]))
            hb, hc = hold[0]["metrics"], hold[1]["metrics"]
            gain = (hc["return_pct"] - hb["return_pct"]) / max(abs(hb["return_pct"]), 0.5)
            adopt = (best_i != 0 and gain >= min_gain and hc["max_drawdown_pct"] <= max(hb["max_drawdown_pct"], max_dd_pct)
                     and hc["trades"] >= min_trades // 2)
            scen = {
                "maker": maker, "taker": taker,
                "selection": [{"conf": c, "metrics": r["metrics"], "score": s, "robust": rb}
                              for c, r, s, rb in zip(configs, sel, sc, rob)],
                "chosen_index": best_i, "chosen": chosen, "default": base,
                "holdout": {"default": hb, "chosen": hc, "relative_gain": round(gain, 3)},
                "full": {"default": full[0]["metrics"], "chosen": full[1]["metrics"]},
                "rolling": {"default": rolling(full[0]["trades"], t0, t1, capital),
                            "chosen": rolling(full[1]["trades"], t0, t1, capital)},
                "adopt": adopt,
                "verdict": ("chosen set beats the defaults on the unseen holdout by "
                            f"{gain:+.0%} -> adopt" if adopt else
                            f"holdout gain {gain:+.0%} < {min_gain:.0%} or worse drawdown -> keep defaults"),
            }
            log.info("fee %.4f: chosen #%d holdout %+.2f%% (dd %.2f%%) vs default %+.2f%% (dd %.2f%%) -> %s",
                     maker, best_i, hc["return_pct"], hc["max_drawdown_pct"], hb["return_pct"],
                     hb["max_drawdown_pct"], "ADOPT" if adopt else "keep")
            out["scenarios"].append(scen)
    return out


def write(out: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float))
    return path
