"""Walk-forward training and evaluation of the trade-filter model.

1. *Collect*: simulate the strategy with unlimited capacity on the training window; every
   would-be fill becomes a labelled example (features at order time -> was it profitable
   after fees?).
2. *Fit*: calibrated LightGBM (time-ordered split + embargo inside ``ProbModel.fit``).
3. *Test*: run the real portfolio backtest on the later, unseen window with and without
   the model (ablation). The gap between the two is what the AI adds.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ..backtest import sim
from ..strategy.reversion import FEATURES, Params
from .model import ProbModel

log = logging.getLogger(__name__)


def model_path(cfg, zero_fee: bool) -> Path:
    base = cfg.resolve(cfg.get("ml.model_path", "models/reversion.pkl"))
    return base.with_name(base.stem + ("_zerofee" if zero_fee else "_fee") + base.suffix)


def collect_events(prep: dict, params: Params, settings: sim.SimSettings, start: int, end: int, risk_cfg) -> pd.DataFrame:
    s = sim.SimSettings(**{**settings.__dict__, "collect": True})
    return sim.run_backtest(prep, params, s, None, start, end, risk_cfg).events


def fit_model(events: pd.DataFrame, name: str) -> ProbModel:
    m = ProbModel(FEATURES, name=name)
    if len(events):
        m.fit(events.rename(columns={"ts": "ts"}), label="y", time_col="ts")
    return m


def walk_forward(cfg, prep: dict, params: Params, maker: float, taker: float) -> dict:
    grid = prep["grid"]
    t0, t1 = int(grid[0]), int(grid[-1])
    warm = int(params.warmup_minutes * 60 + 3600)
    split = int(t0 + (t1 - t0) * float(cfg.get("backtest.train_frac", 0.6)))
    settings = sim.SimSettings.from_cfg(cfg, maker, taker)
    events = collect_events(prep, params, settings, t0 + warm, split, cfg)
    model = fit_model(events, f"reversion fee={maker:.4f}")
    rules = sim.run_backtest(prep, params, settings, None, split, t1, cfg)
    with_ml = sim.run_backtest(prep, params, settings, model if model.ready else None, split, t1, cfg)
    full_rules = sim.run_backtest(prep, params, settings, None, t0 + warm, t1, cfg)
    return {"maker": maker, "taker": taker, "split_ts": split, "train_events": len(events),
            "model": model, "rules_test": rules, "ml_test": with_ml, "rules_full": full_rules}


def train_deploy_models(cfg, prep: dict, params: Params) -> dict[str, Path]:
    """Train on the full history for both fee regimes and save them for the live bot."""
    t0, t1 = int(prep["grid"][0]), int(prep["grid"][-1])
    warm = int(params.warmup_minutes * 60 + 3600)
    out = {}
    fee = (float(cfg.get("exchange.fallback_maker_fee", 0.0025)), float(cfg.get("exchange.fallback_taker_fee", 0.003)))
    for zero, (mk, tk) in ((False, fee), (True, (0.0, 0.0))):
        st = sim.SimSettings.from_cfg(cfg, mk, tk)
        ev = collect_events(prep, params, st, t0 + warm, t1, cfg)
        m = fit_model(ev, f"reversion deploy fee={mk:.4f}")
        if m.ready:
            path = model_path(cfg, zero)
            m.save(path)
            out["zero" if zero else "fee"] = path
            log.info("saved %s (%d events)", path, len(ev))
    return out
