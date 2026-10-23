"""`python -m pishro backtest`: walk-forward backtest + ablation + report + deploy models."""
from __future__ import annotations

import datetime as dt
import logging

from ..ml.train import train_deploy_models, walk_forward
from ..strategy.reversion import Params
from . import data, report, sim

log = logging.getLogger(__name__)


def run_all(cfg, train_deploy: bool = True, subtitle: str = "") -> dict:
    hist = data.load_history(cfg)
    prep = sim.prepare(hist)
    params = Params.from_cfg(cfg)
    capital = float(cfg.get("initial_capital_tmn", 5e6))
    out_dir = cfg.data_dir.parent / "reports" if cfg.data_dir.name == "data" else cfg.data_dir / "reports"
    results = []
    for maker, taker in cfg.get("backtest.fee_scenarios", [[0.0025, 0.003], [0.0, 0.0]]):
        log.info("walk-forward with fees maker=%s taker=%s", maker, taker)
        wf = walk_forward(cfg, prep, params, float(maker), float(taker))
        r = {"maker": float(maker), "taker": float(taker), "capital": capital, "train_events": wf["train_events"]}
        for key in ("rules_test", "ml_test", "rules_full"):
            res = wf[key]
            r[f"m_{key}"] = report.metrics(res.trades, res.equity, capital)
            r[f"eq_{key}"] = res.equity
            r[f"mc_{key}"] = report.monte_carlo_3day(res.trades, capital, r[f"m_{key}"]["days"])
            if len(res.trades):
                out_dir.mkdir(parents=True, exist_ok=True)
                res.trades.to_csv(out_dir / f"trades_{key}_fee{maker}.csv", index=False)
        r["model_report"] = wf["model"].report.to_dict() if wf["model"].ready else {}
        results.append(r)
    grid = prep["grid"]
    meta = {
        "subtitle": subtitle or (f"{len(prep['markets'])} بازار تومانی، "
                                 f"{dt.datetime.utcfromtimestamp(int(grid[0])):%Y-%m-%d} تا "
                                 f"{dt.datetime.utcfromtimestamp(int(grid[-1])):%Y-%m-%d} (UTC)، "
                                 f"آموزش {int(float(cfg.get('backtest.train_frac', 0.6)) * 100)}٪ اول / تست بقیه"),
        "markets": sorted(prep["markets"]), "params": params.__dict__,
    }
    path = report.write_report(results, out_dir, meta)
    log.info("report: %s", path)
    models = train_deploy_models(cfg, prep, params) if train_deploy else {}
    return {"report": str(path), "models": {k: str(v) for k, v in models.items()},
            "summary": [{k: v for k, v in r.items() if k.startswith("m_")} | {"maker": r["maker"]} for r in results]}
