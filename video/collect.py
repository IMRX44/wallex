"""Collect everything the video shows into one JSON (real data or demo data).

Sources inside ``data_dir``: pishro.db / pishro_paper.db (trades, equity, quotes, agent
reviews), decisions*.jsonl, reports/backtest.json, reports/trades_*.csv, models/*.json.
"""
from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def _q(db: Path, sql: str, args=()) -> list[dict]:
    if not db.exists():
        return []
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in con.execute(sql, args).fetchall()]
    except sqlite3.OperationalError:
        return []
    finally:
        con.close()


def _down(rows: list, n: int) -> list:
    if len(rows) <= n:
        return rows
    step = len(rows) / n
    return [rows[int(i * step)] for i in range(n)] + [rows[-1]]


def _ewm_baseline(dev: np.ndarray, halflife: float = 30) -> np.ndarray:
    a = 1 - 0.5 ** (1 / halflife)
    out = np.empty_like(dev)
    m = dev[0]
    for i, x in enumerate(dev):
        m = m + a * (x - m)
        out[i] = m
    return out


def collect(data_dir: Path, demo: bool, capital: float = 5_000_000, live_db: str | None = None) -> dict:
    data_dir = Path(data_dir)
    db = data_dir / (live_db or ("pishro.db" if (data_dir / "pishro.db").exists() else "pishro_paper.db"))
    dec_path = data_dir / ("decisions.jsonl" if db.name == "pishro.db" else "decisions_paper.jsonl")
    out: dict = {"demo": demo, "capital": capital, "mode": "live" if db.name == "pishro.db" else "paper"}

    # ---------------- live / shadow run
    eq = _q(db, "SELECT ts, equity, drawdown FROM equity ORDER BY ts")
    trades = _q(db, "SELECT * FROM trades ORDER BY exit_ts")
    if eq:
        t0 = eq[0]["ts"]
        out["equity"] = [[(r["ts"] - t0) / 3600, r["equity"]] for r in _down(eq, 360)]
        vals = np.array([r["equity"] for r in eq])
        peak = np.maximum.accumulate(vals)
        out["max_dd_pct"] = float(((1 - vals / peak).max()) * 100)
        out["final_equity"] = float(vals[-1])
        out["hours"] = (eq[-1]["ts"] - t0) / 3600
    else:
        out["equity"], out["max_dd_pct"], out["final_equity"], out["hours"] = [], 0.0, capital, 0
    out["return_pct"] = (out["final_equity"] / capital - 1) * 100
    if trades:
        df = pd.DataFrame(trades)
        df["cost"] = df.entry_price * df.qty
        out["n_trades"] = len(df)
        out["win_rate"] = float((df.pnl_tmn > 0).mean() * 100)
        out["realized_pnl"] = float(df.pnl_tmn.sum())
        out["volume"] = float((df.entry_price * df.qty + df.exit_price * df.qty).sum())
        out["avg_trade"] = float(df.cost.mean())
        out["avg_ret"] = float(df.ret.mean() * 100)
        out["exit_reasons"] = df.reason_out.value_counts().to_dict()
        out["avg_hold_min"] = float(((df.exit_ts - df.entry_ts) / 60).mean())
        per = df.groupby("symbol").agg(n=("pnl_tmn", "size"), pnl=("pnl_tmn", "sum")).sort_values("pnl", ascending=False)
        out["per_market"] = [{"symbol": s, "n": int(r.n), "pnl": float(r.pnl)} for s, r in per.head(10).iterrows()]
        out["trades_recent"] = [{"symbol": r.symbol, "pnl": float(r.pnl_tmn), "ret": float(r.ret * 100),
                                 "reason": r.reason_out, "hold": float((r.exit_ts - r.entry_ts) / 60)}
                                for r in df.tail(12).itertuples()]
        if out["hours"]:
            out["trades_per_day"] = len(df) / max(out["hours"] / 24, 1e-9)
        losses = df[df.pnl_tmn <= 0]
        out["worst_trade_pct"] = float(df.ret.min() * 100)
        out["n_losses"] = len(losses)
    else:
        out.update({"n_trades": 0, "win_rate": 0, "realized_pnl": 0, "volume": 0, "avg_trade": 0, "avg_ret": 0,
                    "exit_reasons": {}, "per_market": [], "trades_recent": []})

    # ---------------- decision log
    decisions = []
    if dec_path.exists():
        with open(dec_path, encoding="utf-8") as f:
            decisions = [json.loads(x) for x in f if x.strip()]
    kinds: dict[str, int] = {}
    for d in decisions:
        kinds[d["kind"]] = kinds.get(d["kind"], 0) + 1
    out["decision_counts"] = kinds
    out["n_decisions"] = len(decisions)
    buys = [d for d in decisions if d["kind"] == "buy_order" and d.get("reason")]
    sells = [d for d in decisions if d["kind"] == "sell_fill"]
    out["log_samples"] = [{"kind": d["kind"], "symbol": d["symbol"], "time": d["time_tehran"][11:19],
                           "reason": d["reason"][:220]} for d in _down(buys, 4)[:4] + sells[-4:]]
    out["risk_events"] = [{"kind": d["kind"], "reason": d["reason"][:160]} for d in decisions
                          if d["kind"] in ("fee_regime", "cancel_buy") and "ریسک" in d.get("reason", "")][:4]
    probs = [d.get("prob") for d in buys if d.get("prob") is not None]
    out["prob_hist"] = np.histogram(probs, bins=10, range=(0.5, 1.0))[0].tolist() if probs else []

    # ---------------- one trade explained with its price path
    out["example"] = _example_trade(db, decisions)

    # ---------------- agent
    reviews = _q(db, "SELECT ts, model, input, output, rationale FROM decisions WHERE kind='agent_review' ORDER BY ts DESC LIMIT 6")
    out["agent"] = []
    for r in reviews:
        try:
            inp, outp = json.loads(r["input"]), json.loads(r["output"])
        except (TypeError, ValueError):
            continue
        out["agent"].append({"rationale": r["rationale"], "model": r["model"],
                             "tools": [t.get("tool") for t in (inp.get("transcript") or []) if t.get("tool")],
                             "actions": [{k: v for k, v in a.items() if k in ("tool", "symbol", "mode", "allow", "reason", "size_mult")}
                                         for a in (outp.get("actions") or [])]})

    # ---------------- backtest + ML
    rep = data_dir / "reports" / "backtest.json"
    if rep.exists():
        bt = json.loads(rep.read_text())
        out["backtest_meta"] = bt.get("meta", {}).get("subtitle", "")
        out["backtest"] = []
        for r in bt.get("results", []):
            row = {"maker": r["maker"], "taker": r["taker"],
                   "rules": {k: r["rules_test"].get(k) for k in ("return_pct", "max_drawdown_pct", "trades", "win_rate_pct", "profit_factor")},
                   "ml": {k: r["ml_test"].get(k) for k in ("return_pct", "max_drawdown_pct", "trades", "win_rate_pct", "profit_factor")},
                   "mc": r.get("mc") or {}, "model": {k: (r.get("model") or {}).get(k) for k in ("auc", "brier", "base_rate", "n_train")},
                   "importance": list(((r.get("model") or {}).get("feature_importance") or {}).items())[:8],
                   "curves": {k: [[t, v] for t, v in (r.get("curves") or {}).get(k, [])] for k in ("rules_test", "ml_test")}}
            fee_tag = f"fee{r['maker']}"
            tpath = data_dir / "reports" / f"trades_ml_test_{fee_tag}.csv"
            if not tpath.exists():
                tpath = data_dir / "reports" / f"trades_rules_test_{fee_tag}.csv"
            row["mc_hist"] = _mc_hist(tpath, capital, r["ml_test"].get("days") or r["rules_test"].get("days") or 1)
            out["backtest"].append(row)
    return out


def _mc_hist(path: Path, capital: float, days: float, sims: int = 20000) -> dict:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if len(df) < 5:
        return {}
    rng = np.random.default_rng(7)
    pnl = df.pnl.to_numpy() / capital * 100
    lam = len(df) / max(days, 1e-9) * 3
    res = np.array([pnl[rng.integers(0, len(pnl), c)].sum() if c else 0.0 for c in rng.poisson(lam, sims)])
    lo, hi = np.percentile(res, 0.5), np.percentile(res, 99.5)
    h, edges = np.histogram(res, bins=36, range=(lo, hi))
    return {"counts": h.tolist(), "edges": [float(e) for e in edges], "p_pos": float((res > 0).mean() * 100),
            "median": float(np.median(res)), "p5": float(np.percentile(res, 5)), "p95": float(np.percentile(res, 95))}


def _example_trade(db: Path, decisions: list[dict]) -> dict:
    """Pick a clean winning maker trade and return its price path (Wallex mid, fair, normal-premium line)."""
    fills = {d.get("position_id"): d for d in decisions if d["kind"] == "buy_fill"}
    orders = {d.get("client_id"): d for d in decisions if d["kind"] == "buy_order"}
    best = None
    for s in decisions:
        if s["kind"] != "sell_fill" or s.get("exit_kind") != "target" or (s.get("pnl_tmn") or 0) <= 0:
            continue
        b = fills.get(s.get("position_id"))
        if not b:
            continue
        o = orders.get(b.get("client_id"))
        if not o:
            continue
        hold = s["ts"] - b["ts"]
        if 300 < hold < 7200 and (best is None or s["pnl_tmn"] > best[2]["pnl_tmn"]):
            best = (o, b, s)
    if best is None:
        return {}
    o, b, s = best
    sym = o["symbol"]
    rows = _q(db, "SELECT ts, bid, ask, fair FROM quotes WHERE symbol=? AND ts BETWEEN ? AND ? ORDER BY ts",
              (sym, b["ts"] - 3 * 3600, s["ts"] + 1800))
    if len(rows) < 30:
        return {}
    df = pd.DataFrame(rows).dropna()
    mid = ((df.bid + df.ask) / 2).to_numpy()
    fair = df.fair.to_numpy()
    base = _ewm_baseline(mid / fair - 1)
    view = df.ts >= b["ts"] - 90 * 60
    t0 = float(df.ts[view].iloc[0])
    path = [[(t - t0) / 60, m, f, f * (1 + bl)] for t, m, f, bl, v in zip(df.ts, mid, fair, base, view) if v]
    return {"symbol": sym, "path": _down(path, 160), "buy_t": (b["ts"] - t0) / 60, "buy_px": b.get("price"),
            "sell_t": (s["ts"] - t0) / 60, "sell_px": s.get("price"), "pnl": s.get("pnl_tmn"), "ret": (s.get("ret") or 0) * 100,
            "edge": (o.get("edge") or 0) * 100, "prob": o.get("prob"), "baseline": (o.get("baseline") or 0) * 100,
            "std": (o.get("std") or 0) * 100, "reason_buy": o.get("reason", "")[:260], "reason_sell": s.get("reason", "")[:200],
            "level": o.get("level"), "fair": o.get("fair"), "features": {k: o.get("features", {}).get(k) for k in
                                                                         ("ref_ret_15", "revert_rate", "std_ratio", "mkt_ret_15")}}


def safe(x):
    if isinstance(x, float):
        return None if (math.isnan(x) or math.isinf(x)) else round(x, 6)
    if isinstance(x, dict):
        return {str(k): safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [safe(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return safe(float(x))
    return x
