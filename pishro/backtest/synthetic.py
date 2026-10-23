"""Synthetic market data for *testing the code path only* (never for performance claims).

Generates a global reference (GBM), a USDT/TMN walk, and Wallex-like toman markets that
follow the reference with a lag, a slowly drifting premium and occasional local dumps
that mean-revert. Written in the same file format as ``fetch_history`` so the whole
pipeline (load -> simulate -> train -> report) can run without network access.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd


def generate(out_dir, days: float = 6, markets: int = 6, seed: int = 1, revert: float = 0.15) -> None:
    rng = np.random.default_rng(seed)
    n = int(days * 1440)
    t0 = 1_760_000_000 - 1_760_000_000 % 60
    ts = t0 + 60 * np.arange(n)
    usdt = 100_000 * np.exp(np.cumsum(rng.normal(0, 0.0004, n)))
    out_dir.mkdir(parents=True, exist_ok=True)
    _ohlc(ts, usdt, rng, 5e9).to_csv(out_dir / "wallex_USDTTMN.csv.gz", index=False)
    meta = {"USDTTMN": {"base": "USDT", "quote": "TMN", "vol": 1e12}}
    for k in range(markets):
        base = f"SYN{k}"
        ref = (10 ** rng.uniform(-1, 3)) * np.exp(np.cumsum(rng.normal(0, 0.0012, n)))
        prem = np.zeros(n)
        drift = 0.003 * rng.normal()
        for i in range(1, n):
            drift *= 0.9995                     # re-pricings fade over ~1 day
            prem[i] = prem[i - 1] + revert * (drift - prem[i - 1]) + rng.normal(0, 0.0008)
            if rng.random() < 0.004:           # local dump into a thin book
                shock = rng.uniform(0.006, 0.02)
                prem[i] -= shock
                # "informed" dumps re-price the market for good; more likely after the reference fell
                falling = i > 15 and ref[i] < ref[i - 15] * 0.997
                if rng.random() < (0.7 if falling else 0.25):
                    drift -= shock
        lag = rng.integers(1, 4)
        fair = ref * usdt
        lagged = np.concatenate([np.full(lag, fair[0]), fair[:-lag]])
        price = lagged * (1 + prem)
        _ohlc(ts, price, rng, 2e8).to_csv(out_dir / f"wallex_{base}TMN.csv.gz", index=False)
        pd.DataFrame({"ts": ts, "c": ref}).to_csv(out_dir / f"binance_{base}.csv.gz", index=False)
        meta[f"{base}TMN"] = {"base": base, "quote": "TMN", "vol": 5e10}
    (out_dir / "markets.json").write_text(json.dumps(meta))


def _ohlc(ts, close, rng, vol_scale) -> pd.DataFrame:
    noise = np.abs(rng.normal(0, 0.0012, len(close)))
    o = np.concatenate([[close[0]], close[:-1]])
    h = np.maximum(o, close) * (1 + noise)
    lo = np.minimum(o, close) * (1 - noise)
    v = rng.exponential(vol_scale, len(close)) / close
    v[rng.random(len(close)) < 0.15] = 0.0
    return pd.DataFrame({"ts": ts, "o": o, "h": h, "l": lo, "c": close, "v": v})
