"""Download and load 1-minute history for the backtest.

* Wallex ``/v1/udf/history`` (resolution 1) answers ``no_data`` for long ranges, so it is
  paged in 12-hour windows.
* Binance 1-minute klines come from the ``data-api.binance.vision`` mirror.
* Files: ``data/history/wallex_<SYMBOL>.csv.gz`` (ts,o,h,l,c,v) and
  ``data/history/binance_<BASE>.csv.gz`` (ts,c), plus ``markets.json``.

Note: Wallex UDF volume ``v`` is reported in an unconfirmed unit (it has been observed as
2× the summed trade quantity); we only use it for a coarse liquidity gate.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path

import pandas as pd

from ..exchange.binance import BinanceRef
from ..exchange.wallex import WallexClient
from ..market.universe import load_excluded

log = logging.getLogger(__name__)
PAGE = 43200


def history_dir(cfg) -> Path:
    d = cfg.data_dir / "history"
    d.mkdir(parents=True, exist_ok=True)
    return d


async def _fetch_wallex(client: WallexClient, symbol: str, frm: int, to: int, sem: asyncio.Semaphore) -> pd.DataFrame:
    rows = []
    for start in range(frm, to, PAGE):
        async with sem:
            try:
                d = await client.candles(symbol, "1", start, min(start + PAGE, to))
            except Exception as e:  # noqa: BLE001
                log.warning("udf %s %s: %s", symbol, start, e)
                continue
        rows += list(zip(d["t"], d["o"], d["h"], d["l"], d["c"], d["v"]))
    df = pd.DataFrame(rows, columns=["ts", "o", "h", "l", "c", "v"]).drop_duplicates("ts").sort_values("ts")
    return df


async def _fetch_binance(bn: BinanceRef, base: str, frm: int, to: int, sem: asyncio.Semaphore) -> pd.DataFrame:
    async with sem:
        k = await bn.klines(base, "1m", frm * 1000, to * 1000)
    return pd.DataFrame([(x[0] // 1000, float(x[4])) for x in k], columns=["ts", "c"]).drop_duplicates("ts")


async def fetch_history(cfg, days: float | None = None, max_markets: int | None = None) -> list[str]:
    days = float(days or cfg.get("backtest.days", 30))
    max_markets = int(max_markets or cfg.get("backtest.max_markets", 40))
    out = history_dir(cfg)
    client = WallexClient(cfg.get("exchange.base_url"))
    bn = BinanceRef(cfg.get("references.binance_rest"), fallback_rest=cfg.get("references.binance_rest_fallbacks", []))
    try:
        infos, quotes = await client.markets()
        (out / "markets.json").write_text(json.dumps(
            {s: {"base": i.base, "quote": i.quote, "vol": quotes[s].quote_volume_24h if s in quotes else 0,
                 "zero_fee": i.zero_fee, "tick": i.tick, "step": i.step, "min_notional": i.min_notional}
             for s, i in infos.items()}, ensure_ascii=False))
        excluded = load_excluded(cfg.get("universe.excluded_file"))
        try:
            binance_bases = await bn.load_symbols()
        except Exception as e:  # noqa: BLE001
            log.warning("binance exchangeInfo failed: %s", e)
            binance_bases = set()
        cands = [s for s, i in infos.items() if i.quote == "TMN" and s not in excluded and i.base != "USDT"]
        cands.sort(key=lambda s: -(quotes[s].quote_volume_24h if s in quotes else 0))
        chosen = cands[:max_markets]
        to = int(time.time()) // 60 * 60
        frm = to - int(days * 86400)
        sem = asyncio.Semaphore(4)
        wallex_syms = list(chosen) + ["USDTTMN"]
        for s in chosen:
            base = infos[s].base
            if base not in binance_bases and f"{base}USDT" in infos:
                wallex_syms.append(f"{base}USDT")

        async def wjob(sym):
            df = await _fetch_wallex(client, sym, frm, to, sem)
            if len(df):
                df.to_csv(out / f"wallex_{sym}.csv.gz", index=False)
            log.info("wallex %s: %d bars", sym, len(df))

        async def bjob(base):
            df = await _fetch_binance(bn, base, frm, to, sem)
            if len(df):
                df.to_csv(out / f"binance_{base}.csv.gz", index=False)
            log.info("binance %s: %d bars", base, len(df))

        await asyncio.gather(*[wjob(s) for s in wallex_syms],
                             *[bjob(infos[s].base) for s in chosen if infos[s].base in binance_bases])
        return chosen
    finally:
        await client.close()
        await bn.close()


def _read(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df.drop_duplicates("ts").sort_values("ts").set_index("ts") if len(df) else None


def load_history(cfg) -> dict:
    """-> {"usdt": df, "markets": {SYMBOL: {"wallex": df, "ref": Series(USDT price), "source": str}}, "meta": {...}}"""
    d = history_dir(cfg)
    usdt = _read(d / "wallex_USDTTMN.csv.gz")
    if usdt is None:
        raise FileNotFoundError(f"no history in {d}; run `python -m pishro fetch` first (from an Iranian IP)")
    meta = json.loads((d / "markets.json").read_text()) if (d / "markets.json").exists() else {}
    excluded = load_excluded(cfg.get("universe.excluded_file"))
    markets = {}
    for f in sorted(d.glob("wallex_*TMN.csv.gz")):
        sym = f.name[len("wallex_"):-len(".csv.gz")]
        if sym == "USDTTMN" or sym in excluded:
            continue
        base = meta.get(sym, {}).get("base") or sym[:-3]
        wx = _read(f)
        ref, source = None, ""
        b = _read(d / f"binance_{base}.csv.gz")
        if b is not None:
            ref, source = b["c"], "binance"
        else:
            u = _read(d / f"wallex_{base}USDT.csv.gz")
            if u is not None:
                ref, source = u["c"], "wallex_usdt"
        if wx is None or ref is None:
            continue
        markets[sym] = {"wallex": wx, "ref": ref, "source": source, "base": base}
    return {"usdt": usdt, "markets": markets, "meta": meta}
