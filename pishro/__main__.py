"""Command line.

  python -m pishro check        # connectivity: Wallex, Binance, Nobitex, account, fees, LLM
  python -m pishro serve        # bot + dashboard (PISHRO_MODE=paper|live)
  python -m pishro fetch        # download 1-minute history (Wallex + Binance) for the backtest
  python -m pishro backtest     # walk-forward backtest, ablation, HTML report, train deploy models
  python -m pishro demo         # same pipeline on synthetic data (offline smoke test only)
  python -m pishro export       # copy decision log / trades / equity to ./deliverables
  python -m pishro mcp          # MCP server exposing the bot's tools to external agents
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import sys
import time

from .config import ROOT, load_config


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpcore", "openai", "websockets"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


async def _check(cfg) -> int:
    from .exchange.binance import BinanceRef
    from .exchange.nobitex import NobitexClient
    from .exchange.wallex import WallexClient
    from .market.universe import Universe, load_excluded

    ok = True
    wx = WallexClient(cfg.get("exchange.base_url"), cfg.secrets.wallex_api_key)
    bn = BinanceRef(cfg.get("references.binance_rest"), fallback_rest=cfg.get("references.binance_rest_fallbacks", []))
    nb = NobitexClient(cfg.get("references.nobitex_base_url"))

    def line(name, good, detail=""):
        print(f"{'OK ' if good else 'ERR'}  {name:<28} {detail}")

    try:
        t0 = time.time()
        infos, quotes = await wx.markets()
        uni = Universe(load_excluded(cfg.get("universe.excluded_file")), ["TMN"],
                       float(cfg.get("universe.min_volume_24h_tmn", 0)), float(cfg.get("universe.max_spread_pct", 0.012)))
        usdt = quotes.get("USDTTMN")
        tr = uni.refresh(infos, quotes, usdt.mid if usdt else 0)
        zero = sorted(s for s, i in infos.items() if i.zero_fee)
        line("Wallex /v1/markets", True, f"{len(infos)} markets, {len(tr)} tradable toman markets, "
             f"{(time.time() - t0) * 1000:.0f} ms, zero-fee: {', '.join(zero[:12]) or '-'}")
        print("     tradable:", ", ".join(sorted(tr)))
    except Exception as e:  # noqa: BLE001
        ok = False
        line("Wallex /v1/markets", False, repr(e))
    try:
        mids = await bn.poll()
        line("Binance (binance.vision)", True, f"{len(mids)} USDT pairs")
    except Exception as e:  # noqa: BLE001
        ok = False
        line("Binance (binance.vision)", False, repr(e))
    try:
        m = await nb.mids()
        line("Nobitex (optional)", True, f"{len(m)} coins")
    except Exception as e:  # noqa: BLE001
        line("Nobitex (optional)", False, repr(e))
    if cfg.secrets.wallex_api_key:
        try:
            b = await wx.balances()
            line("Wallex account balances", True, json.dumps({k: v for k, v in b.items() if v[0] > 0}, ensure_ascii=False))
            f = await wx.fees(float(cfg.get("exchange.fallback_maker_fee")), float(cfg.get("exchange.fallback_taker_fee")))
            sample = {k: f[k] for k in list(f)[:3]}
            line("Wallex /v1/account/fee", True, f"{len(f)} markets, e.g. {sample}")
        except Exception as e:  # noqa: BLE001
            ok = False
            line("Wallex account (API key)", False, repr(e))
    else:
        line("Wallex account (API key)", False, "WALLEX_API_KEY not set (needed for live mode only)")
    if cfg.ai_enabled:
        from .ai.llm import LLMClient
        try:
            llm = LLMClient(cfg.secrets.llm_base_url, cfg.secrets.llm_api_key, cfg.get("ai.model"), 60, 50)
            t0 = time.time()
            msg = await llm.chat([{"role": "user", "content": "Reply with the single word: ready"}])
            line(f"LLM {cfg.get('ai.model')}", True, f"{(msg.content or '').strip()[:30]} ({time.time() - t0:.1f}s)")
        except Exception as e:  # noqa: BLE001
            ok = False
            line(f"LLM {cfg.get('ai.model')}", False, repr(e)[:200])
    else:
        line("LLM", False, "LLM_BASE_URL / LLM_API_KEY not set (agent disabled)")
    for c in (wx, bn, nb):
        await c.close()
    return 0 if ok else 1


def _serve(cfg, host: str, port: int) -> None:
    import uvicorn

    from .web.app import create_app
    uvicorn.run(create_app(cfg), host=host, port=port, log_level="warning")


def _export(cfg) -> None:
    out = ROOT / "deliverables"
    out.mkdir(exist_ok=True)
    for name in ("decisions.jsonl", "decisions_paper.jsonl", "pishro.db", "pishro_paper.db"):
        p = cfg.data_dir / name
        if p.exists():
            shutil.copy2(p, out / name)
            print("copied", out / name)
    from .store import Store
    for db in ("pishro.db", "pishro_paper.db"):
        p = cfg.data_dir / db
        if not p.exists():
            continue
        st = Store(p)
        import csv
        for table in ("trades", "equity", "orders"):
            rows = st.query(f"SELECT * FROM {table}")
            if rows:
                path = out / f"{db.split('.')[0]}_{table}.csv"
                with open(path, "w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0]))
                    w.writeheader()
                    w.writerows(rows)
                print("wrote", path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pishro", description="Pishro — Wallex premium-reversion trading bot")
    ap.add_argument("--config", default=None)
    ap.add_argument("--log", default=os.environ.get("LOG_LEVEL", "INFO"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    s = sub.add_parser("serve")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    f = sub.add_parser("fetch")
    f.add_argument("--days", type=float, default=None)
    f.add_argument("--markets", type=int, default=None)
    b = sub.add_parser("backtest")
    b.add_argument("--no-deploy-models", action="store_true")
    d = sub.add_parser("demo")
    d.add_argument("--days", type=float, default=21)
    sub.add_parser("export")
    m = sub.add_parser("mcp")
    m.add_argument("--url", default=os.environ.get("PISHRO_URL", "http://127.0.0.1:8000"))
    a = ap.parse_args(argv)
    _setup_logging(a.log)
    cfg = load_config(a.config)

    if a.cmd == "check":
        return asyncio.run(_check(cfg))
    if a.cmd == "serve":
        _serve(cfg, a.host, a.port)
        return 0
    if a.cmd == "fetch":
        from .backtest.data import fetch_history
        chosen = asyncio.run(fetch_history(cfg, a.days, a.markets))
        print(f"fetched {len(chosen)} markets into {cfg.data_dir / 'history'}")
        return 0
    if a.cmd == "backtest":
        from .backtest.runner import run_all
        res = run_all(cfg, train_deploy=not a.no_deploy_models)
        print(json.dumps(res, ensure_ascii=False, indent=2, default=str))
        return 0
    if a.cmd == "demo":
        from .backtest import synthetic
        from .backtest.runner import run_all
        from .config import load_config as _lc
        demo_dir = ROOT / "data_demo"
        dcfg = _lc(a.config, overrides={"data_dir": str(demo_dir)})
        synthetic.generate(dcfg.data_dir / "history", days=a.days, markets=12)
        res = run_all(dcfg, train_deploy=False, subtitle="⚠️ داده‌ی مصنوعی — فقط برای آزمایش اجرای کد، نه ادعای سودآوری")
        print(json.dumps(res, ensure_ascii=False, indent=2, default=str))
        return 0
    if a.cmd == "export":
        _export(cfg)
        return 0
    if a.cmd == "mcp":
        from .mcp_server import run_mcp
        run_mcp(a.url)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
