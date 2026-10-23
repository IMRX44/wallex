"""Produce *demo* data for the video when no real trading data exists yet.

Everything here runs the real Pishro code on SYNTHETIC market data:
  1. synthetic history (27 days, 12 markets); the first 24 days feed the walk-forward
     backtest + ML training (``pishro.backtest.runner``)
  2. the real paper Bot is driven through the last 3 days (a simulated 3-day challenge)
     with fake Wallex/Binance feeds and a fake clock; ML model enabled
  3. the real agent code path runs with a *scripted* stand-in for the LLM (tool results
     are real, the model's choices are simulated - the video says so)
  4. screenshots of the real dashboard and backtest report

The video labels all of it as demo data. For the real video run ``python -m video.render
--data data`` after the challenge.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import json
import logging
import shutil
import sys
import threading
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pishro.backtest import data as bdata  # noqa: E402
from pishro.backtest import runner, synthetic  # noqa: E402
from pishro.bot import Bot  # noqa: E402
from pishro.config import load_config  # noqa: E402
from tests.fakes import Clock, FakeBinance, FakeWallex, install  # noqa: E402

log = logging.getLogger("demo")
BUILD = ROOT / "video" / "build"
DEMO = BUILD / "demo_data"


class ScriptedLLM:
    """Stand-in for Claude in the offline demo: reads the real tool results and applies simple rules."""
    tools_supported = True

    def __init__(self):
        self.step = 0
        self.calls = 0

    async def chat(self, messages, tools=None, temperature=0.2):
        def call(i, name, args):
            return types.SimpleNamespace(id=f"c{self.calls}_{i}", function=types.SimpleNamespace(
                name=name, arguments=json.dumps(args)))
        self.calls += 1
        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        if len(tool_msgs) == 0:
            calls = [call(1, "get_portfolio_status", {}), call(2, "get_market_overview", {})]
        elif len(tool_msgs) == 2:
            calls = [call(3, "get_opportunities", {"limit": 10}), call(4, "get_recent_trades", {"limit": 10})]
        elif len(tool_msgs) == 4:
            status = json.loads(tool_msgs[0]["content"])
            opps = json.loads(tool_msgs[2]["content"])
            calls, n = [], 5
            weak = [o for o in opps if o.get("revert_rate", 1) < 0.5][:2]
            for o in weak:
                calls.append(call(n, "set_market_policy", {"symbol": o["symbol"], "allow": False,
                                                           "reason": f"نرخ برگشت افت‌ها {o['revert_rate']:.0%}؛ ریسک ادامه‌ی افت"}))
                n += 1
            dd = status.get("drawdown_pct", 0)
            mode = "defensive" if dd > 2 else "normal"
            calls.append(call(n, "set_risk_mode", {"mode": mode, "size_mult": 0.7 if mode == "defensive" else 1.0,
                                                   "reason": f"افت فعلی {dd:.1f}٪"}))
            summary = (f"ارزش حساب {status.get('equity_tmn', 0):,.0f} تومان، افت از سقف {dd:.1f}٪. "
                       f"{len(opps)} فرصت فعال؛ "
                       + (f"{'، '.join(o['symbol'] for o in weak)} به‌دلیل برنگشتن افت‌ها مسدود شد؛ " if weak else "")
                       + f"حالت ریسک: {mode}.")
            calls.append(call(n + 1, "finish", {"summary_fa": summary, "outlook": "neutral" if mode == "normal" else "risky"}))
        else:
            calls = []
        return types.SimpleNamespace(content="", tool_calls=calls)


def build_history() -> tuple[dict, int, int]:
    tmp = BUILD / "synthetic_full"
    shutil.rmtree(tmp, ignore_errors=True)
    synthetic.generate(tmp, days=27, markets=12, seed=11)
    cfg_tmp = load_config(overrides={"data_dir": str(tmp.parent / "synthetic_full_cfg")})
    (cfg_tmp.data_dir / "history").mkdir(parents=True, exist_ok=True)
    for f in tmp.iterdir():
        shutil.copy2(f, cfg_tmp.data_dir / "history" / f.name)
    full = bdata.load_history(cfg_tmp)
    t0 = int(full["usdt"].index[0])
    cut = t0 + 24 * 86400
    hist_dir = DEMO / "history"
    hist_dir.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    for f in tmp.iterdir():
        if f.suffix == ".gz":
            df = pd.read_csv(f)
            df[df.ts < cut].to_csv(hist_dir / f.name, index=False)
        else:
            shutil.copy2(f, hist_dir / f.name)
    return full, cut, t0


async def drive_bot(cfg, full: dict, start: int, hours: float, clock: Clock) -> Bot:
    bot = Bot(cfg)
    wx, bn = FakeWallex(full, clock), FakeBinance(full, clock)
    install(bot, wx, bn)
    await bot.market.warmup(days=2)
    await bot.broker.start()
    bot.status_text = "running"
    if bot.agent is None:
        from pishro.ai.agent import TradingAgent
        bot.agent = TradingAgent(cfg, ScriptedLLM(), bot)
    minutes = int(hours * 60)
    for m in range(minutes):
        for ph in range(4):
            wx.phase = ph
            clock.t += 15
            await bot.market.refresh_markets()
            await bot.market.refresh_refs()
            bot.market.update()
            await bot._on_tick()
        if m % 15 == 0:
            acc = bot.engine.account
            bot.store.add_equity(acc.equity, acc.free_tmn, 0.0, acc.exposure, bot.risk.drawdown)
            bot.equity_curve.append((clock.t, acc.equity))
        if m % 240 == 30:
            await bot.agent.run_once("scheduled review" if m else "start")
        if m == int(minutes * 0.62):          # live-looking dashboard: positions, orders, opportunities
            await asyncio.to_thread(screenshots, cfg, bot, BUILD / "shots", "mid_")
    acc = bot.engine.account
    bot.store.add_equity(acc.equity, acc.free_tmn, 0.0, acc.exposure, bot.risk.drawdown)
    return bot


def screenshots(cfg, bot: Bot, out: Path, prefix: str = "") -> list[str]:
    import uvicorn
    from playwright.sync_api import sync_playwright

    from pishro.web.app import create_app
    app = create_app(cfg, start_bot=False, bot_instance=bot)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=8765, log_level="warning"))
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    time.sleep(2.5)
    out.mkdir(parents=True, exist_ok=True)
    shots = []
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=_chrome())
        pg = b.new_page(viewport={"width": 1600, "height": 900}, device_scale_factor=1.2, color_scheme="dark")
        pg.goto("http://127.0.0.1:8765/", wait_until="networkidle")
        time.sleep(2.0)
        pg.screenshot(path=str(out / f"{prefix}dashboard_full.png"), full_page=True)
        pg.screenshot(path=str(out / f"{prefix}dashboard_top.png"))
        shots += [f"{prefix}dashboard_full.png", f"{prefix}dashboard_top.png"]
        pg.goto("http://127.0.0.1:8765/report", wait_until="networkidle")
        time.sleep(1.0)
        pg.screenshot(path=str(out / f"{prefix}report_full.png"), full_page=True)
        shots.append(f"{prefix}report_full.png")
        b.close()
    server.should_exit = True
    th.join(timeout=5)
    return shots


def _chrome() -> str | None:
    import glob
    c = sorted(glob.glob("/opt/pw-browsers/chromium-*/chrome-linux/chrome"))
    return c[-1] if c else None


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for n in ("httpx", "pishro.backtest.data", "pishro.risk"):
        logging.getLogger(n).setLevel(logging.WARNING)
    shutil.rmtree(DEMO, ignore_errors=True)
    full, cut, t0 = build_history()
    end_iso = dt.datetime.fromtimestamp(cut + 72 * 3600, dt.timezone.utc).isoformat()
    cfg = load_config(overrides={"data_dir": str(DEMO), "challenge.end": end_iso, "ai.enabled": False,
                                 "universe.min_volume_24h_tmn": 0, "references.max_ref_disagreement": 1.0,
                                 "universe.max_spread_pct": 0.05})
    log.info("walk-forward backtest on 24 synthetic days")
    runner.run_all(cfg, subtitle="⚠️ داده‌ی مصنوعی (نسخه‌ی نمایشی) — فقط برای نمایش عملکرد کد")
    real_time = time.time
    clock = Clock(cut)
    time.time = clock
    try:
        log.info("driving the paper bot through a simulated 3-day challenge")
        bot = asyncio.run(drive_bot(cfg, full, cut, 72, clock))
    finally:
        time.time = real_time
    log.info("trades: %s", bot.store.query("SELECT count(*) n, sum(pnl_tmn) pnl FROM trades"))
    bot.status_text = "running"                 # the demo clock ended; present the state as live
    bot.market.last_ok = time.time()
    bot.policy.ts = time.time()
    shots = screenshots(cfg, bot, BUILD / "shots")
    (BUILD / "demo_meta.json").write_text(json.dumps({"data_dir": str(DEMO), "start": cut, "end": cut + 72 * 3600,
                                                      "shots": shots}, ensure_ascii=False))
    log.info("demo data ready in %s", DEMO)


if __name__ == "__main__":
    main()
