"""End-to-end: the real Bot in paper mode on synthetic data with a fake clock."""
import asyncio
import datetime as dt
import json
import time

from pishro.backtest import data, synthetic
from pishro.bot import Bot
from pishro.config import load_config

from .fakes import Clock, FakeBinance, FakeWallex, drive, install


def test_paper_bot_end_to_end(tmp_path, monkeypatch):
    cfg0 = load_config(overrides={"data_dir": str(tmp_path)})
    synthetic.generate(cfg0.data_dir / "history", days=4, markets=5, seed=3)
    hist = data.load_history(cfg0)
    start = int(hist["usdt"].index[0]) + 2 * 86400 + 60
    hours = 14
    end_iso = dt.datetime.fromtimestamp(start + hours * 3600, dt.timezone.utc).isoformat()
    cfg = load_config(overrides={"data_dir": str(tmp_path), "challenge.end": end_iso, "ai.enabled": False,
                                 "universe.min_volume_24h_tmn": 0, "universe.max_spread_pct": 0.05,
                                 "references.max_ref_disagreement": 1.0, "ml.enabled": False,
                                 "strategy.warmup_minutes": 30})
    clock = Clock(start)
    monkeypatch.setattr(time, "time", clock)

    async def run():
        bot = Bot(cfg)
        wx, bn = FakeWallex(hist, clock), FakeBinance(hist, clock)
        install(bot, wx, bn)
        await bot.market.warmup(days=2)
        await bot.broker.start()
        assert len(bot.market.universe.tradable) == 5
        assert all(st.ready for st in bot.market.states.values())
        await drive(bot, wx, clock, minutes=hours * 60 + 5)
        return bot

    bot = asyncio.run(run())
    kinds = [json.loads(line)["kind"] for line in open(bot.dlog.path, encoding="utf-8")]
    assert "buy_order" in kinds and "buy_fill" in kinds and "sell_fill" in kinds, set(kinds)
    trades = bot.store.query("SELECT * FROM trades")
    assert len(trades) > 0
    # the challenge is scored in toman: nothing but toman may remain after the end
    bals = asyncio.run(bot.broker.balances())
    coins = {k: v for k, v in bals.items() if k != "TMN" and v[0] * 1 > 0}
    assert not bot.portfolio.positions, bot.portfolio.positions
    assert all(bot.market.to_tmn(k, v[0]) < 1000 for k, v in coins.items()), coins
    # every fill has a reason in the decision log
    for line in open(bot.dlog.path, encoding="utf-8"):
        rec = json.loads(line)
        if rec["kind"] in ("buy_order", "buy_fill", "sell_order", "sell_fill"):
            assert rec["reason"], rec
    # realized P&L in the trades table matches the account change
    realized = sum(t["pnl_tmn"] for t in trades)
    assert abs((bals["TMN"][0] - 5_000_000) - realized) < 5_000_000 * 0.002, (bals["TMN"][0], realized)
