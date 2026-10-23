"""Regressions found on the first run from an Iranian IP (2026-10-23)."""
import asyncio
import datetime as dt
import json

import httpx

from pishro.exchange.binance import BinanceRef
from pishro.store import Store


def test_store_serializes_datetimes(tmp_path):
    st = Store(tmp_path / "t.db")
    end = dt.datetime(2026, 10, 26, 23, 50, tzinfo=dt.timezone(dt.timedelta(hours=3, minutes=30)))
    st.add_signal("start", "", {"challenge_end": end, "x": 1}, None, "start", "ok")
    row = st.query("SELECT features FROM signals")[0]
    assert json.loads(row["features"])["challenge_end"].startswith("2026-10-26")


def test_reference_falls_back_and_pages_500():
    """binance.vision answers 451 from Iran; MEXC serves the same API but 500 klines per page."""
    start = 1_700_000_000_000

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.host == "blocked.example":
            return httpx.Response(451)
        if req.url.path.endswith("/klines"):
            s = int(req.url.params["startTime"])
            e = int(req.url.params["endTime"])
            bars = [[t, "1", "1", "1", "1", "1"] for t in range(s, min(e, s + 500 * 60_000), 60_000)]
            return httpx.Response(200, json=bars)
        if req.url.path.endswith("/exchangeInfo"):
            return httpx.Response(200, json={"symbols": [{"baseAsset": "BTC", "quoteAsset": "USDT", "status": "1"}]})
        return httpx.Response(404)

    async def run():
        bn = BinanceRef("https://blocked.example", fallback_rest=["https://mexc.example"])
        bn._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        syms = await bn.load_symbols()
        k = await bn.klines("BTC", "1m", start, start + 1200 * 60_000)
        await bn.close()
        return syms, k, bn.rest

    syms, k, rest = asyncio.run(run())
    assert syms == {"BTC"}
    assert len(k) == 1200
    assert rest == "https://mexc.example"


# ---------------------------------------------------------------- 2026-10-23 (part 2)
from types import SimpleNamespace  # noqa: E402

from pishro.engine.reversion import needs_requote, select_resting  # noqa: E402
from pishro.exchange.wallex import parse_order  # noqa: E402


def _c(sym, score):
    return {"symbol": sym, "score": score}


def test_resting_hysteresis_keeps_incumbents():
    """Paper run: 195 of 203 resting bids were cancelled only because the ranking jittered."""
    now = 1000.0
    resting = {"A": SimpleNamespace(since=now - 30, placed=now - 30), "B": SimpleNamespace(since=now - 200, placed=now - 5)}
    # C ranks above B by 10%: not enough to displace it
    want, swaps = select_resting([_c("C", 1.10), _c("B", 1.0), _c("A", 0.9)], resting, 2, now, 90, 1.25)
    assert set(want) == {"A", "B"} and not swaps
    # C is 30% better than the weakest incumbent, but that one (A) is young -> kept
    want, swaps = select_resting([_c("C", 1.30), _c("B", 1.2), _c("A", 1.0)], resting, 2, now, 90, 1.25)
    assert set(want) == {"A", "B"}
    # ... once A is old enough it is displaced
    resting["A"].since = now - 120
    want, swaps = select_resting([_c("C", 1.30), _c("B", 1.2), _c("A", 1.0)], resting, 2, now, 90, 1.25)
    assert set(want) == {"B", "C"} and swaps == {"A": "C"}
    # an incumbent that lost eligibility (gate) is not kept; a free slot is filled at once
    want, _ = select_resting([_c("C", 0.5)], resting, 2, now, 90, 1.25)
    assert set(want) == {"C"}


def test_requote_rules():
    assert needs_requote(101.0, 100.0, 1, 5e-4, 90)        # fair value fell 1%: move down at once
    assert not needs_requote(100.1, 100.0, 5, 5e-4, 90)     # 0.1% above: wait (no cancel/replace every tick)
    assert needs_requote(100.1, 100.0, 25, 5e-4, 90)        # ... but not forever
    assert not needs_requote(100.0, 100.02, 500, 5e-4, 90)  # tiny move: stay
    assert not needs_requote(100.0, 101.0, 30, 5e-4, 90)    # higher level, but too young
    assert needs_requote(100.0, 101.0, 120, 5e-4, 90)


def test_fee_in_toman_is_not_zero_fee_and_missing_fee_is_not_reported():
    r = {"symbol": "BTCTMN", "side": "BUY", "type": "LIMIT", "price": "100", "origQty": "2", "executedQty": "2",
         "executedSum": "200", "status": "FILLED", "fills": [{"fee": "0.5", "feeAsset": "TMN"}]}
    o = parse_order(r)
    assert o.fee_reported and o.fee_quote == 0.5 and o.fee_base == 0.0
    r["fills"] = [{"price": "100", "quantity": "2"}]
    assert not parse_order(r).fee_reported


def test_model_auc_guard():
    from pishro.bot import model_usable
    cfg = SimpleNamespace(get=lambda k, d=None: {"ml.min_auc": 0.55}.get(k, d))
    assert not model_usable(SimpleNamespace(report=SimpleNamespace(auc=0.50)), cfg)
    assert not model_usable(SimpleNamespace(report=SimpleNamespace(auc=float("nan"))), cfg)
    assert model_usable(SimpleNamespace(report=SimpleNamespace(auc=0.58)), cfg)


def test_live_start_cancels_only_own_stale_orders(tmp_path):
    from pishro.bot import Bot

    class FakeClient:
        def __init__(self):
            self.cancelled = []

        async def open_orders(self):
            return [{"clientOrderId": "PRREVABC", "symbol": "BTCTMN", "side": "BUY", "price": "1", "origQty": "1"},
                    {"clientOrderId": "someone-else", "symbol": "ETHTMN"}]

        async def cancel_order(self, cid):
            self.cancelled.append(cid)

    class FakeBroker:
        async def balances(self, refresh=False):
            return {}

    logged = []
    bot = Bot.__new__(Bot)
    bot.client, bot.broker = FakeClient(), FakeBroker()
    bot.dlog = SimpleNamespace(log=lambda kind, *a, **k: logged.append(kind))
    asyncio.run(bot._cancel_stale_orders())
    assert bot.client.cancelled == ["PRREVABC"] and logged == ["stale_order_cancel"]


def test_trading_capital_cap_reserves_the_rest(tmp_path):
    """First live test runs on 1M of the 5M account; limits apply to the 1M."""
    from pishro.bot import Bot

    store = Store(tmp_path / "c.db")
    cfgd = {"trading_capital_tmn": 1_000_000}
    bot = Bot.__new__(Bot)
    bot.cfg = SimpleNamespace(get=lambda k, d=None: cfgd.get(k, d))
    bot.store, bot._cap_state, bot.capital = store, None, 5e6
    bot.risk = SimpleNamespace(peak=5e6, day_start=5e6)
    logged = []
    bot.dlog = SimpleNamespace(log=lambda kind, *a, **k: logged.append(kind))
    assert bot._capital_reserve(5_000_000) == 4_000_000
    assert bot.risk.peak == 1_000_000 and bot.risk.day_start == 1_000_000 and bot.capital == 1_000_000
    # trading P&L does not move the reserve
    assert bot._capital_reserve(5_030_000) == 4_000_000
    # raising the cap later shrinks the reserve and restarts the high-water mark
    cfgd["trading_capital_tmn"] = 5_000_000
    assert bot._capital_reserve(5_030_000) == 30_000
    assert bot.risk.peak == 5_000_000 and logged == ["capital_cap", "capital_cap"]
    # persisted across restarts
    bot2 = Bot.__new__(Bot)
    bot2.cfg, bot2.store, bot2._cap_state, bot2.capital, bot2.risk, bot2.dlog = bot.cfg, store, None, 0, bot.risk, bot.dlog
    assert bot2._capital_reserve(5_100_000) == 30_000


def test_klines_skip_blocked_page_and_never_thrash():
    """On Liara MEXC answered 451 to some kline pages: the history got truncated and then forward-filled."""
    start = 1_700_000_000_000
    calls = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.host == "blocked.example":
            return httpx.Response(451)
        s = int(req.url.params["startTime"])
        e = int(req.url.params["endTime"])
        calls["n"] += 1
        if start + 500 * 60_000 <= s < start + 1000 * 60_000:      # the second page is always blocked
            return httpx.Response(451)
        return httpx.Response(200, json=[[t, "1", "1", "1", "1", "1"] for t in range(s, min(e, s + 500 * 60_000), 60_000)])

    async def run():
        bn = BinanceRef("https://blocked.example", fallback_rest=["https://mexc.example"])
        bn._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        import pishro.exchange.binance as m
        orig = m.asyncio.sleep

        async def fast(_):
            return None
        m.asyncio.sleep = fast
        try:
            k = await bn.klines("BTC", "1m", start, start + 1500 * 60_000)
        finally:
            m.asyncio.sleep = orig
            await bn.close()
        return k, bn.rest

    k, rest = asyncio.run(run())
    ts = [x[0] for x in k]
    assert len(k) == 1000 and ts[0] == start and ts[-1] == start + 1499 * 60_000   # pages 1 and 3 kept, gap stays
    assert rest == "https://mexc.example"



def test_llm_falls_back_when_balance_refused():
    """Live 18:32: Liara refused Opus with 402 (balance below the model's minimum)."""
    from pishro.ai.llm import LLMClient

    llm = LLMClient("http://x", "k", "anthropic/claude-opus-5.5", fallback_model="anthropic/claude-sonnet-5")
    used = []

    async def fake(messages, tools=None, temperature=0.2):
        used.append(llm.model)
        if llm.model.endswith("opus-5.5"):
            raise RuntimeError("Error code: 402 - Your balance must be higher than 871968 tomans")
        return SimpleNamespace(content="ok")

    llm._chat = fake
    assert asyncio.run(llm.chat([{"role": "user", "content": "hi"}])).content == "ok"
    assert used == ["anthropic/claude-opus-5.5", "anthropic/claude-sonnet-5"]
    assert llm.model == "anthropic/claude-sonnet-5"      # stays on the fallback for a while
