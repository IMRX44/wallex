import asyncio
import json
import types

import pytest

from pishro.ai.policy import LIMITS, MarketPolicy, Policy
from pishro.broker.paper import PaperBroker
from pishro.config import load_config
from pishro.exchange.nobitex import parse_orderbook_all
from pishro.exchange.wallex import parse_depth, parse_fee_rate, parse_markets, parse_order
from pishro.market.fairvalue import compute_fair
from pishro.market.universe import Universe, load_excluded
from pishro.risk import RiskManager
from pishro.strategy.reversion import (Params, PremiumState, breakeven, entry_edge, exit_target, gates,
                                       size_from_prob)
from pishro.types import BUY, LIMIT, SELL, MarketInfo, Order, Quote

MARKETS = {"success": True, "result": {"symbols": {
    "BTCTMN": {"symbol": "BTCTMN", "baseAsset": "BTC", "quoteAsset": "TMN", "stepSize": 6, "tickSize": 0,
               "minQty": "0.000001", "minNotional": "100000", "faName": "بیت کوین", "isZeroFee": False,
               "stats": {"bidPrice": "9000000000", "askPrice": "9010000000", "bidVolume": "0.5", "askVolume": "0.4",
                         "lastPrice": "9005000000", "24h_quoteVolume": "50000000000", "24h_ch": -1.2}},
    "DOGETMN": {"symbol": "DOGETMN", "baseAsset": "DOGE", "quoteAsset": "TMN", "stepSize": 0, "tickSize": 1,
                "isZeroFee": True, "stats": {"bidPrice": "20000", "askPrice": "20020"}},
    "USDTTMN": {"symbol": "USDTTMN", "baseAsset": "USDT", "quoteAsset": "TMN", "stepSize": 2, "tickSize": 0,
                "stats": {"bidPrice": "100000", "askPrice": "100100", "24h_quoteVolume": "1e12"}},
    "ATMN": {"symbol": "ATMN", "baseAsset": "A", "quoteAsset": "TMN", "stats": {"bidPrice": "1", "askPrice": "1.01",
                                                                             "24h_quoteVolume": "1e12"}},
}}}


def test_parse_markets_and_rounding():
    infos, quotes = parse_markets(MARKETS)
    btc = infos["BTCTMN"]
    assert btc.step == 1e-6 and btc.tick == 1.0 and btc.min_notional == 100000
    assert infos["DOGETMN"].zero_fee and infos["DOGETMN"].tick == 0.1
    assert btc.round_price(9000000000.7, BUY) == 9000000000 and btc.round_price(9000000000.2, SELL) == 9000000001
    assert btc.round_qty(0.1234567) == pytest.approx(0.123456)
    q = quotes["BTCTMN"]
    assert q.valid and q.spread_pct == pytest.approx(10e6 / 9005e6)


def test_universe_excludes_hackathon_markets_and_usdt():
    infos, quotes = parse_markets(MARKETS)
    excluded = load_excluded(load_config().get("universe.excluded_file"))
    assert "ATMN" in excluded and "1BBABYDOGETMN" in excluded and "BTCTMN" not in excluded
    u = Universe(excluded, ["TMN"], 1e9, 0.02)
    tr = u.refresh(infos, quotes, 100050)
    assert tr == {"BTCTMN"}  # ATMN excluded, USDTTMN is the reference, DOGETMN has no volume


def test_parse_order_fills_and_fee():
    o = Order("cid", "BTCTMN", BUY, LIMIT, 1.0, 0.01)
    r = {"result": {"symbol": "BTCTMN", "side": "BUY", "type": "LIMIT", "price": "9000", "origQty": "0.01",
                    "executedQty": "0.01", "executedSum": "90", "status": "FILLED", "active": False,
                    "fills": [{"fee": "0.00002", "feeAsset": "BTC"}]}}
    o = parse_order(r, o)
    assert o.status == "FILLED" and o.avg_price == 9000 and o.fee_base == pytest.approx(0.00002) and o.fee_reported
    assert parse_fee_rate("0.25", 0.1) == 0.0025 and parse_fee_rate("0.0025", 0.1) == 0.0025


def test_depth_and_nobitex_parsers():
    d = parse_depth({"result": {"ask": [{"price": "11", "quantity": 1}, {"price": "10", "quantity": 2}],
                                "bid": [{"price": "8", "quantity": 1}, {"price": "9", "quantity": 3}]}})
    assert d["asks"][0] == (10, 2) and d["bids"][0] == (9, 3)
    nb = parse_orderbook_all({"status": "ok", "BTCIRT": {"bids": [["900", "1"]], "asks": [["1100", "1"]]},
                              "BTCUSDT": {"bids": [["9", "1"]], "asks": [["11", "1"]]}})
    assert nb["BTC"] == {"TMN": 100.0, "USDT": 10.0}


def test_fair_value_and_disagreement():
    quotes = {"USDTTMN": Quote("USDTTMN", 100_000, 100_000), "BTCUSDT": Quote("BTCUSDT", 100, 100)}
    fv = compute_fair("BTC", quotes, {"BTC": 101.0})
    assert fv.source == "binance" and fv.fair == pytest.approx(101.0 * 100_000)
    assert fv.disagreement == pytest.approx(1 / 101, rel=1e-3) and fv.ok(0.012)
    fv2 = compute_fair("BTC", quotes, {})
    assert fv2.source == "wallex_usdt"
    assert compute_fair("XYZ", quotes, {}) is None


def _state(n=200, dev=0.01, noise=0.0):
    st = PremiumState("XTMN", Params(warmup_minutes=50))
    for i in range(n):
        st.on_bar(i * 60, 100 * (1 + dev + (noise if i % 2 else -noise)), 99, 101, 100, 1e6, 1e5)
    return st


def test_premium_state_learns_baseline():
    st = _state(dev=0.01, noise=0.002)
    assert st.ready and st.baseline == pytest.approx(0.01, abs=1e-3)
    assert 0.001 < st.std < 0.004
    assert st.fair_adj(100) == pytest.approx(101, rel=1e-3)
    f = st.features(1_700_000_000, 0.005, -0.004)
    assert set(f) >= {"edge", "std", "revert_rate", "ref_ret_5"}


def test_entry_edge_respects_fees_and_noise():
    st = _state(noise=0.0)
    p = Params()
    assert entry_edge(st, 0.0, 0.0, p) == pytest.approx(p.edge_floor_zero_fee)
    with_fee = entry_edge(st, 0.0025, 0.003, p)
    assert with_fee >= 1.6 * 0.005 + p.min_profit - 1e-12
    assert entry_edge(st, 0.0025, 0.003, p, taker=True) > with_fee
    assert entry_edge(st, 0.0, 0.0, p, agent_mult=0.1) == pytest.approx(p.edge_floor_zero_fee * 0.8)  # clamp 0.8
    noisy = _state(noise=0.01)
    assert entry_edge(noisy, 0.0, 0.0, p) > p.edge_floor_zero_fee


def test_exit_never_below_breakeven():
    st = _state(dev=-0.05)
    p = Params(min_profit=0.001)
    be = breakeven(100, 0.0025, 0.0025)
    assert be == pytest.approx(100 / (0.9975 ** 2))
    assert exit_target(st, 100, entry_price=100, buy_fee=0.0025, sell_fee=0.0025, p=p) >= be * 1.001 - 1e-9


def test_gates_block_crash_and_regime():
    st = _state()
    p = Params(warmup_minutes=50)
    assert gates(st, p) == ""
    for k in range(5):                       # Binance falls 1% in 5 minutes -> falling knife
        st.on_bar(20000 + k * 60, 101, 100, 102, 100 * (1 - 0.002 * (k + 1)), 1e6, 1e5)
    assert gates(st, p) == "shock_ref"


def test_size_from_prob():
    assert size_from_prob(None, 0.55, 0.72) == 1.0
    assert size_from_prob(0.5, 0.55, 0.72) == 0.0
    assert size_from_prob(0.55, 0.55, 0.72) == pytest.approx(0.5)
    assert size_from_prob(0.9, 0.55, 0.72) == 1.0


def test_policy_clamps_and_decay():
    p = Policy(mode="weird", size_mult=5, edge_mult=0.1, exit_edge=1,
               markets={"BTCTMN": MarketPolicy(allow=True, edge_mult=10, size_mult=3)}).clamp()
    assert p.mode == "normal" and p.size_mult == 1.0 and p.edge_mult == LIMITS["edge_mult"][0]
    assert p.exit_edge == LIMITS["exit_edge"][1]
    allowed, e, s = p.for_market("BTCTMN")
    assert allowed and s <= 1.0 and e <= LIMITS["edge_mult"][1]
    p.source, p.ts = "agent", 0
    eff = p.effective(60, now=1000)
    assert eff.mode == "defensive" and eff.source == "fallback"
    assert not Policy(mode="halt_entries").for_market("X")[0]
    rt = Policy.from_dict(json.loads(json.dumps(p.to_dict())))
    assert rt.markets["BTCTMN"].edge_mult == p.markets["BTCTMN"].edge_mult


def test_risk_manager_limits():
    cfg = load_config()
    r = RiskManager(cfg)
    r.update_equity(5_000_000, now=1_700_000_000)
    d = r.check_open(notional=2e6, equity=5e6, exposure_total=0, exposure_base=0, n_open=0, free_quote_tmn=5e6,
                     data_age=1, api_errors=0)
    assert d.ok and d.max_notional <= 5e6 * r.max_position
    d = r.check_open(notional=2e6, equity=5e6, exposure_total=0, exposure_base=0, n_open=0, free_quote_tmn=1.05e6,
                     data_age=1, api_errors=0)
    assert not d.ok  # would break the 20% cash reserve
    assert not r.check_open(notional=1e6, equity=5e6, exposure_total=0, exposure_base=0, n_open=0,
                            free_quote_tmn=5e6, data_age=999, api_errors=0).ok
    assert r.size(5e6, 100, 97.5) == pytest.approx(min(5e6 * r.risk_per_trade / 0.025, 5e6 * r.max_position))
    r.update_equity(5e6 * (1 - r.max_dd - 0.001), now=1_700_000_100)
    assert r.halted and r.flatten_requested and not r.trading_allowed()[0]


def test_risk_consecutive_losses_pause():
    r = RiskManager(load_config())
    for _ in range(r.max_consec):
        r.on_trade_closed(-0.01, now=1000)
    ok, why = r.trading_allowed(now=1001)
    assert not ok and r.trading_allowed(now=1000 + r.cooldown + 1)[0]


def test_paper_broker_fills_and_fees():
    info = {"XTMN": MarketInfo("XTMN", "X", "TMN")}
    quotes = {"XTMN": Quote("XTMN", 99, 101, bid_vol=100, ask_vol=100)}
    b = PaperBroker({"TMN": 10_000}, lambda: quotes, lambda: info, 0.0025, 0.003)

    async def go():
        o = await b.place(Order("a", "XTMN", BUY, LIMIT, 100, 50))   # resting maker bid
        assert o.is_open and b.bal["TMN"][1] == pytest.approx(5000)
        quotes["XTMN"] = Quote("XTMN", 98, 100)                      # ask trades down to our bid
        o = await b.refresh(o)
        assert o.status == "FILLED" and o.maker and b.bal["X"][0] == pytest.approx(50 * (1 - 0.0025))
        s = await b.place(Order("b", "XTMN", SELL, LIMIT, 97, 20))   # marketable sell -> taker at bid
        assert s.status == "FILLED" and not s.maker and s.avg_price == 98
        assert b.bal["TMN"][0] == pytest.approx(10_000 - 5000 + 98 * 20 * (1 - 0.003))
        c = await b.place(Order("c", "XTMN", BUY, LIMIT, 50, 10))
        await b.cancel(c)
        assert b.bal["TMN"][1] == pytest.approx(0)
        assert (await b.place(Order("d", "XTMN", BUY, LIMIT, 100, 1000))).status == "REJECTED"
    asyncio.run(go())


def test_agent_tool_loop_with_fake_llm():
    from pishro.ai.agent import TradingAgent

    class Call:
        def __init__(self, i, name, args):
            self.id = f"c{i}"
            self.function = types.SimpleNamespace(name=name, arguments=json.dumps(args))

    script = [
        [Call(1, "get_portfolio_status", {}), Call(2, "get_opportunities", {"limit": 5})],
        [Call(3, "set_market_policy", {"symbol": "BTCTMN", "allow": False, "reason": "dips stopped reverting"}),
         Call(4, "set_risk_mode", {"mode": "defensive", "size_mult": 7, "reason": "volatile"})],
        [Call(5, "finish", {"summary_fa": "بازار پرنوسان است؛ BTC مسدود شد", "outlook": "risky"})],
    ]

    class FakeLLM:
        tools_supported = True

        def __init__(self):
            self.i = 0

        async def chat(self, messages, tools=None):
            calls = script[self.i]
            self.i += 1
            return types.SimpleNamespace(content="", tool_calls=calls)

    applied = {}

    class FakeBot:
        policy = Policy()
        market = types.SimpleNamespace(universe=types.SimpleNamespace(infos={"BTCTMN": 1}))
        dlog = types.SimpleNamespace(log=lambda *a, **k: None)

        def status(self):
            return {"equity_tmn": 5e6}

        def opportunities(self, n):
            return []

        def set_policy(self, p, **kw):
            applied["policy"], applied["kw"] = p, kw

    agent = TradingAgent(load_config(), FakeLLM(), FakeBot())
    asyncio.run(agent.run_once("test"))
    p = applied["policy"]
    assert p.mode == "defensive" and p.size_mult == 1.0          # 7 clamped to 1
    assert p.markets["BTCTMN"].allow is False and p.source == "agent"
    assert "BTC" in p.rationale and len(applied["kw"]["actions"]) == 2
