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
