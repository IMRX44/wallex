"""Async client for the Wallex REST API (https://api-docs.wallex.ir).

Public endpoints: markets, depth, udf/history, currencies/stats.
Private endpoints (header ``x-api-key``): balances, fee, orders.

The parsers are deliberately defensive: Wallex returns most numbers as strings and a
few fields changed shape over the years, so every value goes through ``_f``.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import httpx

from ..types import BUY, LIMIT, MarketInfo, Order, Quote

log = logging.getLogger(__name__)


class WallexError(Exception):
    def __init__(self, message: str, status: int = 0, payload: Any = None):
        super().__init__(message)
        self.status = status
        self.payload = payload


def _f(v: Any, default: float = 0.0) -> float:
    try:
        if v is None or v == "":
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def _increment(v: Any) -> float:
    """Wallex reports stepSize/tickSize as a number of decimals (e.g. 6 -> 1e-6).

    Some responses use the increment itself (e.g. 0.01); accept both.
    """
    x = _f(v, -1)
    if x < 0:
        return 0.0
    if x >= 1 and float(x).is_integer() and x <= 18:
        return 10 ** (-int(x))
    if x == 0:
        return 1.0
    return x


def split_symbol(symbol: str, quotes: tuple[str, ...] = ("USDT", "TMN")) -> tuple[str, str]:
    for q in quotes:
        if symbol.endswith(q) and len(symbol) > len(q):
            return symbol[: -len(q)], q
    return symbol, ""


def parse_markets(payload: dict) -> tuple[dict[str, MarketInfo], dict[str, Quote]]:
    result = payload.get("result", payload)
    symbols = result.get("symbols", result) if isinstance(result, dict) else {}
    if isinstance(symbols, list):
        symbols = {s.get("symbol"): s for s in symbols if isinstance(s, dict)}
    infos: dict[str, MarketInfo] = {}
    quotes: dict[str, Quote] = {}
    ts = time.time()
    for sym, m in symbols.items():
        if not isinstance(m, dict):
            continue
        sym = (m.get("symbol") or sym or "").upper()
        base = (m.get("baseAsset") or "").upper()
        quote = (m.get("quoteAsset") or "").upper()
        if not base or not quote:
            base, quote = split_symbol(sym)
        infos[sym] = MarketInfo(
            symbol=sym,
            base=base,
            quote=quote,
            tick=_increment(m.get("tickSize")),
            step=_increment(m.get("stepSize")),
            min_qty=_f(m.get("minQty")),
            min_notional=_f(m.get("minNotional")),
            fa_name=m.get("faName") or "",
            zero_fee=bool(m.get("isZeroFee", False)),
        )
        st = m.get("stats") or {}
        q = Quote(
            symbol=sym,
            bid=_f(st.get("bidPrice")),
            ask=_f(st.get("askPrice")),
            bid_vol=_f(st.get("bidVolume")),
            ask_vol=_f(st.get("askVolume")),
            last=_f(st.get("lastPrice")),
            quote_volume_24h=_f(st.get("24h_quoteVolume")),
            change_24h=_f(st.get("24h_ch")),
            ts=ts,
        )
        quotes[sym] = q
    return infos, quotes


def parse_depth(payload: dict) -> dict[str, list[tuple[float, float]]]:
    r = payload.get("result", payload)

    def side(rows: Any) -> list[tuple[float, float]]:
        out = []
        for row in rows or []:
            if isinstance(row, dict):
                out.append((_f(row.get("price")), _f(row.get("quantity"))))
            elif isinstance(row, (list, tuple)) and len(row) >= 2:
                out.append((_f(row[0]), _f(row[1])))
        return [x for x in out if x[0] > 0 and x[1] > 0]

    bids = sorted(side(r.get("bid") or r.get("bids")), key=lambda x: -x[0])
    asks = sorted(side(r.get("ask") or r.get("asks")), key=lambda x: x[0])
    return {"bids": bids, "asks": asks}


def parse_fee_rate(v: Any, fallback: float) -> float:
    x = _f(v, -1)
    if x < 0:
        return fallback
    # Wallex shows fees as percentages ("0.25"); anything above 5% must be a percent.
    return x / 100 if x > 0.05 else x


def parse_order(r: dict, fallback: Order | None = None) -> Order:
    r = r.get("result", r) if isinstance(r, dict) else {}
    o = fallback or Order(client_id="", symbol="", side=BUY, type=LIMIT, price=0, qty=0)
    o.client_id = r.get("clientOrderId") or r.get("client_id") or o.client_id
    o.symbol = (r.get("symbol") or o.symbol).upper()
    o.side = (r.get("side") or o.side).upper()
    o.type = (r.get("type") or o.type).upper()
    o.price = _f(r.get("price"), o.price)
    o.qty = _f(r.get("origQty"), o.qty)
    o.filled_qty = _f(r.get("executedQty"), o.filled_qty)
    exec_sum = _f(r.get("executedSum"))
    if o.filled_qty > 0:
        o.avg_price = exec_sum / o.filled_qty if exec_sum > 0 else _f(r.get("executedPrice"), o.price)
    fills = r.get("fills")
    if isinstance(fills, list) and fills:
        quote = o.symbol[-4:] if o.symbol.endswith("USDT") else o.symbol[-3:]
        o.fee_quote = sum(_f(x.get("fee")) for x in fills if (x.get("feeAsset") or "").upper() == quote)
        o.fee_base = sum(_f(x.get("fee")) for x in fills if (x.get("feeAsset") or "").upper() not in (quote, ""))
        o.fee_reported = True
    status = (r.get("status") or o.status).upper()
    if status in ("CANCELLED",):
        status = "CANCELED"
    if r.get("active") is False and status in ("NEW", "PARTIALLY_FILLED"):
        status = "FILLED" if o.filled_qty >= o.qty * 0.999 else "CANCELED"
    o.status = status
    o.updated_ts = time.time()
    return o


class WallexClient:
    def __init__(self, base_url: str = "https://api.wallex.ir", api_key: str = "",
                 timeout: float = 10.0, max_retries: int = 3):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.max_retries = max_retries
        self._client = httpx.AsyncClient(timeout=timeout, headers={"User-Agent": "pishro-bot/1.0"})
        self.error_times: list[float] = []

    async def close(self) -> None:
        await self._client.aclose()

    def errors_last_minute(self) -> int:
        cutoff = time.time() - 60
        self.error_times = [t for t in self.error_times if t > cutoff]
        return len(self.error_times)

    async def _request(self, method: str, path: str, signed: bool = False, **kw) -> dict:
        url = f"{self.base_url}/{path.lstrip('/')}"
        headers = kw.pop("headers", {})
        if signed:
            if not self.api_key:
                raise WallexError("WALLEX_API_KEY is not set")
            headers["x-api-key"] = self.api_key
        delay = 0.5
        for attempt in range(self.max_retries + 1):
            try:
                resp = await self._client.request(method, url, headers=headers, **kw)
            except httpx.HTTPError as e:
                self.error_times.append(time.time())
                if attempt >= self.max_retries:
                    raise WallexError(f"network error: {e}") from e
                await asyncio.sleep(delay)
                delay *= 2
                continue
            if resp.status_code == 429 or resp.status_code >= 500:
                self.error_times.append(time.time())
                if attempt >= self.max_retries:
                    raise WallexError(f"HTTP {resp.status_code}", resp.status_code, resp.text[:300])
                await asyncio.sleep(delay)
                delay *= 2
                continue
            try:
                data = resp.json()
            except ValueError:
                self.error_times.append(time.time())
                raise WallexError(f"invalid JSON (HTTP {resp.status_code})", resp.status_code, resp.text[:300]) from None
            if resp.status_code >= 400 or (isinstance(data, dict) and data.get("success") is False):
                self.error_times.append(time.time())
                msg = data.get("message") if isinstance(data, dict) else str(data)
                raise WallexError(f"Wallex error: {msg}", resp.status_code, data)
            return data
        raise WallexError("unreachable")

    # ---------- public ----------
    async def markets(self) -> tuple[dict[str, MarketInfo], dict[str, Quote]]:
        return parse_markets(await self._request("GET", "v1/markets"))

    async def depth(self, symbol: str) -> dict[str, list[tuple[float, float]]]:
        return parse_depth(await self._request("GET", "v1/depth", params={"symbol": symbol}))

    async def candles(self, symbol: str, resolution: str, frm: int, to: int) -> dict[str, list]:
        data = await self._request("GET", "v1/udf/history", params={
            "symbol": symbol, "resolution": resolution, "from": int(frm), "to": int(to)})
        if data.get("s") not in ("ok", None):
            return {"t": [], "o": [], "h": [], "l": [], "c": [], "v": []}
        return {k: [_f(x) for x in data.get(k, [])] for k in ("t", "o", "h", "l", "c", "v")}

    async def global_stats(self) -> dict[str, float]:
        """USD prices of coins as published by Wallex (global reference)."""
        data = await self._request("GET", "v1/currencies/stats")
        out: dict[str, float] = {}
        for row in data.get("result", []) or []:
            if isinstance(row, dict) and row.get("key"):
                p = _f(row.get("price"))
                if p > 0:
                    out[str(row["key"]).upper()] = p
        return out

    # ---------- private ----------
    async def balances(self) -> dict[str, tuple[float, float]]:
        """asset -> (total, locked)."""
        data = await self._request("GET", "v1/account/balances", signed=True)
        bal = (data.get("result") or {}).get("balances") or {}
        return {k.upper(): (_f(v.get("value")), _f(v.get("locked"))) for k, v in bal.items() if isinstance(v, dict)}

    async def fees(self, fallback_maker: float, fallback_taker: float) -> dict[str, tuple[float, float]]:
        data = await self._request("GET", "v1/account/fee", signed=True)
        out = {}
        for sym, v in (data.get("result") or {}).items():
            if isinstance(v, dict):
                out[sym.upper()] = (parse_fee_rate(v.get("makerFeeRate"), fallback_maker),
                                    parse_fee_rate(v.get("takerFeeRate"), fallback_taker))
        return out

    async def place_order(self, order: Order) -> Order:
        body: dict[str, Any] = {
            "symbol": order.symbol,
            "type": order.type,
            "side": order.side,
            "quantity": _fmt(order.qty),
            "client_id": order.client_id,
        }
        if order.type == LIMIT:
            body["price"] = _fmt(order.price)
        data = await self._request("POST", "v1/account/orders", signed=True, json=body)
        return parse_order(data, order)

    async def cancel_order(self, client_id: str) -> None:
        try:
            await self._request("DELETE", f"v1/account/orders/{client_id}", signed=True)
        except WallexError as e:
            if e.status not in (404, 405):
                raise
            await self._request("DELETE", "v1/account/orders", signed=True, json={"clientOrderId": client_id})

    async def get_order(self, order: Order) -> Order:
        data = await self._request("GET", f"v1/account/orders/{order.client_id}", signed=True)
        return parse_order(data, order)

    async def open_orders(self, symbol: str | None = None) -> list[dict]:
        params = {"symbol": symbol} if symbol else {}
        data = await self._request("GET", "v1/account/openOrders", signed=True, params=params)
        r = data.get("result") or {}
        orders = r.get("orders", r) if isinstance(r, dict) else r
        if isinstance(orders, dict):  # some responses group by symbol
            return [o for v in orders.values() if isinstance(v, list) for o in v]
        return orders or []


def _fmt(x: float) -> str:
    s = f"{x:.10f}".rstrip("0").rstrip(".")
    return s or "0"
