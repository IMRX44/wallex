"""Paper broker for shadow trading: real live Wallex prices, simulated money.

Fill model (deliberately conservative so shadow results are not flattering):
  * Marketable orders fill immediately at the opposite best price as taker; if the order
    is larger than the displayed best-level size a slippage penalty is applied.
  * Resting orders fill as maker only when the market trades *through* them: the best
    ask drops to/below our bid (or best bid rises to/above our ask), or a public trade
    prints strictly beyond our price. Queue position is ignored, which can only
    under-count maker fills.
  * Fees: buys pay the fee in the base asset, sells in the quote asset (as on Wallex).
"""
from __future__ import annotations

import time
from typing import Callable

from ..types import BUY, MARKET, MarketInfo, Order, Quote
from .base import Broker

SLIPPAGE_OVERSIZE = 0.0008


class PaperBroker(Broker):
    name = "paper"

    def __init__(self, initial: dict[str, float], quotes: Callable[[], dict[str, Quote]],
                 infos: Callable[[], dict[str, MarketInfo]], fallback_maker: float, fallback_taker: float):
        super().__init__(fallback_maker, fallback_taker)
        self.bal: dict[str, list[float]] = {k: [float(v), 0.0] for k, v in initial.items()}
        self._quotes = quotes
        self._infos = infos
        self.orders: dict[str, Order] = {}

    async def balances(self, refresh: bool = False) -> dict[str, tuple[float, float]]:
        return {k: (v[0], v[1]) for k, v in self.bal.items() if v[0] > 1e-12 or v[1] > 1e-12}

    def _b(self, asset: str) -> list[float]:
        return self.bal.setdefault(asset, [0.0, 0.0])

    def _assets(self, symbol: str) -> tuple[str, str]:
        info = self._infos().get(symbol)
        if info:
            return info.base, info.quote
        for q in ("USDT", "TMN"):
            if symbol.endswith(q):
                return symbol[: -len(q)], q
        raise ValueError(f"unknown symbol {symbol}")

    async def place(self, order: Order) -> Order:
        t0 = time.time()
        base, quote = self._assets(order.symbol)
        q = self._quotes().get(order.symbol)
        if q is None or not q.valid:
            order.status = "REJECTED"
            return order
        if order.type == MARKET:
            order.price = q.ask * 1.02 if order.side == BUY else q.bid * 0.98
        if order.side == BUY:
            need = order.price * order.qty
            b = self._b(quote)
            if b[0] - b[1] < need - 1e-6:
                order.status = "REJECTED"
                return order
            b[1] += need
        else:
            b = self._b(base)
            if b[0] - b[1] < order.qty - 1e-12:
                order.status = "REJECTED"
                return order
            b[1] += order.qty
        order.status = "NEW"
        order.maker = True
        self.orders[order.client_id] = order
        marketable = (order.side == BUY and order.price >= q.ask) or (order.side != BUY and order.price <= q.bid)
        if marketable or order.type == MARKET:
            self._fill(order, q, taker=True)
        order.latency_ms = (time.time() - t0) * 1000
        return order

    def _fill(self, order: Order, q: Quote | None, taker: bool, price: float | None = None) -> None:
        base, quote = self._assets(order.symbol)
        maker_fee, taker_fee = self.fee(order.symbol)
        fee = taker_fee if taker else maker_fee
        qty = order.remaining
        if taker and q is not None:
            px = q.ask if order.side == BUY else q.bid
            shown = q.ask_vol if order.side == BUY else q.bid_vol
            if shown > 0 and qty > shown:
                px *= (1 + SLIPPAGE_OVERSIZE) if order.side == BUY else (1 - SLIPPAGE_OVERSIZE)
            px = min(px, order.price) if order.side == BUY else max(px, order.price)
        else:
            px = price or order.price
        if order.side == BUY:
            qb = self._b(quote)
            qb[1] -= order.price * qty          # release the reservation
            qb[0] -= px * qty
            self._b(base)[0] += qty * (1 - fee)
            order.fee_base += qty * fee
        else:
            bb = self._b(base)
            bb[1] -= qty
            bb[0] -= qty
            self._b(quote)[0] += px * qty * (1 - fee)
            order.fee_quote += px * qty * fee
        prev = order.filled_qty
        order.filled_qty += qty
        order.avg_price = (order.avg_price * prev + px * qty) / order.filled_qty
        order.status = "FILLED"
        order.maker = not taker
        order.fee_reported = True
        order.updated_ts = time.time()

    async def cancel(self, order: Order) -> Order:
        o = self.orders.get(order.client_id, order)
        if o.is_open:
            base, quote = self._assets(o.symbol)
            if o.side == BUY:
                self._b(quote)[1] -= o.price * o.remaining
            else:
                self._b(base)[1] -= o.remaining
            o.status = "CANCELED"
            o.updated_ts = time.time()
        self.orders.pop(o.client_id, None)
        return o

    async def refresh(self, order: Order) -> Order:
        o = self.orders.get(order.client_id, order)
        if o.is_open:
            q = self._quotes().get(o.symbol)
            if q is not None and q.valid:
                if (o.side == BUY and q.ask <= o.price) or (o.side != BUY and q.bid >= o.price):
                    self._fill(o, None, taker=False)
        if not o.is_open:
            self.orders.pop(o.client_id, None)
        return o

    def on_trade(self, symbol: str, price: float) -> None:
        """A public trade printed at ``price``: fill resting orders it traded through."""
        for o in list(self.orders.values()):
            if o.symbol != symbol or not o.is_open:
                continue
            if (o.side == BUY and price < o.price) or (o.side != BUY and price > o.price):
                self._fill(o, None, taker=False)
